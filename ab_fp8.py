"""A/B harness: bf16 vs FP8-block-scaled training, same config and seed.

Trains the model from the same seed and data twice — once in bf16, once with
per-block-scaled FP8 simulation — and compares loss + grad-norm trajectories.

The FP8 path uses the precision-equivalent simulation from fp8.py (cast to fp8
with optimization barrier, then bf16 matmul). It executes in bf16, so each
step is SLOWER than a real FP8 matmul would be — but the precision matches.
The A/B answers "does FP8 hurt quality on this model?"; it does NOT measure
training throughput. For a throughput test, swap the simulation for
jax.nn.scaled_matmul in fp8.py.

Run:
    modal run ab_fp8.py
    modal run ab_fp8.py --n-steps 1000 --batch-size 16
    modal run ab_fp8.py --config full --n-steps 200 --batch-size 4

Output:
    - Per-step loss + grad-norm for both runs, saved to the trace volume and
      downloaded locally as ab_fp8_curves.json.
    - Side-by-side table at the end with key stats.
"""
import json
from pathlib import Path

import modal

TRACE_VOLUME_NAME = "train-gpt-traces"
REMOTE_TRACE_DIR = "/traces"
DATA_VOLUME_NAME = "fineweb-edu-data"
REMOTE_DATA_DIR = "/data"
REAL_TOKENS_FILE = "fineweb-edu-10BT-train.bin"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "jax[cuda13]==0.9.2",
        "optax==0.2.8",
        "numpy==2.4.5",
        "chex==0.1.91",
    )
    .add_local_python_source(
        "config", "model", "losses", "data", "kernels", "fp8",
    )
)

app = modal.App("train-gpt-ab-fp8")
trace_vol = modal.Volume.from_name(TRACE_VOLUME_NAME, create_if_missing=True)
data_vol = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    gpu="B200",
    memory=48 * 1024,
    timeout=60 * 60,
    volumes={REMOTE_TRACE_DIR: trace_vol, REMOTE_DATA_DIR: data_vol},
)
def run_ab(
    config: str = "small",
    batch_size: int = 8,
    n_steps: int = 500,
    lr: float = 1e-4,
    weight_decay: float = 0.1,
    seed: int = 0,
    log_every: int = 25,
    n_tokens: int = 4_000_000,
    use_real_data: bool = True,
    warmup_steps: int = 100,
    clip_norm: float = 1.0,
) -> dict:
    import os
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

    import dataclasses
    import math
    import time

    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax

    from config import SMALL, Config
    from data import batch_iter, synth_bin, load_tokens
    from losses import cross_entropy_loss
    from model import init_params, param_count

    base_cfg = SMALL if config == "small" else Config()

    # Real text data (FineWeb-Edu, GPT-2 BPE) lives on the data volume. We only
    # use it for FULL config — SMALL has vocab=256 and wouldn't be compatible
    # with the GPT-2 50257-vocab tokens.
    real_data_path = Path(REMOTE_DATA_DIR) / REAL_TOKENS_FILE
    if use_real_data and config != "small" and real_data_path.exists():
        data_path = real_data_path
        print(f"using real corpus: {data_path} ({data_path.stat().st_size:,} bytes)")
    else:
        data_path = Path("/tmp/ab_tokens.bin")
        if not data_path.exists():
            synth_bin(data_path, base_cfg.vocab_size, n_tokens=n_tokens, seed=seed)
        print(f"using synth corpus: {data_path}")
    tokens = load_tokens(data_path)

    def train_run(precision: str) -> dict:
        cfg = dataclasses.replace(base_cfg, matmul_precision=precision)
        key = jax.random.PRNGKey(seed)
        params = init_params(key, cfg)
        optimizer = optax.chain(
            optax.clip_by_global_norm(clip_norm),
            optax.adamw(learning_rate=lr, weight_decay=weight_decay),
        )
        opt_state = optimizer.init(params)

        @jax.jit
        def train_step(params, opt_state, inputs, targets):
            loss, grads = jax.value_and_grad(cross_entropy_loss)(
                params, inputs, targets, cfg
            )
            grad_norm = optax.global_norm(grads)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, grad_norm

        batches = batch_iter(tokens, batch_size, cfg.seq_len, seed=seed)
        tokens_per_step = batch_size * cfg.seq_len

        # Warm up (compile) — not counted in metrics.
        inputs, targets = next(batches)
        params, opt_state, loss, gn = train_step(params, opt_state, inputs, targets)
        jax.block_until_ready((loss, gn))
        warmup_loss = float(loss)
        print(f"  [{precision}] compile done, warmup loss={warmup_loss:.4f}")
        if not math.isfinite(warmup_loss):
            raise RuntimeError(
                f"{precision} warmup loss is non-finite ({warmup_loss}) — "
                "model/data setup is broken before training even starts"
            )

        losses, grad_norms = [], []
        t0 = time.perf_counter()
        last_log_t, last_log_step = t0, 0
        for step in range(1, n_steps + 1):
            inputs, targets = next(batches)
            params, opt_state, loss, gn = train_step(
                params, opt_state, inputs, targets
            )
            losses.append(float(loss))
            grad_norms.append(float(gn))
            if step % log_every == 0 or step == n_steps:
                jax.block_until_ready((loss, gn))
                now = time.perf_counter()
                tps = (step - last_log_step) * tokens_per_step / (now - last_log_t)
                print(f"  [{precision}] step {step:5d}  loss {losses[-1]:.4f}  "
                      f"grad_norm {grad_norms[-1]:.3f}  {tps:,.0f} tok/s")
                last_log_t, last_log_step = now, step

        total_dt = time.perf_counter() - t0
        return {
            "precision": precision,
            "losses": losses,
            "grad_norms": grad_norms,
            "wall_seconds": total_dt,
            "tokens_per_sec": n_steps * tokens_per_step / total_dt,
            "n_params": param_count(params),
        }

    print(f"\nA/B harness  config={config}  batch={batch_size}  "
          f"steps={n_steps}  seed={seed}")
    print(f"device: {jax.devices()[0]}")
    print(f"\n=== run 1: bf16 ===")
    bf16 = train_run("bf16")
    print(f"\n=== run 2: fp8_block ===")
    fp8 = train_run("fp8_block")

    out = {
        "config": config,
        "batch_size": batch_size,
        "n_steps": n_steps,
        "seed": seed,
        "lr": lr,
        "bf16": bf16,
        "fp8_block": fp8,
    }
    out_path = Path(REMOTE_TRACE_DIR) / "ab_fp8_curves.json"
    out_path.write_text(json.dumps(out))
    trace_vol.commit()
    return out


@app.local_entrypoint()
def main(
    config: str = "small",
    batch_size: int = 8,
    n_steps: int = 500,
    lr: float = 1e-4,
    weight_decay: float = 0.1,
    seed: int = 0,
    log_every: int = 25,
    n_tokens: int = 4_000_000,
    use_real_data: bool = True,
    warmup_steps: int = 100,
    clip_norm: float = 1.0,
) -> None:
    r = run_ab.remote(
        config=config,
        batch_size=batch_size,
        n_steps=n_steps,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        log_every=log_every,
        n_tokens=n_tokens,
        use_real_data=use_real_data,
        warmup_steps=warmup_steps,
        clip_norm=clip_norm,
    )

    local_path = Path("ab_fp8_curves.json")
    with open(local_path, "wb") as f:
        for chunk in trace_vol.read_file("ab_fp8_curves.json"):
            f.write(chunk)
    print(f"\ncurves saved to: {local_path.resolve()}")

    bf16_losses = r["bf16"]["losses"]
    fp8_losses = r["fp8_block"]["losses"]
    bf16_gns = r["bf16"]["grad_norms"]
    fp8_gns = r["fp8_block"]["grad_norms"]
    n = len(bf16_losses)

    def mean_last(xs, k):
        tail = xs[-k:]
        return sum(tail) / len(tail)

    k_tail = max(10, n // 10)
    bf16_tail = mean_last(bf16_losses, k_tail)
    fp8_tail = mean_last(fp8_losses, k_tail)
    bf16_gn_tail = mean_last(bf16_gns, k_tail)
    fp8_gn_tail = mean_last(fp8_gns, k_tail)

    print(f"\n{'─' * 60}")
    print(f"  A/B summary  ({n} steps, last {k_tail} averaged)")
    print(f"{'─' * 60}")
    print(f"  {'metric':<28}  {'bf16':>12}  {'fp8_block':>12}  {'Δ':>8}")
    print(f"  {'-' * 28}  {'-' * 12}  {'-' * 12}  {'-' * 8}")
    print(f"  {'tail loss':<28}  {bf16_tail:>12.4f}  {fp8_tail:>12.4f}  "
          f"{(fp8_tail - bf16_tail) / bf16_tail * 100:>+7.2f}%")
    print(f"  {'tail grad_norm':<28}  {bf16_gn_tail:>12.3f}  {fp8_gn_tail:>12.3f}  "
          f"{(fp8_gn_tail - bf16_gn_tail) / bf16_gn_tail * 100:>+7.2f}%")
    print(f"  {'step 0 loss':<28}  {bf16_losses[0]:>12.4f}  "
          f"{fp8_losses[0]:>12.4f}")
    print(f"  {'step ' + str(n // 2) + ' loss':<28}  "
          f"{bf16_losses[n // 2]:>12.4f}  {fp8_losses[n // 2]:>12.4f}")
    print(f"  {'final-step loss':<28}  {bf16_losses[-1]:>12.4f}  "
          f"{fp8_losses[-1]:>12.4f}")
    print(f"  {'tok/sec':<28}  {r['bf16']['tokens_per_sec']:>12,.0f}  "
          f"{r['fp8_block']['tokens_per_sec']:>12,.0f}")
    print(f"{'─' * 60}")

    # Quick divergence flag: any step where fp8 loss exceeds bf16 by >50%
    diverged_at = None
    for i, (bl, fl) in enumerate(zip(bf16_losses, fp8_losses)):
        if bl > 0 and (fl - bl) / bl > 0.5:
            diverged_at = i + 1
            break
    if diverged_at is not None:
        print(f"  WARNING: fp8 loss diverged >50% above bf16 at step {diverged_at}")
    else:
        print(f"  fp8 stays within 50% of bf16 loss across all {n} steps")
    print(f"{'─' * 60}")
    print(f"\n  full per-step data: {local_path.resolve()}")
    print(f"  (load with: json.load(open('{local_path.name}')))")
