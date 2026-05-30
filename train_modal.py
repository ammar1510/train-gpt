"""Modal B200 launcher for the bf16 training run on FineWeb-Edu.

Runs the pure-bf16 loop (model.py / train.py — no fp8 anywhere) against the
real corpus on the `fineweb-edu-data` volume, and reports the same throughput
metrics as bench.py: tok/s, MFU, and peak HBM. Checkpoints are written to the
`train-gpt-checkpoints` volume (final + every --ckpt-every steps).

The MFU/FLOP helpers are copied from bench.py verbatim so the numbers are
directly comparable to a `modal run bench.py` reading. Like bench.py, the FLOP
estimate is the parameter-based 6*N*T approximation: it omits attention
score/value einsums and the embedding+logit matmul, so reported MFU is a
slight under-count of the true tensor-core utilisation. Good enough for
tracking run health and comparing against the benchmark.

Run (does NOT execute automatically — you launch it):
    modal run train_modal.py
    modal run train_modal.py --n-steps 5000 --batch-size 8 --lr 1e-4
    modal run train_modal.py --config small        # synth-data smoke test

Resume from a checkpoint on the volume:
    modal run train_modal.py --resume-from step_2000.pkl --n-steps 5000

Stable defaults (see train.py docstring): grad clip 1.0, lr 1e-4, batch >= 8.
"""
import pickle
from pathlib import Path

import modal

# B200 SXM BF16 dense tensor-core peak (TFLOP/s). Same reference as bench.py —
# verify against the NVIDIA product page before trusting MFU numbers.
B200_BF16_TFLOPS = 2_250.0

DATA_VOLUME_NAME = "fineweb-edu-data"
REMOTE_DATA_DIR = "/data"
REAL_TOKENS_FILE = "fineweb-edu-10BT-train.bin"

CKPT_VOLUME_NAME = "train-gpt-checkpoints"
REMOTE_CKPT_DIR = "/checkpoints"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "jax[cuda13]==0.9.2",
        "optax==0.2.8",
        "numpy==2.4.5",
        "chex==0.1.91",
    )
    .add_local_python_source("config", "model", "losses", "train", "data", "kernels")
)

app = modal.App("train-gpt-train")
data_vol = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
ckpt_vol = modal.Volume.from_name(CKPT_VOLUME_NAME, create_if_missing=True)


def _tree_to_numpy(tree):
    """Host-side numpy copy of a param pytree, for dependency-free pickling.
    Structure (plain dicts + stacked layer arrays) survives a pickle round-trip
    because it contains only dicts and arrays — no JAX-specific objects."""
    import numpy as np
    import jax

    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)


@app.function(
    image=image,
    gpu="B200",
    memory=48 * 1024,
    timeout=24 * 60 * 60,  # long-running training; cap at 24h per Modal limits
    volumes={REMOTE_DATA_DIR: data_vol, REMOTE_CKPT_DIR: ckpt_vol},
)
def run_train(
    config: str = "full",
    batch_size: int = 8,
    n_steps: int = 5000,
    lr: float = 1e-4,
    weight_decay: float = 0.1,
    clip_norm: float = 1.0,
    log_every: int = 50,
    ckpt_every: int = 1000,
    seed: int = 0,
    peak_tflops: float = B200_BF16_TFLOPS,
    resume_from: str = "",
) -> dict:
    import os
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

    import math
    import time

    import jax
    import jax.numpy as jnp
    import numpy as np

    from config import SMALL, Config
    from data import batch_iter, load_tokens, synth_bin
    from losses import cross_entropy_loss
    from model import init_params, param_count
    from train import make_optimizer, make_train_step

    cfg = SMALL if config == "small" else Config()

    # --- FLOP / MFU helpers (copied from bench.py for comparable numbers) ---
    def estimate_flops(cfg: Config, batch_size: int) -> float:
        d, h, ff, L = cfg.d_model, cfg.n_heads * cfg.head_dim, cfg.d_ff, cfg.n_layers
        layer_params = 3 * d * h + h * d + 2 * d * ff + ff * d
        tokens = batch_size * cfg.seq_len
        return 6.0 * L * layer_params * tokens  # fwd+bwd ≈ 6 * N * T

    def mfu(flops_per_step: float, step_time_s: float) -> float:
        achieved_tflops = flops_per_step / step_time_s / 1e12
        return achieved_tflops / peak_tflops

    def memory_gb() -> tuple[float, float]:
        try:
            stats = jax.devices()[0].memory_stats()
            return (
                stats["peak_bytes_in_use"] / 2**30,
                stats["bytes_limit"] / 2**30,
            )
        except Exception:
            return (0.0, 0.0)

    # --- corpus selection: real text for FULL, synth fallback for SMALL ---
    real_data_path = Path(REMOTE_DATA_DIR) / REAL_TOKENS_FILE
    if config != "small":
        if not real_data_path.exists():
            raise FileNotFoundError(
                f"expected real corpus at {real_data_path} on volume "
                f"'{DATA_VOLUME_NAME}', but it is missing. Run prepare_data first "
                f"or pass --config small for a synth-data smoke test."
            )
        data_path = real_data_path
        print(f"corpus: {data_path} ({data_path.stat().st_size:,} bytes)")
    else:
        data_path = Path("/tmp/train_tokens.bin")
        if not data_path.exists():
            synth_bin(data_path, cfg.vocab_size, n_tokens=2_000_000, seed=seed)
        print(f"corpus (synth): {data_path}")

    tokens = load_tokens(data_path)
    print(f"loaded {len(tokens):,} tokens")

    # --- params: fresh init or resume from a checkpoint on the volume ---
    key = jax.random.PRNGKey(seed)
    params = init_params(key, cfg)
    start_step = 0
    if resume_from:
        ckpt_path = Path(REMOTE_CKPT_DIR) / resume_from
        if not ckpt_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {ckpt_path}")
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        # Restore as device arrays in the model dtype.
        params = jax.tree_util.tree_map(
            lambda x: jnp.asarray(x, dtype=cfg.dtype), ckpt["params"]
        )
        start_step = int(ckpt.get("step", 0))
        print(f"resumed from {resume_from} at step {start_step}")

    n_params = param_count(params)
    print(f"params: {n_params:,}  (bf16)")

    optimizer = make_optimizer(lr, weight_decay, clip_norm)
    opt_state = optimizer.init(params)
    train_step = make_train_step(cfg, optimizer)

    batches = batch_iter(tokens, batch_size, cfg.seq_len, seed=seed + start_step)
    tokens_per_step = batch_size * cfg.seq_len
    flops_per_step = estimate_flops(cfg, batch_size)

    def save_ckpt(params, step: int) -> str:
        """Pickle a host-side numpy copy of params to the checkpoint volume."""
        name = f"step_{step}.pkl"
        path = Path(REMOTE_CKPT_DIR) / name
        payload = {
            "step": step,
            "config": config,
            "params": _tree_to_numpy(params),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        ckpt_vol.commit()
        print(f"  checkpoint saved: {name}")
        return name

    # --- warmup step: compile + fail-fast on a broken setup ---
    inputs, targets = next(batches)
    params, opt_state, loss, grad_norm = train_step(
        params, opt_state, inputs, targets
    )
    jax.block_until_ready((loss, grad_norm))
    warmup_loss = float(loss)
    print(f"compile + warmup done, loss={warmup_loss:.4f}  "
          f"grad_norm={float(grad_norm):.3f}")
    if not math.isfinite(warmup_loss):
        raise RuntimeError(
            f"warmup loss is non-finite ({warmup_loss}); aborting before training"
        )

    history = []  # (step, loss, grad_norm) for the returned summary
    t0 = time.perf_counter()
    last_log_t, last_log_step = t0, 0
    peak_mfu = 0.0

    for step in range(start_step + 1, start_step + n_steps + 1):
        inputs, targets = next(batches)
        params, opt_state, loss, grad_norm = train_step(
            params, opt_state, inputs, targets
        )

        if step % log_every == 0:
            jax.block_until_ready((loss, grad_norm))
            loss_val = float(loss)
            gn_val = float(grad_norm)
            if not math.isfinite(loss_val):
                # Save what we have so the run isn't a total loss, then abort.
                save_ckpt(params, step)
                raise RuntimeError(
                    f"loss went non-finite at step {step} ({loss_val}); aborting. "
                    f"Try a lower lr or a larger batch_size. (checkpoint saved)"
                )
            now = time.perf_counter()
            dt = now - last_log_t
            steps_done = step - last_log_step
            tps = steps_done * tokens_per_step / dt
            step_time = dt / steps_done
            cur_mfu = mfu(flops_per_step, step_time)
            peak_mfu = max(peak_mfu, cur_mfu)
            done = step - start_step
            eta_s = (n_steps - done) * step_time
            print(f"step {step:6d}  loss {loss_val:7.4f}  "
                  f"grad_norm {gn_val:7.3f}  {tps:>10,.0f} tok/s  "
                  f"MFU {cur_mfu * 100:4.1f}%  ETA {eta_s / 60:5.1f}m")
            history.append((step, loss_val, gn_val))
            last_log_t, last_log_step = now, step

        if ckpt_every > 0 and step % ckpt_every == 0:
            jax.block_until_ready((loss, grad_norm))
            save_ckpt(params, step)

    jax.block_until_ready(loss)
    total_dt = time.perf_counter() - t0
    final_step = start_step + n_steps

    final_ckpt = ""
    if ckpt_every > 0:
        final_ckpt = save_ckpt(params, final_step)

    avg_tps = n_steps * tokens_per_step / total_dt
    avg_step_time = total_dt / n_steps
    avg_mfu = mfu(flops_per_step, avg_step_time)
    used_gb, limit_gb = memory_gb()

    return {
        "config": config,
        "n_params": n_params,
        "batch_size": batch_size,
        "seq_len": cfg.seq_len,
        "device": str(jax.devices()[0]),
        "device_kind": jax.devices()[0].device_kind,
        "n_steps": n_steps,
        "start_step": start_step,
        "final_step": final_step,
        "tokens_trained": n_steps * tokens_per_step,
        "wall_seconds": total_dt,
        "avg_tok_per_sec": avg_tps,
        "avg_mfu": avg_mfu,
        "peak_mfu": peak_mfu,
        "hbm_peak_gb": used_gb,
        "hbm_limit_gb": limit_gb,
        "final_loss": history[-1][1] if history else warmup_loss,
        "final_grad_norm": history[-1][2] if history else None,
        "final_checkpoint": final_ckpt,
        "peak_tflops": peak_tflops,
    }


@app.local_entrypoint()
def main(
    config: str = "full",
    batch_size: int = 8,
    n_steps: int = 5000,
    lr: float = 1e-4,
    weight_decay: float = 0.1,
    clip_norm: float = 1.0,
    log_every: int = 50,
    ckpt_every: int = 1000,
    seed: int = 0,
    peak_tflops: float = B200_BF16_TFLOPS,
    resume_from: str = "",
):
    r = run_train.remote(
        config=config,
        batch_size=batch_size,
        n_steps=n_steps,
        lr=lr,
        weight_decay=weight_decay,
        clip_norm=clip_norm,
        log_every=log_every,
        ckpt_every=ckpt_every,
        seed=seed,
        peak_tflops=peak_tflops,
        resume_from=resume_from,
    )

    print(f"\n{'─' * 60}")
    print(f"  config      : {r['config']}  ({r['n_params']:,} params)")
    print(f"  batch       : {r['batch_size']}  seq_len {r['seq_len']}")
    print(f"  device      : {r['device']}  ({r['device_kind']})")
    print(f"  peak ref    : {r['peak_tflops']:,.0f} TFLOP/s  (BF16 dense)")
    print(f"  steps       : {r['start_step']} → {r['final_step']}  "
          f"({r['n_steps']} this run)")
    print(f"{'─' * 60}")
    print(f"  tokens      : {r['tokens_trained']:,}")
    print(f"  wall time   : {r['wall_seconds'] / 60:.1f} min")
    print(f"  throughput  : {r['avg_tok_per_sec']:,.0f} tok/s avg")
    print(f"  MFU         : {r['avg_mfu'] * 100:.1f}% avg   "
          f"{r['peak_mfu'] * 100:.1f}% peak")
    print(f"  HBM peak    : {r['hbm_peak_gb']:.1f} / {r['hbm_limit_gb']:.1f} GB")
    print(f"  final loss  : {r['final_loss']:.4f}")
    if r["final_grad_norm"] is not None:
        print(f"  final |g|   : {r['final_grad_norm']:.3f}")
    if r["final_checkpoint"]:
        print(f"  checkpoint  : {r['final_checkpoint']}  "
              f"(volume '{CKPT_VOLUME_NAME}')")
    print(f"{'─' * 60}")
    print(f"\n  download checkpoint with:")
    print(f"    modal volume get {CKPT_VOLUME_NAME} "
          f"{r['final_checkpoint'] or 'step_<N>.pkl'} ./")
