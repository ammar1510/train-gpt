"""Modal B200 launcher for instruction fine-tuning (SFT) on No Robots.

Loads a *pretrained* base checkpoint, then continues training with the masked
SFT objective (loss only on assistant responses — see
losses.masked_cross_entropy_loss) over the padded No Robots arrays produced by
prepare_sft_data.py. Unlike train_modal.py's --resume-from (which continues a
pretraining run mid-schedule), SFT starts a FRESH short schedule at step 0: it
only borrows the base weights, not the optimizer state or step counter.

Defaults are SFT-appropriate and ~10x gentler than pretraining (peak lr 1e-5,
no weight decay, a few epochs) so the model adapts to the chat format without
washing out what it learned during pretraining.

Prereqs:
    modal run prepare_sft_data.py          # writes arrays to train-gpt-sft-data
    # a base checkpoint on train-gpt-checkpoints, e.g. run_.../step_10000.pkl

Run (you launch it; nothing runs automatically):
    modal run sft_modal.py --base-checkpoint run_20260601_120000/step_10000.pkl
    modal run sft_modal.py --base-checkpoint <ckpt> --n-epochs 3 --lr 1e-5 \
        --batch-size 8
"""
import pickle
import re
from pathlib import Path

import modal

# B200 SXM BF16 dense tensor-core peak (TFLOP/s) — same reference as bench.py.
B200_BF16_TFLOPS = 2_250.0

SFT_VOLUME_NAME = "train-gpt-sft-data"
REMOTE_SFT_DIR = "/sft-data"
SFT_TRAIN_IDS = "sft-no_robots-train-input_ids.npy"
SFT_TRAIN_MASK = "sft-no_robots-train-loss_mask.npy"

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
    .add_local_python_source(
        "config", "model", "losses", "train", "data", "sft_data", "kernels"
    )
)

app = modal.App("train-gpt-sft")
sft_vol = modal.Volume.from_name(SFT_VOLUME_NAME, create_if_missing=True)
ckpt_vol = modal.Volume.from_name(CKPT_VOLUME_NAME, create_if_missing=True)


def _make_run_id() -> str:
    import time

    return time.strftime("sft_%Y%m%d_%H%M%S")


def _safe_run_id(run_id: str) -> str:
    if run_id in (".", "..") or ".." in run_id or not re.fullmatch(
        r"[A-Za-z0-9._-]+", run_id
    ):
        raise ValueError(
            f"run_id must be a single safe path segment [A-Za-z0-9._-] with no "
            f"'..'; got {run_id!r}"
        )
    return run_id


def _tree_to_numpy(tree):
    import jax
    import numpy as np

    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)


@app.function(
    image=image,
    gpu="B200",
    memory=48 * 1024,
    timeout=12 * 60 * 60,
    volumes={REMOTE_SFT_DIR: sft_vol, REMOTE_CKPT_DIR: ckpt_vol},
)
def run_sft(
    base_checkpoint: str,
    batch_size: int = 256,
    n_epochs: int = 3,
    max_steps: int = 0,
    lr: float = 1e-5,
    warmup_steps: int = 50,
    lr_schedule: str = "cosine",
    lr_end_frac: float = 0.1,
    weight_decay: float = 0.0,
    clip_norm: float = 1.0,
    log_every: int = 25,
    ckpt_every_epochs: int = 1,
    seed: int = 0,
    peak_tflops: float = B200_BF16_TFLOPS,
    use_remat: bool = True,
    use_pallas_norm: bool = False,
    run_id: str = "",
) -> dict:
    import os
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

    import dataclasses
    import math
    import time

    import jax
    import jax.numpy as jnp

    from config import Config
    from model import init_params, param_count
    from sft_data import (
        aligned_seq_len, load_sft, sft_batch_iter, steps_per_epoch
    )
    from train import make_lr_schedule, make_optimizer, make_sft_train_step

    if not base_checkpoint:
        raise ValueError("--base-checkpoint is required (the pretrained weights)")

    run_id = _safe_run_id(run_id or _make_run_id())
    print(f"run_id: {run_id}  (SFT checkpoints under {REMOTE_CKPT_DIR}/{run_id}/)")

    # use_pallas_norm defaults False: the Pallas rms_norm kernel NaNs on B200
    # (see memory: batch128-bf16-divergence). Match the pretraining/inference path.
    cfg = dataclasses.replace(
        Config(), use_remat=use_remat, use_pallas_norm=use_pallas_norm
    )

    # --- load SFT data (padded arrays from prepare_sft_data.py) ---
    ids_path = Path(REMOTE_SFT_DIR) / SFT_TRAIN_IDS
    mask_path = Path(REMOTE_SFT_DIR) / SFT_TRAIN_MASK
    if not ids_path.exists() or not mask_path.exists():
        raise FileNotFoundError(
            f"SFT arrays not found on volume '{SFT_VOLUME_NAME}' "
            f"({ids_path.name} / {mask_path.name}). Run prepare_sft_data.py first."
        )
    input_ids, loss_mask = load_sft(ids_path, mask_path)
    n_examples, row_len = input_ids.shape
    spe = steps_per_epoch(n_examples, batch_size)
    if spe == 0:
        raise ValueError(
            f"batch_size {batch_size} exceeds dataset size {n_examples}"
        )
    n_steps = spe * n_epochs  # full schedule horizon (warmup/decay span this)
    # --max-steps caps how many steps actually RUN (a quick smoke / data subset:
    # only max_steps*batch examples are touched) WITHOUT reshaping the LR
    # schedule, so the early-step loss/LR you observe match a real run's start.
    run_steps = min(n_steps, max_steps) if max_steps > 0 else n_steps
    # Trained length = (row_len - 1) shifted, padded up to a cuDNN-friendly
    # multiple of 128 by the loader (flash-attention rejects e.g. 1023).
    seq_T = aligned_seq_len(row_len)
    smoke = " [SMOKE: capped]" if run_steps < n_steps else ""
    print(f"SFT data: {n_examples:,} examples x {row_len} tok  "
          f"({spe} steps/epoch x {n_epochs} epochs = {n_steps} steps horizon; "
          f"running {run_steps} steps{smoke})")

    # --- load pretrained weights (params only; fresh optimizer + schedule) ---
    ckpt_root = Path(REMOTE_CKPT_DIR).resolve()
    ckpt_path = (ckpt_root / base_checkpoint).resolve()
    if ckpt_path != ckpt_root and ckpt_root not in ckpt_path.parents:
        raise ValueError(
            f"base_checkpoint must stay within {REMOTE_CKPT_DIR}; got "
            f"{base_checkpoint!r}"
        )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"base checkpoint not found: {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    if ckpt.get("config") == "small":
        raise ValueError(
            "base checkpoint is a SMALL-config model; SFT launcher targets the "
            "FULL config used for the real pretraining run"
        )
    params = jax.tree_util.tree_map(
        lambda x: jnp.asarray(x, dtype=cfg.dtype), ckpt["params"]
    )
    base_step = int(ckpt.get("step", 0))
    n_params = param_count(params)
    print(f"loaded base weights from {base_checkpoint} (pretrain step "
          f"{base_step}); params: {n_params:,}  use_remat={use_remat}")

    # --- fresh schedule (starts at 0) + optimizer + masked train step ---
    sched = make_lr_schedule(
        lr, n_steps, warmup_steps=warmup_steps,
        end_lr_frac=lr_end_frac, kind=lr_schedule,
    )
    lr_at = (lambda s: float(sched(s))) if callable(sched) else (lambda s: float(sched))
    optimizer = make_optimizer(sched, weight_decay, clip_norm)
    opt_state = optimizer.init(params)
    train_step = make_sft_train_step(cfg, optimizer)

    batches = sft_batch_iter(input_ids, loss_mask, batch_size, seed=seed)
    tokens_per_step = batch_size * seq_T

    def estimate_flops() -> float:
        d, h, ff, L = cfg.d_model, cfg.n_heads * cfg.head_dim, cfg.d_ff, cfg.n_layers
        layer_params = 3 * d * h + h * d + 2 * d * ff + ff * d
        return 6.0 * L * layer_params * tokens_per_step  # fwd+bwd ~ 6*N*T
    flops_per_step = estimate_flops()

    def mfu(step_time_s: float) -> float:
        return (flops_per_step / step_time_s / 1e12) / peak_tflops

    def memory_gb() -> tuple[float, float]:
        try:
            s = jax.devices()[0].memory_stats()
            return s["peak_bytes_in_use"] / 2**30, s["bytes_limit"] / 2**30
        except Exception:
            return 0.0, 0.0

    def save_ckpt(params, step: int) -> str:
        name = f"step_{step}.pkl"
        rel = f"{run_id}/{name}"
        path = Path(REMOTE_CKPT_DIR) / run_id / name
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "step": step,
            "config": ckpt.get("config", "full"),
            "run_id": run_id,
            "sft": True,
            "base_checkpoint": base_checkpoint,
            "params": _tree_to_numpy(params),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        ckpt_vol.commit()
        print(f"  checkpoint saved: {rel}")
        return rel

    # --- warmup step: compile + fail-fast on a broken setup ---
    inputs, targets, mask = next(batches)
    params, opt_state, loss, grad_norm = train_step(
        params, opt_state, inputs, targets, mask
    )
    jax.block_until_ready((loss, grad_norm))
    warmup_loss = float(loss)
    print(f"compile + warmup done, loss={warmup_loss:.4f}  "
          f"grad_norm={float(grad_norm):.3f}")
    if not math.isfinite(warmup_loss):
        raise RuntimeError(
            f"warmup loss is non-finite ({warmup_loss}); aborting before SFT"
        )

    history = []
    t0 = time.perf_counter()
    last_log_t, last_log_step = t0, 0
    peak_mfu = 0.0
    final_ckpt = ""
    ckpt_every = spe * max(1, ckpt_every_epochs)

    for step in range(1, run_steps + 1):
        inputs, targets, mask = next(batches)
        params, opt_state, loss, grad_norm = train_step(
            params, opt_state, inputs, targets, mask
        )

        if step % log_every == 0:
            jax.block_until_ready((loss, grad_norm))
            loss_val, gn_val = float(loss), float(grad_norm)
            if not math.isfinite(loss_val):
                save_ckpt(params, step)
                raise RuntimeError(
                    f"loss went non-finite at step {step} ({loss_val}); aborting. "
                    f"Try a lower lr. (checkpoint saved)"
                )
            now = time.perf_counter()
            dt = now - last_log_t
            steps_done = step - last_log_step
            tps = steps_done * tokens_per_step / dt
            cur_mfu = mfu(dt / steps_done)
            peak_mfu = max(peak_mfu, cur_mfu)
            epoch = (step - 1) // spe + 1
            print(f"step {step:6d}/{run_steps}  epoch {epoch}/{n_epochs}  "
                  f"loss {loss_val:7.4f}  grad_norm {gn_val:7.3f}  "
                  f"lr {lr_at(step):.2e}  {tps:>9,.0f} tok/s  "
                  f"MFU {cur_mfu * 100:4.1f}%")
            history.append((step, loss_val, gn_val))
            last_log_t, last_log_step = now, step

        if step % ckpt_every == 0 and step < run_steps:
            jax.block_until_ready((loss, grad_norm))
            save_ckpt(params, step)

    jax.block_until_ready(loss)
    total_dt = time.perf_counter() - t0
    final_ckpt = save_ckpt(params, run_steps)
    used_gb, limit_gb = memory_gb()

    return {
        "run_id": run_id,
        "base_checkpoint": base_checkpoint,
        "n_params": n_params,
        "batch_size": batch_size,
        "seq_len_trained": seq_T,
        "n_examples": n_examples,
        "n_epochs": n_epochs,
        "n_steps": run_steps,
        "horizon_steps": n_steps,
        "device": str(jax.devices()[0]),
        "device_kind": jax.devices()[0].device_kind,
        "tokens_trained": run_steps * tokens_per_step,
        "wall_seconds": total_dt,
        "avg_tok_per_sec": run_steps * tokens_per_step / total_dt,
        "avg_mfu": mfu(total_dt / run_steps),
        "peak_mfu": peak_mfu,
        "hbm_peak_gb": used_gb,
        "hbm_limit_gb": limit_gb,
        "final_loss": history[-1][1] if history else warmup_loss,
        "final_checkpoint": final_ckpt,
        "peak_tflops": peak_tflops,
    }


@app.local_entrypoint()
def main(
    base_checkpoint: str,
    batch_size: int = 256,
    n_epochs: int = 3,
    max_steps: int = 0,
    lr: float = 1e-5,
    warmup_steps: int = 50,
    lr_schedule: str = "cosine",
    lr_end_frac: float = 0.1,
    weight_decay: float = 0.0,
    clip_norm: float = 1.0,
    log_every: int = 25,
    ckpt_every_epochs: int = 1,
    seed: int = 0,
    peak_tflops: float = B200_BF16_TFLOPS,
    use_remat: bool = True,
    use_pallas_norm: bool = False,
    run_id: str = "",
):
    run_id = _safe_run_id(run_id or _make_run_id())
    print(f"launching SFT run_id: {run_id}  (base: {base_checkpoint})")
    r = run_sft.remote(
        base_checkpoint=base_checkpoint,
        batch_size=batch_size,
        n_epochs=n_epochs,
        max_steps=max_steps,
        lr=lr,
        warmup_steps=warmup_steps,
        lr_schedule=lr_schedule,
        lr_end_frac=lr_end_frac,
        weight_decay=weight_decay,
        clip_norm=clip_norm,
        log_every=log_every,
        ckpt_every_epochs=ckpt_every_epochs,
        seed=seed,
        peak_tflops=peak_tflops,
        use_remat=use_remat,
        use_pallas_norm=use_pallas_norm,
        run_id=run_id,
    )

    print(f"\n{'─' * 60}")
    print(f"  run_id      : {r['run_id']}  ({r['n_params']:,} params)")
    print(f"  base ckpt   : {r['base_checkpoint']}")
    print(f"  data        : {r['n_examples']:,} examples  "
          f"batch {r['batch_size']}  seq {r['seq_len_trained']}")
    print(f"  device      : {r['device']}  ({r['device_kind']})")
    print(f"  steps       : {r['n_steps']}  ({r['n_epochs']} epochs)")
    print(f"{'─' * 60}")
    print(f"  tokens      : {r['tokens_trained']:,}")
    print(f"  wall time   : {r['wall_seconds'] / 60:.1f} min")
    print(f"  throughput  : {r['avg_tok_per_sec']:,.0f} tok/s avg")
    print(f"  MFU         : {r['avg_mfu'] * 100:.1f}% avg   "
          f"{r['peak_mfu'] * 100:.1f}% peak")
    print(f"  HBM peak    : {r['hbm_peak_gb']:.1f} / {r['hbm_limit_gb']:.1f} GB")
    print(f"  final loss  : {r['final_loss']:.4f}")
    if r["final_checkpoint"]:
        print(f"  checkpoint  : {r['final_checkpoint']}  "
              f"(volume '{CKPT_VOLUME_NAME}')")
    print(f"{'─' * 60}")
    if r["final_checkpoint"]:
        print(f"\n  chat with the fine-tuned model:")
        print(f"    modal run generate_modal.py --checkpoint "
              f"{r['final_checkpoint']} --chat --prompt \"Explain photosynthesis.\"")
