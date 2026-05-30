"""Baseline bf16 training loop. Single device, no sharding, no checkpointing.

The model is pure bf16 (every matmul is a plain `a @ b`). FP8 experiments live
entirely in the fp8-specific files (fp8.py, ab_fp8.py, bench_fp8.py).

Stable defaults learned empirically on FineWeb-Edu / FULL config:
  - grad clipping at 1.0 — without it, adamw on cold init NaNs within ~50 steps
  - lr 1e-4 — 3e-4 diverges on real-text gradients
  - batch_size >= 8 — batch=4 NaNs around step 100 (gradient variance too high)
A warmup loss guard fails fast if the setup is broken before wasting a run.

Local smoke (SMALL config, synth data):
    python train.py

Real run (FULL config, real corpus) — call train() directly or from a Modal
wrapper, e.g.:
    from config import Config
    train(cfg=Config(), data_path=Path("data/fineweb-edu-10BT-train.bin"),
          batch_size=8, n_steps=5000, lr=1e-4)
"""
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from config import SMALL, Config
from data import batch_iter, load_tokens, synth_bin
from losses import cross_entropy_loss
from model import init_params, param_count


def make_optimizer(lr: float, weight_decay: float, clip_norm: float):
    """AdamW with global-norm grad clipping. Clipping is required for stability
    on cold init — see module docstring."""
    return optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.adamw(learning_rate=lr, weight_decay=weight_decay),
    )


def make_train_step(cfg: Config, optimizer: optax.GradientTransformation):
    @jax.jit
    def train_step(params, opt_state, inputs, targets):
        loss, grads = jax.value_and_grad(cross_entropy_loss)(
            params, inputs, targets, cfg
        )
        grad_norm = optax.global_norm(grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, grad_norm

    return train_step


def train(
    cfg: Config = SMALL,
    data_path: Path = Path("data/tokens.bin"),
    batch_size: int = 8,
    n_steps: int = 100,
    lr: float = 1e-4,
    weight_decay: float = 0.1,
    clip_norm: float = 1.0,
    log_every: int = 10,
    seed: int = 0,
):
    if not data_path.exists():
        print(f"no corpus at {data_path}; synthesizing {2_000_000} random tokens")
        synth_bin(data_path, cfg.vocab_size, n_tokens=2_000_000, seed=seed)

    tokens = load_tokens(data_path)
    print(f"corpus: {len(tokens):,} tokens at {data_path}")

    key = jax.random.PRNGKey(seed)
    params = init_params(key, cfg)
    print(f"params: {param_count(params):,}  (bf16)")

    optimizer = make_optimizer(lr, weight_decay, clip_norm)
    opt_state = optimizer.init(params)
    train_step = make_train_step(cfg, optimizer)

    batches = batch_iter(tokens, batch_size, cfg.seq_len, seed=seed)
    tokens_per_step = batch_size * cfg.seq_len

    inputs, targets = next(batches)
    params, opt_state, loss, grad_norm = train_step(
        params, opt_state, inputs, targets
    )
    jax.block_until_ready((loss, grad_norm))
    warmup_loss = float(loss)
    print(f"compile + step 0 done, loss={warmup_loss:.4f}  "
          f"grad_norm={float(grad_norm):.3f}")
    # Fail fast: a non-finite warmup loss means the model/data/hparams are
    # broken before training even starts — don't burn the whole run.
    if not math.isfinite(warmup_loss):
        raise RuntimeError(
            f"warmup loss is non-finite ({warmup_loss}); aborting before training"
        )

    t0 = time.perf_counter()
    last_log_t = t0
    last_log_step = 0
    for step in range(1, n_steps + 1):
        inputs, targets = next(batches)
        params, opt_state, loss, grad_norm = train_step(
            params, opt_state, inputs, targets
        )
        if step % log_every == 0:
            jax.block_until_ready((loss, grad_norm))
            loss_val = float(loss)
            if not math.isfinite(loss_val):
                raise RuntimeError(
                    f"loss went non-finite at step {step} ({loss_val}); aborting. "
                    f"Try a lower lr or a larger batch_size."
                )
            now = time.perf_counter()
            dt = now - last_log_t
            steps = step - last_log_step
            tps = steps * tokens_per_step / dt
            print(f"step {step:5d}  loss {loss_val:.4f}  "
                  f"grad_norm {float(grad_norm):.3f}  {tps:,.0f} tok/s")
            last_log_t, last_log_step = now, step

    jax.block_until_ready(loss)
    total_dt = time.perf_counter() - t0
    print(f"\n{n_steps} steps in {total_dt:.1f}s  "
          f"avg {n_steps * tokens_per_step / total_dt:,.0f} tok/s")
    return params


if __name__ == "__main__":
    train()
