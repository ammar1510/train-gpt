"""Baseline training loop. Single device, no sharding, no checkpointing."""
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from config import SMALL, Config
from data import batch_iter, load_tokens, synth_bin
from losses import cross_entropy_loss
from model import init_params, param_count


def make_train_step(cfg: Config, optimizer: optax.GradientTransformation):
    @jax.jit
    def train_step(params, opt_state, inputs, targets):
        loss, grads = jax.value_and_grad(cross_entropy_loss)(
            params, inputs, targets, cfg
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return train_step


def train(
    cfg: Config = SMALL,
    data_path: Path = Path("data/tokens.bin"),
    batch_size: int = 8,
    n_steps: int = 100,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
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
    print(f"params: {param_count(params):,}")

    optimizer = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    opt_state = optimizer.init(params)
    train_step = make_train_step(cfg, optimizer)

    batches = batch_iter(tokens, batch_size, cfg.seq_len, seed=seed)
    tokens_per_step = batch_size * cfg.seq_len

    inputs, targets = next(batches)
    params, opt_state, loss = train_step(params, opt_state, inputs, targets)
    jax.block_until_ready(loss)
    print(f"compile + step 0 done, loss={float(loss):.4f}")

    t0 = time.perf_counter()
    last_log_t = t0
    last_log_step = 0
    for step in range(1, n_steps + 1):
        inputs, targets = next(batches)
        params, opt_state, loss = train_step(params, opt_state, inputs, targets)
        if step % log_every == 0:
            jax.block_until_ready(loss)
            now = time.perf_counter()
            dt = now - last_log_t
            steps = step - last_log_step
            tps = steps * tokens_per_step / dt
            print(f"step {step:5d}  loss {float(loss):.4f}  {tps:,.0f} tok/s")
            last_log_t, last_log_step = now, step

    jax.block_until_ready(loss)
    total_dt = time.perf_counter() - t0
    print(f"\n{n_steps} steps in {total_dt:.1f}s  "
          f"avg {n_steps * tokens_per_step / total_dt:,.0f} tok/s")
    return params


if __name__ == "__main__":
    train()
