"""Correctness gate: tiny model must overfit a fixed batch to near-zero loss."""
import jax
import jax.numpy as jnp
import optax

from config import SMALL
from losses import cross_entropy_loss as loss_fn
from model import init_params, param_count


def main():
    cfg = SMALL
    key = jax.random.PRNGKey(0)
    k_init, k_data = jax.random.split(key)

    params = init_params(k_init, cfg)
    print(f"param count: {param_count(params):,}")

    B, T = 2, 32
    tokens = jax.random.randint(k_data, (B, T + 1), 0, cfg.vocab_size)
    inputs, targets = tokens[:, :-1], tokens[:, 1:]

    optimizer = optax.adamw(learning_rate=3e-3)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, inputs, targets):
        loss, grads = jax.value_and_grad(loss_fn)(params, inputs, targets, cfg)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    initial_loss = float(loss_fn(params, inputs, targets, cfg))
    expected = jnp.log(cfg.vocab_size)
    print(f"initial loss: {initial_loss:.3f}  (random baseline ~{float(expected):.3f})")

    final = initial_loss
    for i in range(300):
        params, opt_state, loss = step(params, opt_state, inputs, targets)
        final = float(loss)
        if i % 25 == 0 or i == 299:
            print(f"step {i:3d}  loss {final:.4f}")

    assert final < 0.1, f"overfit failed: final loss {final:.4f}"
    print(f"\nPASS: overfit to {final:.4f}")


if __name__ == "__main__":
    main()
