import jax
import jax.numpy as jnp

from config import Config
from model import forward


def cross_entropy_loss(params, input_ids, target_ids, cfg: Config):
    logits = forward(params, input_ids, cfg).astype(jnp.float32)
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, target_ids[..., None], axis=-1).squeeze(-1)
    return nll.mean()


def masked_cross_entropy_loss(params, input_ids, target_ids, loss_mask, cfg: Config):
    """Cross-entropy averaged over masked target positions only — the SFT loss.

    Identical to `cross_entropy_loss` except the per-token NLL is weighted by
    `loss_mask` (1.0 on tokens we want the model to learn to generate, 0.0 on
    everything else) before averaging. For instruction tuning the mask is 0 over
    the prompt / "User:" turns and padding, and 1 over the assistant response
    tokens (and its closing end-of-text), so gradients flow only from the
    completion — the model learns to *answer*, not to reproduce prompts.

    `loss_mask` is aligned to `target_ids` (same shape) and must contain only
    0.0/1.0. The denominator is clamped to >= 1 so an all-padding batch yields a
    finite (zero) loss instead of a 0/0 NaN.
    """
    logits = forward(params, input_ids, cfg).astype(jnp.float32)
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, target_ids[..., None], axis=-1).squeeze(-1)
    mask = loss_mask.astype(jnp.float32)
    return (nll * mask).sum() / jnp.maximum(mask.sum(), 1.0)
