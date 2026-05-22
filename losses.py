import jax
import jax.numpy as jnp

from config import Config
from model import forward


def cross_entropy_loss(params, input_ids, target_ids, cfg: Config):
    logits = forward(params, input_ids, cfg).astype(jnp.float32)
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, target_ids[..., None], axis=-1).squeeze(-1)
    return nll.mean()
