from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class Config:
    vocab_size: int = 50257
    seq_len: int = 2048
    d_model: int = 2304
    n_layers: int = 24
    n_heads: int = 18
    head_dim: int = 128
    d_ff: int = 9216
    dtype: jnp.dtype = jnp.bfloat16
    init_std: float = 0.02
    rms_eps: float = 1e-5
    tie_embeddings: bool = False
    logit_cap: float = 15.0
    # Gradient (activation) checkpointing on each transformer block. True trades
    # compute (a forward recompute in the backward pass) for much lower peak HBM
    # — required to fit large batch/seq/depth. False keeps all per-layer
    # activations live: faster per step, higher memory. See model.forward.
    use_remat: bool = True
    # Use the custom Pallas rms_norm kernel (True) or the pure-XLA fallback
    # (False). The Pallas kernel is faster but has miscompiled on B200 before
    # (see memory: fp8-pallas-rmsnorm-nan); the toggle lets us A/B it as a
    # suspect for numerical issues. model.forward picks the impl from this.
    use_pallas_norm: bool = True


SMALL = Config(
    vocab_size=256,
    seq_len=128,
    d_model=128,
    n_layers=2,
    n_heads=4,
    head_dim=32,
    d_ff=512,
)
