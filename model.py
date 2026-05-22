import math

import jax
import jax.numpy as jnp
from jax import lax

from config import Config


def init_params(key, cfg: Config):
    k_embed, k_pos, k_layers, k_out = jax.random.split(key, 4)
    std = cfg.init_std
    proj_std = cfg.init_std / math.sqrt(2.0 * cfg.n_layers)

    def normal(k, shape, s=std):
        return (jax.random.normal(k, shape) * s).astype(cfg.dtype)

    layer_keys = jax.random.split(k_layers, cfg.n_layers)

    def init_layer(lk):
        k_qkv, k_o, k_up, k_down, k_n1, k_n2 = jax.random.split(lk, 6)
        d = cfg.d_model
        qkv_out = 3 * cfg.n_heads * cfg.head_dim
        return {
            "norm1": jnp.ones((d,), dtype=cfg.dtype),
            "wqkv": normal(k_qkv, (d, qkv_out)),
            "wo": normal(k_o, (cfg.n_heads * cfg.head_dim, d), s=proj_std),
            "norm2": jnp.ones((d,), dtype=cfg.dtype),
            "w_up": normal(k_up, (d, cfg.d_ff)),
            "w_down": normal(k_down, (cfg.d_ff, d), s=proj_std),
        }

    layers = jax.vmap(init_layer)(layer_keys)

    params = {
        "embed": normal(k_embed, (cfg.vocab_size, cfg.d_model)),
        "pos": normal(k_pos, (cfg.seq_len, cfg.d_model)),
        "layers": layers,
        "final_norm": jnp.ones((cfg.d_model,), dtype=cfg.dtype),
    }
    if not cfg.tie_embeddings:
        params["unembed"] = normal(k_out, (cfg.d_model, cfg.vocab_size))
    return params


def rms_norm(x, scale, eps):
    var = jnp.mean(x.astype(jnp.float32) ** 2, axis=-1, keepdims=True)
    x = x * jax.lax.rsqrt(var + eps).astype(x.dtype)
    return x * scale



def attention(x, p, cfg: Config):
    B, T, D = x.shape
    H, Dh = cfg.n_heads, cfg.head_dim
    qkv = x @ p["wqkv"]
    qkv = qkv.reshape(B, T, 3, H, Dh)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    # dot_product_attention expects (B, T, H, Dh) — no transpose needed
    impl = "cudnn" if jax.default_backend() == "gpu" else "xla"
    out = jax.nn.dot_product_attention(q, k, v, is_causal=True, implementation=impl)
    out = out.reshape(B, T, H * Dh)
    return out @ p["wo"]


def mlp(x, p):
    return (jax.nn.relu(x @ p["w_up"]) ** 2) @ p["w_down"]


def block(x, p, cfg: Config):
    x = x + attention(rms_norm(x, p["norm1"], cfg.rms_eps), p, cfg)
    x = x + mlp(rms_norm(x, p["norm2"], cfg.rms_eps), p)
    return x


def forward(params, input_ids, cfg: Config):
    B, T = input_ids.shape
    x = params["embed"][input_ids] + params["pos"][:T]

    def step(carry, layer_p):
        return block(carry, layer_p, cfg), None

    x, _ = lax.scan(step, x, params["layers"])
    x = rms_norm(x, params["final_norm"], cfg.rms_eps)

    w_out = params["embed"].T if cfg.tie_embeddings else params["unembed"]
    logits = x @ w_out
    logits = cfg.logit_cap * jnp.tanh(logits / cfg.logit_cap)
    return logits


def param_count(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
