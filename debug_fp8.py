"""Minimal Modal smoke test for jax.nn.scaled_matmul on B200.

Tests: (1) does it run at all, (2) does it produce finite output on random data
matching our model's shapes, (3) what's the precision vs bf16 ref.
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "jax[cuda13]==0.9.2",
        "optax==0.2.8",
        "numpy==2.4.5",
        "chex==0.1.91",
    )
    .add_local_python_source("config", "model", "losses", "data", "kernels", "fp8")
)

app = modal.App("train-gpt-fp8-smoke")


@app.function(image=image, gpu="B200", timeout=10 * 60)
def smoke() -> str:
    import jax
    import jax.numpy as jnp

    E4M3_MAX = 448.0
    block_k = 128

    def quant(x_f32, block_k):
        *leading, K = x_f32.shape
        n_tiles = K // block_k
        x_tiled = x_f32.reshape(*leading, n_tiles, block_k)
        amax = jnp.max(jnp.abs(x_tiled), axis=-1)
        scales = jnp.maximum(amax / E4M3_MAX, 1e-30)
        x_q = (x_tiled / scales[..., None]).astype(jnp.float8_e4m3fn).reshape(*leading, K)
        return x_q, scales

    def bsm(a_bf, b_bf, block_k):
        # a: (M, K), b: (K, N)
        a_f8, a_s = quant(a_bf.astype(jnp.float32), block_k)
        b_t_f8, b_s = quant(b_bf.T.astype(jnp.float32), block_k)
        out = jax.nn.scaled_matmul(
            a_f8[None], b_t_f8[None], a_s[None], b_s[None],
            preferred_element_type=jnp.float32,
        )[0]
        return out.astype(jnp.bfloat16)

    print("=== scaled_matmul smoke ===")

    # Case 1: tiny shapes, identity-ish input
    a = jnp.ones((4, 128), dtype=jnp.bfloat16)
    b = jnp.ones((128, 4), dtype=jnp.bfloat16)
    out = bsm(a, b, 128)
    ref = a.astype(jnp.float32) @ b.astype(jnp.float32)
    print(f"tiny ones: out_max={float(jnp.abs(out).max()):.3f} ref_max={float(jnp.abs(ref).max()):.3f} finite={bool(jnp.isfinite(out).all())}")

    # Case 2: real model shape, random gaussian (matches our QKV matmul shape)
    key = jax.random.PRNGKey(0)
    ka, kb = jax.random.split(key)
    a = jax.random.normal(ka, (8 * 2048, 2304), dtype=jnp.bfloat16)
    b = jax.random.normal(kb, (2304, 2304), dtype=jnp.bfloat16) * 0.02
    out = bsm(a, b, 128)
    ref = (a.astype(jnp.float32) @ b.astype(jnp.float32)).astype(jnp.bfloat16)
    rel = float(jnp.linalg.norm((out - ref).astype(jnp.float32)) / jnp.linalg.norm(ref.astype(jnp.float32)))
    finite = bool(jnp.isfinite(out).all())
    out_max = float(jnp.abs(jnp.where(jnp.isfinite(out), out, 0)).max())
    ref_max = float(jnp.abs(ref).max())
    print(f"QKV shape M={a.shape[0]} K=2304 N=2304: finite={finite} out_max={out_max:.3f} ref_max={ref_max:.3f} rel_err={rel:.4f}")

    # Case 3: MLP up shape
    b2 = jax.random.normal(kb, (2304, 9216), dtype=jnp.bfloat16) * 0.02
    out = bsm(a, b2, 128)
    ref = (a.astype(jnp.float32) @ b2.astype(jnp.float32)).astype(jnp.bfloat16)
    rel = float(jnp.linalg.norm((out - ref).astype(jnp.float32)) / jnp.linalg.norm(ref.astype(jnp.float32)))
    finite = bool(jnp.isfinite(out).all())
    print(f"MLP-up K=2304 N=9216: finite={finite} rel_err={rel:.4f}")

    # Case 4: MLP down shape
    a3 = jax.random.normal(ka, (8 * 2048, 9216), dtype=jnp.bfloat16)
    b3 = jax.random.normal(kb, (9216, 2304), dtype=jnp.bfloat16) * 0.02
    out = bsm(a3, b3, 128)
    ref = (a3.astype(jnp.float32) @ b3.astype(jnp.float32)).astype(jnp.bfloat16)
    rel = float(jnp.linalg.norm((out - ref).astype(jnp.float32)) / jnp.linalg.norm(ref.astype(jnp.float32)))
    finite = bool(jnp.isfinite(out).all())
    print(f"MLP-dn K=9216 N=2304: finite={finite} rel_err={rel:.4f}")

    return "done"


@app.function(image=image, gpu="B200", memory=48 * 1024, timeout=20 * 60)
def smoke_model() -> str:
    import os
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    import dataclasses
    import jax
    import jax.numpy as jnp
    from config import Config
    from model import init_params
    from losses import cross_entropy_loss

    base_cfg = Config()
    cfg_fp8 = dataclasses.replace(base_cfg, matmul_precision="fp8_block")
    cfg_bf16 = base_cfg

    key = jax.random.PRNGKey(0)
    params = init_params(key, cfg_bf16)
    input_ids = jax.random.randint(jax.random.PRNGKey(1), (8, 2048), 0, cfg_bf16.vocab_size, dtype=jnp.int32)
    targets = jax.random.randint(jax.random.PRNGKey(2), (8, 2048), 0, cfg_bf16.vocab_size, dtype=jnp.int32)

    @jax.jit
    def loss_and_grad_bf16(p, ids, tgt):
        return jax.value_and_grad(cross_entropy_loss)(p, ids, tgt, cfg_bf16)
    @jax.jit
    def loss_and_grad_fp8(p, ids, tgt):
        return jax.value_and_grad(cross_entropy_loss)(p, ids, tgt, cfg_fp8)

    print("bf16 value_and_grad...")
    loss_bf, grads_bf = loss_and_grad_bf16(params, input_ids, targets)
    print(f"  loss={float(loss_bf):.4f}  any_grad_nan={any(bool(jnp.isnan(g).any()) for g in jax.tree.leaves(grads_bf))}")

    print("fp8 value_and_grad...")
    loss_fp, grads_fp = loss_and_grad_fp8(params, input_ids, targets)
    print(f"  loss={float(loss_fp):.4f}  any_grad_nan={any(bool(jnp.isnan(g).any()) for g in jax.tree.leaves(grads_fp))}")
    return "done"


@app.local_entrypoint()
def main():
    print("--- scaled_matmul shape smoke ---")
    print(smoke.remote())
    print("--- full model forward smoke ---")
    print(smoke_model.remote())
