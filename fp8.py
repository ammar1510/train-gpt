"""Per-block-scaled FP8 matmul via real fp8 kernel (jax.nn.scaled_matmul).

Forward: quantize both operands to float8_e4m3fn with per-K-tile fp32 scales,
then call jax.nn.scaled_matmul. On B200 this dispatches to a real fp8 GEMM
(cuBLAS-LT / cuDNN) — the actual matmul runs in fp8 with fp32 accumulator,
giving the expected ~1.8x throughput vs bf16.

Backward: jax.nn.scaled_matmul has no built-in autodiff rule in JAX 0.10, so
we define a custom_vjp. The backward uses straight-through-style bf16 matmul
gradients with the original (un-quantized) operands. This pretends the forward
was a clean bf16 matmul for gradient purposes — standard fake-quant convention.
Forward keeps fp8 quantization noise; gradients are clean.

NOTE: production FP8 training typically uses e5m2 (not bf16) for the backward
gradient matmuls, with their own scaling. The bf16 backward here is a
simplification — it'll over-estimate gradient precision relative to a true
e5m2 backward. If A/B looks good, validate with a real e5m2 backward kernel
before committing to a long training run.
"""
import functools

import jax
import jax.numpy as jnp


E4M3_MAX = 448.0
DEFAULT_BLOCK_K = 128


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def block_scaled_matmul(a: jax.Array, b: jax.Array, block_k: int = DEFAULT_BLOCK_K) -> jax.Array:
    """Per-K-tile block-scaled FP8 matmul.

    a: (..., K), b: (K, N) → (..., N). Requires K % block_k == 0.
    """
    return _fp8_forward(a, b, block_k)


def _quantize_to_fp8(x_f32: jax.Array, block_k: int):
    *leading, K = x_f32.shape
    n_tiles = K // block_k
    x_tiled = x_f32.reshape(*leading, n_tiles, block_k)
    amax = jnp.max(jnp.abs(x_tiled), axis=-1)              # (..., n_tiles)
    scales = jnp.maximum(amax / E4M3_MAX, 1e-30)
    x_quant = x_tiled / scales[..., None]
    x_f8 = x_quant.astype(jnp.float8_e4m3fn).reshape(*leading, K)
    return x_f8, scales


def _fp8_forward(a: jax.Array, b: jax.Array, block_k: int) -> jax.Array:
    *leading, K = a.shape
    K_b, N = b.shape
    if K != K_b:
        raise ValueError(f"contracted-dim mismatch: a K={K}, b K={K_b}")
    if K % block_k != 0:
        raise ValueError(f"K={K} not divisible by block_k={block_k}")

    # NOTE: on B200 there is a real miscompile when the Pallas rms_norm kernel
    # and jax.nn.scaled_matmul live in the SAME XLA program -- the rms_norm
    # output comes back NaN (confirmed in debug_fp8.bridge). It is NOT fixable
    # with jax.lax.optimization_barrier (the corruption is module-level codegen,
    # not fusion reordering that a barrier can block): only a separate XLA
    # program, or the pure-XLA rms_norm path, avoids it. Callers that mix this op
    # with rms_norm must use the pure-XLA rms_norm. Independent of JAX 0.9.2/0.10.
    a_2d = a.reshape(-1, K)
    a_f8, a_scales = _quantize_to_fp8(a_2d.astype(jnp.float32), block_k)
    # scaled_matmul expects rhs as (B, N, K) — transpose b from (K, N).
    b_t_f8, b_scales = _quantize_to_fp8(b.T.astype(jnp.float32), block_k)

    out_f32 = jax.nn.scaled_matmul(
        a_f8[None],       # (1, M, K)
        b_t_f8[None],     # (1, N, K)
        a_scales[None],   # (1, M, K/block_k)
        b_scales[None],   # (1, N, K/block_k)
        preferred_element_type=jnp.float32,
    )[0]                  # (M, N)
    out_f32 = jax.lax.optimization_barrier(out_f32)
    return out_f32.astype(jnp.bfloat16).reshape(*leading, N)


def _fp8_forward_with_residuals(a, b, block_k):
    return _fp8_forward(a, b, block_k), (a, b)


def _fp8_backward(block_k, residuals, g):
    """Straight-through bf16 backward.

    g shape: (..., N). dL/da = g @ b.T → (..., K). dL/db = a^T @ g → (K, N).
    """
    a, b = residuals
    *leading, K = a.shape
    _, N = b.shape
    a_2d = a.reshape(-1, K).astype(jnp.bfloat16)
    g_2d = g.reshape(-1, N).astype(jnp.bfloat16)
    b_bf = b.astype(jnp.bfloat16)
    da = (g_2d @ b_bf.T).reshape(*leading, K).astype(a.dtype)
    db = (a_2d.T @ g_2d).astype(b.dtype)
    return da, db


block_scaled_matmul.defvjp(_fp8_forward_with_residuals, _fp8_backward)
