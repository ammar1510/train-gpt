"""Pallas GPU kernels for performance-critical ops.

Automatically falls back to pure JAX when running on CPU (e.g. test_overfit).

Mosaic GPU (JAX 0.9.x default on CUDA 13) uses TMA for async GMEM→SMEM copies.
Four TMA constraints must all be satisfied:
  1. Each tile dimension must be <= 256 elements.
  2. All GMEM strides *except the last* must be multiples of 16 bytes.
  3. The last dimension must transfer a number of bits divisible by 128.
  4. Total bytes per block must be divisible by the warpgroup size (128 bytes).

Constraints 3 & 4 rule out tiny scalar tiles; Mosaic GPU also requires 2D
SMEM buffers (1D raises WGStridedFragLayout errors) with element counts
divisible by 128. The kernel processes ROWS_PER_BLOCK=128 rows at a time;
variance is passed as (N, 4) float32 so the block (128, 4) = 2048 bytes
satisfies every rule. Variance is computed in pure JAX (XLA fuses
cast+square+mean); Pallas handles only normalize+scale.
"""
import functools
import math

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Pure-JAX fallback (CPU path, e.g. test_overfit.py)
# ---------------------------------------------------------------------------

def _rms_norm_pure(x, scale, eps):
    var = jnp.mean(x.astype(jnp.float32) ** 2, axis=-1, keepdims=True)
    x_n = x * jax.lax.rsqrt(var + eps).astype(x.dtype)
    return x_n * scale


# ---------------------------------------------------------------------------
# Pallas fused normalize+scale kernel (GPU only)
# Variance is computed outside in pure JAX, then tiled to (N, 4) for TMA.
# ---------------------------------------------------------------------------

def _build_gpu_rms_norm():
    from jax.experimental import pallas as pl

    def _norm_kernel(x_ref, scale_ref, var_ref, out_ref, *, eps: float):
        # x_ref, out_ref : (1, TILE_D) model dtype
        # scale_ref      : (1, TILE_D) model dtype
        # var_ref        : (1, 32) float32 — var_ref[0, 0] is the row variance;
        #                  cols 1–31 are padding (TMA requires 128-byte blocks)
        x = x_ref[0, :].astype(jnp.float32)
        rnorm = jax.lax.rsqrt(var_ref[0, 0] + eps)   # scalar load, not a slice
        out_ref[0, :] = (
            x * rnorm * scale_ref[0, :].astype(jnp.float32)
        ).astype(x_ref.dtype)

    @functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
    def rms_norm_gpu(x: jax.Array, scale: jax.Array, eps: float) -> jax.Array:
        orig = x.shape
        D = orig[-1]
        x2d = x.reshape(-1, D)         # (N, D)
        N = x2d.shape[0]

        TILE_D = math.gcd(D, 256)
        N_TILES = D // TILE_D

        # Variance in pure JAX — XLA fuses cast+square+mean into one kernel.
        var = jnp.mean(x2d.astype(jnp.float32) ** 2, axis=-1)  # (N,) float32

        # Mosaic GPU requires 2D SMEM buffers — 1D raises WGStridedFragLayout
        # errors even at valid sizes. Scalar loads (var_ref[0, 0]) work; tensor
        # slices (var_ref[:, 0]) do not.
        # Grid is flattened to 1D so only blockIdx.x (limit 2^31-1) is used; the
        # 65,535 cap on blockIdx.y/z applies for any multi-axis grid mapping and
        # the Mosaic→CUDA axis mapping is not guaranteed by Pallas. Row and
        # tile indices are recovered with divmod inside the BlockSpec lambdas.
        # Pad to (N, 32): block (1, 32) = 128 bytes satisfies all four TMA rules.
        var32 = jnp.tile(var[:, None], (1, 32))   # (N, 32) float32

        scale_2d = scale[None, :]                  # (1, D)

        out2d = pl.pallas_call(
            functools.partial(_norm_kernel, eps=eps),
            out_shape=jax.ShapeDtypeStruct(x2d.shape, x2d.dtype),
            in_specs=[
                pl.BlockSpec((1, TILE_D), lambda p: (p // N_TILES, (p % N_TILES) * TILE_D)),  # x
                pl.BlockSpec((1, TILE_D), lambda p: (0, (p % N_TILES) * TILE_D)),             # scale
                pl.BlockSpec((1, 32),     lambda p: (p // N_TILES, 0)),                       # var
            ],
            out_specs=pl.BlockSpec((1, TILE_D), lambda p: (p // N_TILES, (p % N_TILES) * TILE_D)),
            grid=(N * N_TILES,),
        )(x2d, scale_2d, var32)

        return out2d.reshape(orig)

    def _fwd(x, scale, eps):
        return rms_norm_gpu(x, scale, eps), (x, scale)

    def _bwd(eps, res, g):
        # Recompute normed from saved x — avoids storing it as a residual.
        x, scale = res
        x32 = x.astype(jnp.float32)
        var = jnp.mean(x32 * x32, axis=-1, keepdims=True)
        rnorm = jax.lax.rsqrt(var + eps)
        normed = x32 * rnorm
        g32 = g.astype(jnp.float32)
        gs = g32 * scale
        dot = jnp.mean(gs * normed, axis=-1, keepdims=True)
        grad_x = (rnorm * (gs - normed * dot)).astype(x.dtype)
        grad_scale = jnp.sum(
            g32 * normed, axis=tuple(range(normed.ndim - 1))
        ).astype(scale.dtype)
        return grad_x, grad_scale

    rms_norm_gpu.defvjp(_fwd, _bwd)
    return rms_norm_gpu


if jax.default_backend() == "gpu":
    rms_norm = _build_gpu_rms_norm()
else:
    rms_norm = _rms_norm_pure
