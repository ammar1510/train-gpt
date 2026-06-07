"""Root-cause harness for the bf16 Pallas rms_norm NaN under fusion.

Context: the custom Pallas rms_norm kernel (kernels.py) is finite in isolation
but NaNs once XLA fuses it into the model forward pass at batch >=64 (directly
observed in bf16; memory: batch128-bf16-divergence). This is NOT fp8-related —
no scaled_matmul anywhere in this file. The goal is to localize WHICH part of
the XLA<->kernel handoff breaks under fusion, then point at a fix that keeps the
kernel (vs. just shipping pure-XLA).

Prime suspect: the variance handoff.
    var   = mean(x^2)             # (N,)   computed in XLA, outside the kernel
    var32 = jnp.tile(var, (1,32)) # (N,32) a BROADCAST — XLA may keep it as a
                                  # stride-0 view / alias instead of a real
                                  # (N,32) buffer once fused. The kernel's TMA
                                  # descriptor assumes a contiguous (N,32) row,
                                  # so a broadcast view => garbage read => NaN.

Functions (run individually, e.g. `modal run debug_bf16_norm.py::sweep_batch`):
  sweep_batch  — find the batch threshold where pallas NaNs; pure stays finite.
  variants     — at a triggering batch, A/B the interventions (barrier,
                 materialized var, kernel-in-its-own-jit) against stock pallas.
  hlo          — dump the compiled custom-call operand layouts, isolated vs
                 fused, so we can SEE if var32 arrives as a broadcast when fused.

DEBUG ONLY. JAX 0.10 is not assumed; this pins the same 0.9.2 toolchain the
training run uses. Any version bump must be verified against the approved-tools
list (raise an IT/security ticket if not yet approved).
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "jax[cuda13]==0.9.2",
        "numpy==2.4.5",
        # absl-py is REQUIRED for the Mosaic GPU pallas backend to import; without
        # it `import jax.experimental.pallas.mosaic_gpu` raises ModuleNotFoundError
        # and pallas_call falls back to "cannot lower on platform: gpu". The other
        # Modal images got it transitively via optax/chex; pin it explicitly here
        # so the kernel lowers on its intended (default) Mosaic backend.
        "absl-py==2.1.0",
    )
    .add_local_python_source("config", "model", "kernels")
)

app = modal.App("train-gpt-bf16-norm")


# ---------------------------------------------------------------------------
# Configurable rms_norm builder — mirrors kernels._build_gpu_rms_norm exactly,
# with two toggles so we can A/B the suspected failure mode without touching
# the shipped kernel:
#   materialize_var : force var32 to a real buffer (optimization_barrier) so XLA
#                     cannot keep it as a stride-0 broadcast view.
#   barrier_inputs  : optimization_barrier on (x2d, scale, var32) right before
#                     the pallas_call — blocks fusion reordering / aliasing.
# Forward-only (no custom_vjp) — these diagnostics never call backward.
# ---------------------------------------------------------------------------
def build_rms_norm(materialize_var=False, barrier_inputs=False, backend=None,
                   fix_indexmap=False):
    import functools
    import math
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    # Backend selection via compiler_params:
    #   None         -> rely on the implicit default (Mosaic GPU) — faithful to
    #                   the shipped kernel, which passes nothing.
    #   "mosaic_gpu" -> select Mosaic GPU EXPLICITLY (same backend as None, but
    #                   clear, and the entry point for Mosaic-specific options).
    #   "triton"     -> select Triton via triton.CompilerParams().
    if backend == "triton":
        from jax.experimental.pallas import triton as pltri
        extra = {"compiler_params": pltri.CompilerParams()}
    elif backend == "mosaic_gpu":
        from jax.experimental.pallas import mosaic_gpu as plmgpu
        extra = {"compiler_params": plmgpu.CompilerParams()}
    else:
        extra = {}

    def _norm_kernel(x_ref, scale_ref, var_ref, out_ref, *, eps):
        x = x_ref[0, :].astype(jnp.float32)
        rnorm = jax.lax.rsqrt(var_ref[0, 0] + eps)
        out_ref[0, :] = (
            x * rnorm * scale_ref[0, :].astype(jnp.float32)
        ).astype(x_ref.dtype)

    @functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
    def rms_norm_gpu(x, scale, eps):
        orig = x.shape
        D = orig[-1]
        x2d = x.reshape(-1, D)
        N = x2d.shape[0]
        TILE_D = math.gcd(D, 256)
        N_TILES = D // TILE_D

        # BlockSpec index_maps return BLOCK indices; Pallas multiplies by the
        # block_shape to get the element offset. The shipped kernel returns the
        # column *byte offset* ((p % N_TILES) * TILE_D), which Pallas multiplies
        # by TILE_D AGAIN -> out of bounds -> clamped to the last tile, so the
        # middle output columns are never written (uninitialized -> NaN). The
        # fixed map returns the tile index (p % N_TILES). row map is unaffected
        # (block height 1). fix_indexmap toggles between the two.
        def map_x(p):
            col = (p % N_TILES) if fix_indexmap else (p % N_TILES) * TILE_D
            return (p // N_TILES, col)

        def map_scale(p):
            col = (p % N_TILES) if fix_indexmap else (p % N_TILES) * TILE_D
            return (0, col)

        var = jnp.mean(x2d.astype(jnp.float32) ** 2, axis=-1)   # (N,)
        var32 = jnp.tile(var[:, None], (1, 32))                  # (N, 32)
        if materialize_var:
            # Force a real (N,32) buffer: a barrier defeats the broadcast/alias
            # rewrite, so the kernel TMA-copies 32 contiguous valid floats.
            var32 = jax.lax.optimization_barrier(var32)

        scale_2d = scale[None, :]

        if barrier_inputs:
            x2d, scale_2d, var32 = jax.lax.optimization_barrier(
                (x2d, scale_2d, var32)
            )

        out2d = pl.pallas_call(
            functools.partial(_norm_kernel, eps=eps),
            out_shape=jax.ShapeDtypeStruct(x2d.shape, x2d.dtype),
            in_specs=[
                pl.BlockSpec((1, TILE_D), map_x),
                pl.BlockSpec((1, TILE_D), map_scale),
                pl.BlockSpec((1, 32),     lambda p: (p // N_TILES, 0)),
            ],
            out_specs=pl.BlockSpec((1, TILE_D), map_x),
            grid=(N * N_TILES,),
            **extra,
        )(x2d, scale_2d, var32)
        return out2d.reshape(orig)

    # Same custom_vjp as kernels.py: the backward is pure-JAX (recompute), so it
    # is independent of the forward index_map — but we wire it identically so
    # verify_fix exercises the real forward+backward composition.
    def _fwd(x, scale, eps):
        return rms_norm_gpu(x, scale, eps), (x, scale)

    def _bwd(eps, res, g):
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


def build_singlepass_rms_norm(rows_per_block=8, backend="mosaic_gpu"):
    """Single-pass rms_norm: one program handles ROWS full rows, computes the
    variance IN-KERNEL (one HBM read of x), normalizes, writes once. No separate
    XLA reduction, no var input, no (N,32) padding.

    Backend note: Triton requires power-of-2 array shapes, and D=2304 isn't one
    (padding to 4096 would read ~1.78x the data — pointless for a bandwidth-bound
    op), so we stay on Mosaic. The Mosaic ≤256 rule is a TMA (GMEM↔SMEM copy)
    granularity limit, not a ban on wide SMEM blocks — Pallas can emit several
    TMA transfers to bring a (ROWS, D) block in, after which the in-SMEM reduce
    over D is unconstrained. This is the load-once-reduce-in-place design the
    original kernel skipped."""
    import functools
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    if backend == "triton":
        from jax.experimental.pallas import triton as plg
        cp = plg.CompilerParams()
    else:
        from jax.experimental.pallas import mosaic_gpu as plg
        cp = plg.CompilerParams()

    import math

    def _kernel(x_ref, scale_ref, out_ref, *, eps):
        # blocks are (ROWS, N_TILES, TILE_D) — the row is split into N_TILES×TILE_D
        # so every TMA dim is ≤256, but the program still sees the FULL row and
        # reduces over both inner axes for the true per-row variance.
        x = x_ref[...].astype(jnp.float32)                  # (ROWS, NT, TILE)
        D = x.shape[1] * x.shape[2]
        # Mosaic reductions only take a single (last) axis, so reduce in two
        # steps: over TILE (axis 2, last of 3D), then over NT (axis 1, last of 2D).
        ss_tile = jnp.sum(x * x, axis=2)                    # (ROWS, NT)
        ss = jnp.sum(ss_tile, axis=1, keepdims=True)        # (ROWS, 1)
        rnorm = jax.lax.rsqrt(ss / D + eps)                 # (ROWS, 1)
        rnorm = rnorm.reshape(rnorm.shape[0], 1, 1)         # (ROWS, 1, 1) to broadcast
        s = scale_ref[...].astype(jnp.float32)              # (1, NT, TILE)
        out_ref[...] = (x * rnorm * s).astype(x_ref.dtype)

    @functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
    def rms(x, scale, eps):
        orig = x.shape
        D = orig[-1]
        x2d = x.reshape(-1, D)
        N = x2d.shape[0]
        R = rows_per_block
        if N % R:
            raise ValueError(f"N={N} not divisible by rows_per_block={R}")
        TILE = math.gcd(D, 256)
        NT = D // TILE
        x3 = x2d.reshape(N, NT, TILE)                       # split row into NT×TILE
        scale3 = scale.reshape(1, NT, TILE)
        out = pl.pallas_call(
            functools.partial(_kernel, eps=eps),
            out_shape=jax.ShapeDtypeStruct(x3.shape, x3.dtype),
            in_specs=[
                pl.BlockSpec((R, NT, TILE), lambda i: (i, 0, 0)),   # ROWS full rows
                pl.BlockSpec((1, NT, TILE), lambda i: (0, 0, 0)),   # scale, shared
            ],
            out_specs=pl.BlockSpec((R, NT, TILE), lambda i: (i, 0, 0)),
            grid=(N // R,),
            compiler_params=cp,
        )(x3, scale3)
        return out.reshape(orig)

    def _fwd(x, scale, eps):
        return rms(x, scale, eps), (x, scale)

    def _bwd(eps, res, g):
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

    rms.defvjp(_fwd, _bwd)
    return rms


@app.function(image=image, gpu="B200", memory=64 * 1024, timeout=20 * 60)
def bench_norm() -> str:
    """Time the three rms_norm impls at batch-128 scale (N=262144, D=2304),
    forward-only, and check correctness vs pure-XLA. Reports ms/call and the
    effective HBM bandwidth (a bandwidth-bound op should approach peak; the more
    traffic an impl moves, the lower its effective GB/s for the same work)."""
    import os
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    import time
    import jax
    import jax.numpy as jnp
    from kernels import _rms_norm_pure

    N, D = 262144, 2304          # batch 128 * seq 2048
    kx, ks = jax.random.split(jax.random.PRNGKey(0))
    x = jax.random.normal(kx, (N, D), dtype=jnp.bfloat16)
    scale = (jax.random.normal(ks, (D,)) * 0.5 + 1.0).astype(jnp.bfloat16)

    old = build_rms_norm(backend="mosaic_gpu", fix_indexmap=True)   # 2-pass, var32
    new = build_singlepass_rms_norm(rows_per_block=8)               # 1-pass, mosaic
    variants = [
        ("old pallas (mosaic, 2-pass)", old),
        ("single-pass (mosaic)", new),
        ("pure-XLA", _rms_norm_pure),
    ]
    ref = jax.block_until_ready(
        jax.jit(lambda x, s: _rms_norm_pure(x, s, 1e-5))(x, scale)
    ).astype(jnp.float32)

    # ideal traffic for rms_norm = read x + write out = 2 * N*D * 2 bytes (bf16)
    ideal_bytes = 2 * N * D * 2
    lines = [f"N={N} D={D}  (batch128-equiv)  ideal traffic={ideal_bytes/1e9:.2f} GB",
             f"{'impl':<30}  {'ms/call':>9}  {'eff GB/s':>9}  {'max_err':>10}"]
    K = 50
    for name, fn in variants:
        f = jax.jit(lambda x, s: fn(x, s, 1e-5))
        out = jax.block_until_ready(f(x, scale))         # compile + warmup
        err = float(jnp.max(jnp.abs(out.astype(jnp.float32) - ref)))
        t0 = time.perf_counter()
        for _ in range(K):
            out = f(x, scale)
        jax.block_until_ready(out)
        ms = (time.perf_counter() - t0) / K * 1000.0
        gbs = ideal_bytes / (ms / 1000.0) / 1e9
        lines.append(f"{name:<30}  {ms:>9.3f}  {gbs:>9.0f}  {err:>10.3e}")
    out_s = "\n".join(lines)
    print(out_s)
    return out_s


def _block_forward(x, lp, cfg, norm):
    """One transformer block in bf16, structured exactly like model.block, with
    the rms_norm impl swapped in. No fp8, no scan, no remat — just enough graph
    to make XLA fuse the kernel with the surrounding matmuls/attention."""
    import jax
    import jax.numpy as jnp

    B, T, _ = x.shape
    H, Dh = cfg.n_heads, cfg.head_dim
    impl = "cudnn" if jax.default_backend() == "gpu" else "xla"

    xn = norm(x, lp["norm1"], cfg.rms_eps)
    q = (xn @ lp["wq"]).reshape(B, T, H, Dh)
    k = (xn @ lp["wk"]).reshape(B, T, H, Dh)
    v = (xn @ lp["wv"]).reshape(B, T, H, Dh)
    attn = jax.nn.dot_product_attention(q, k, v, is_causal=True, implementation=impl)
    o = attn.reshape(B, T, H * Dh) @ lp["wo"]
    x = x + o
    xn2 = norm(x, lp["norm2"], cfg.rms_eps)
    h = jax.nn.relu(xn2 @ lp["w_up"]) ** 2
    d = h @ lp["w_down"]
    return x + d


def _make_inputs(B, T):
    import jax
    import jax.numpy as jnp
    from config import Config
    from model import init_params

    cfg = Config()
    key = jax.random.PRNGKey(0)
    params = init_params(key, cfg)
    input_ids = jax.random.randint(
        jax.random.PRNGKey(1), (B, T), 0, cfg.vocab_size, dtype=jnp.int32
    )
    x = params["embed"][input_ids] + params["pos"][:T]   # (B, T, d_model)
    lp = {k: params["layers"][k][0] for k in params["layers"]}
    return cfg, x, lp


@app.function(image=image, gpu="B200", memory=48 * 1024, timeout=20 * 60)
def probe() -> str:
    """One-shot environment + backend check. Run this FIRST.
    Establishes whether the *shipped* Pallas kernel even lowers in this image,
    so we know if the 'NaN under fusion' problem is real here or if we first
    have a backend-availability problem to solve. Each sub-test is caught so one
    failure doesn't hide the others."""
    import os
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    import jax
    import jax.numpy as jnp
    from importlib.metadata import version, PackageNotFoundError

    def ver(pkg):
        try:
            return version(pkg)
        except PackageNotFoundError:
            return "MISSING"

    # The version triple is the key diagnostic: jaxlib and the CUDA plugin must
    # match exactly, or the plugin's GPU pallas (Triton/Mosaic) lowerings never
    # register and pallas_call has no 'gpu' lowering -> the error we hit.
    lines = [f"jax        = {ver('jax')}  (runtime {jax.__version__})",
             f"jaxlib     = {ver('jaxlib')}",
             f"cuda13-plugin = {ver('jax-cuda13-plugin')}",
             f"cuda13-pjrt   = {ver('jax-cuda13-pjrt')}",
             f"default_backend = {jax.default_backend()}",
             f"devices = {jax.devices()}"]

    # Definitive Mosaic-vs-Triton availability check, via the PUBLIC modules and
    # a trivial kernel on each backend. Captures the EXACT error so we know
    # whether Mosaic is genuinely absent in this build or just misconfigured.
    import traceback

    def import_check(name, modpath):
        try:
            __import__(modpath)
            lines.append(f"import {name}: ok")
        except Exception as e:
            lines.append(f"import {name}: {type(e).__name__}: {str(e).splitlines()[0][:120]}")

    import_check("pallas.triton", "jax.experimental.pallas.triton")
    import_check("pallas.mosaic_gpu", "jax.experimental.pallas.mosaic_gpu")

    def tiny_kernel_check(name, cp):
        # Minimal add-one kernel; the simplest thing that must lower per backend.
        try:
            from jax.experimental import pallas as pl
            def k(x_ref, o_ref):
                o_ref[...] = x_ref[...] + 1.0
            f = pl.pallas_call(
                k,
                out_shape=jax.ShapeDtypeStruct((256,), jnp.float32),
                compiler_params=cp,
            )
            out = jax.block_until_ready(jax.jit(f)(jnp.zeros((256,), jnp.float32)))
            lines.append(f"tiny kernel [{name}]: ok (out[0]={float(out[0])})")
        except Exception as e:
            lines.append(f"tiny kernel [{name}]: {type(e).__name__}: {str(e).splitlines()[0][:120]}")

    tiny_kernel_check("default(None)", None)
    try:
        from jax.experimental.pallas import triton as pltri
        tiny_kernel_check("triton", pltri.CompilerParams())
    except Exception as e:
        lines.append(f"triton.CompilerParams unavailable: {type(e).__name__}")
    try:
        from jax.experimental.pallas import mosaic_gpu as plmgpu
        tiny_kernel_check("mosaic_gpu", plmgpu.CompilerParams())
    except Exception as e:
        lines.append(f"mosaic_gpu.CompilerParams unavailable: {type(e).__name__}")

    x = jax.random.normal(jax.random.PRNGKey(0), (256, 2304), dtype=jnp.bfloat16)
    scale = jnp.ones((2304,), dtype=jnp.bfloat16)
    from kernels import _rms_norm_pure
    ref = jax.block_until_ready(jax.jit(_rms_norm_pure)(x, scale, 1e-5)).astype(jnp.float32)

    def try_case(name, fn):
        try:
            out = jax.block_until_ready(jax.jit(fn)(x, scale))
            finite = bool(jnp.isfinite(out).all())
            # max abs error vs pure-XLA — distinguishes correct from finite-but-
            # wrong (e.g. out-of-bounds block-index clamping reads wrong columns).
            err = float(jnp.max(jnp.abs(out.astype(jnp.float32) - ref)))
            lines.append(f"[OK]   {name:<34} finite={finite}  max_err_vs_pure={err:.3e}")
        except Exception as e:
            msg = str(e).splitlines()[0][:140]
            lines.append(f"[FAIL] {name:<34} {type(e).__name__}: {msg}")

    # 1. The shipped kernel, exactly as the model uses it.
    try:
        from kernels import rms_norm as shipped
        try_case("shipped kernels.rms_norm", lambda x, s: shipped(x, s, 1e-5))
    except Exception as e:
        lines.append(f"[FAIL] import kernels.rms_norm: {type(e).__name__}: {e}")

    # 2. Harness kernel on the default (Mosaic GPU) backend — expected to FAIL,
    #    same as the shipped kernel, confirming the default-backend diagnosis.
    mine_default = build_rms_norm(backend=None)
    try_case("harness (default/mosaic)", lambda x, s: mine_default(x, s, 1e-5))

    # 3. Harness kernel forced onto Triton via compiler_params.
    mine_triton = build_rms_norm(backend="triton")
    try_case("harness (triton)", lambda x, s: mine_triton(x, s, 1e-5))

    # 4. The INDEX_MAP FIX, on each backend. If max_err_vs_pure ~ 0, the
    #    out-of-bounds column mapping was the whole bug.
    fix_mosaic = build_rms_norm(backend="mosaic_gpu", fix_indexmap=True)
    try_case("harness FIXED (mosaic_gpu)", lambda x, s: fix_mosaic(x, s, 1e-5))
    fix_triton = build_rms_norm(backend="triton", fix_indexmap=True)
    try_case("harness FIXED (triton)", lambda x, s: fix_triton(x, s, 1e-5))

    # --- Introspect the real backend-selection API on this exact jaxlib ---
    import inspect
    from jax.experimental import pallas as pl
    lines.append("\n--- pallas API introspection ---")
    try:
        lines.append(f"pallas_call signature: {inspect.signature(pl.pallas_call)}")
    except Exception as e:
        lines.append(f"(signature failed: {e})")
    # What backend/compiler-params knobs exist in the pallas namespace?
    knobs = [n for n in dir(pl) if any(
        t in n.lower() for t in ("backend", "compiler", "params", "triton", "mosaic")
    )]
    lines.append(f"pallas knobs: {knobs}")
    # The Triton submodule + its CompilerParams class (the likely selector).
    for modpath in ("jax.experimental.pallas.triton",
                    "jax.experimental.pallas.gpu"):
        try:
            m = __import__(modpath, fromlist=["*"])
            names = [n for n in dir(m) if not n.startswith("_")]
            lines.append(f"{modpath}: {names}")
        except Exception as e:
            lines.append(f"{modpath}: {type(e).__name__}")

    out = "\n".join(lines)
    print(out)          # stream to terminal under `modal run ::probe`
    return out


@app.function(image=image, gpu="B200", memory=48 * 1024, timeout=20 * 60)
def verify_fix() -> str:
    """Final correctness gate before patching kernels.py. Every prior test used
    scale=ones, which HIDES scale-tiling index_map errors (all columns equal).
    This uses a non-uniform scale, a 3D (B,T,D) input (the reshape path the model
    actually hits), and checks the custom_vjp BACKWARD too — all vs pure-XLA."""
    import os
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    import functools
    import jax
    import jax.numpy as jnp
    from kernels import _rms_norm_pure

    B, T, D = 4, 512, 2304
    kx, ks = jax.random.split(jax.random.PRNGKey(0))
    x = jax.random.normal(kx, (B, T, D), dtype=jnp.bfloat16)
    # NON-UNIFORM scale, with a wide per-column range so any mis-tiling shows up.
    scale = (jax.random.normal(ks, (D,)) * 0.5 + 1.0).astype(jnp.bfloat16)

    fixed = build_rms_norm(backend="mosaic_gpu", fix_indexmap=True)
    f_k = lambda x, s: fixed(x, s, 1e-5)
    f_p = lambda x, s: _rms_norm_pure(x, s, 1e-5)

    out_k = jax.block_until_ready(jax.jit(f_k)(x, scale)).astype(jnp.float32)
    out_p = jax.block_until_ready(jax.jit(f_p)(x, scale)).astype(jnp.float32)
    fwd_err = float(jnp.max(jnp.abs(out_k - out_p)))

    # Backward: grad of a scalar loss wrt x and scale, kernel vs pure.
    def loss(fn, x, s):
        return jnp.sum(fn(x, s).astype(jnp.float32) ** 2)
    gx_k, gs_k = jax.block_until_ready(jax.jit(jax.grad(functools.partial(loss, f_k), argnums=(0, 1)))(x, scale))
    gx_p, gs_p = jax.block_until_ready(jax.jit(jax.grad(functools.partial(loss, f_p), argnums=(0, 1)))(x, scale))
    gx_err = float(jnp.max(jnp.abs(gx_k.astype(jnp.float32) - gx_p.astype(jnp.float32))))
    gs_err = float(jnp.max(jnp.abs(gs_k.astype(jnp.float32) - gs_p.astype(jnp.float32))))

    def rel(a, b):
        a, b = jnp.asarray(a, jnp.float32), jnp.asarray(b, jnp.float32)
        return float(jnp.linalg.norm(a - b) / (jnp.linalg.norm(b) + 1e-9))

    out_s = "\n".join([
        f"shape={x.shape}  non-uniform scale in [{float(scale.min()):.2f},{float(scale.max()):.2f}]",
        f"forward : max_err={fwd_err:.3e}  rel={rel(out_k, out_p):.3e}  "
        f"finite={bool(jnp.isfinite(out_k).all())}",
        f"grad_x  : max_err={gx_err:.3e}  rel={rel(gx_k, gx_p):.3e}",
        f"grad_sc : max_err={gs_err:.3e}  rel={rel(gs_k, gs_p):.3e}",
    ])
    print(out_s)
    return out_s


@app.function(image=image, gpu="B200", memory=48 * 1024, timeout=20 * 60)
def localize() -> str:
    """The index_map fix made the kernel finite but wrong (max_err ~6.6e3).
    Localize the SECOND bug: split error by row vs by column.
      - error clusters by ROW  -> the per-row variance read (var32 / var_ref).
      - error clusters by COL  -> tiling / scale indexing.
    Also dump the worst row's kernel-vs-ref values and the implied rnorm."""
    import os
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    import jax
    import jax.numpy as jnp
    from kernels import _rms_norm_pure

    D = 2304
    fixed = build_rms_norm(backend="mosaic_gpu", fix_indexmap=True)
    pure = lambda x, s: _rms_norm_pure(x, s, 1e-5)

    # Sweep N: does the fixed kernel stay correct as size grows, or does the
    # 6640 error from the probe (N=256) reappear? Each N runs in isolation, so a
    # bad result here can't be contamination from earlier NaN kernels.
    lines = [f"D={D} TILE_D=256 N_TILES={D // 256}  (eps=1e-5, scale=ones)",
             f"{'N':>6}  {'max_err':>10}  {'mean_err':>10}  {'bad_rows(>1)':>12}"]
    for N in [64, 128, 256, 512, 1024]:
        x = jax.random.normal(jax.random.PRNGKey(0), (N, D), dtype=jnp.bfloat16)
        scale = jnp.ones((D,), dtype=jnp.bfloat16)
        out = jax.block_until_ready(
            jax.jit(lambda x, s: fixed(x, s, 1e-5))(x, scale)
        ).astype(jnp.float32)
        ref = jax.block_until_ready(
            jax.jit(pure)(x, scale)
        ).astype(jnp.float32)
        err = jnp.abs(out - ref)
        bad = int(jnp.sum(jnp.max(err, axis=1) > 1.0))
        lines.append(
            f"{N:>6}  {float(err.max()):>10.3e}  {float(err.mean()):>10.3e}  "
            f"{bad:>12}"
        )
    out_s = "\n".join(lines)
    print(out_s)
    return out_s


@app.function(image=image, gpu="B200", memory=48 * 1024, timeout=20 * 60)
def sweep_batch() -> str:
    """Sweep batch and report, for pure-XLA vs stock Pallas norm, whether the
    fused block output is finite. Pinpoints the threshold and confirms it's the
    kernel (pure stays finite where pallas flips to NaN)."""
    import os
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    import jax
    import jax.numpy as jnp
    # Test the ACTUAL shipped kernel (now patched in kernels.py), not the harness
    # replica — this validates the real code path under fusion at batch >=64.
    from kernels import _rms_norm_pure, rms_norm as shipped

    T = 512

    def fin(z):
        return bool(jnp.isfinite(z).all())

    lines = [f"jax {jax.__version__}   T={T}   (pallas = patched kernels.rms_norm, Mosaic)",
             f"{'batch':>6}  {'N=B*T':>9}  {'pure finite':>12}  {'pallas finite':>14}"]
    for B in [8, 32, 64, 128, 256]:
        cfg, x, lp = _make_inputs(B, T)
        f_pure = jax.jit(lambda x, lp: _block_forward(x, lp, cfg, _rms_norm_pure))
        f_pal = jax.jit(lambda x, lp: _block_forward(x, lp, cfg, shipped))
        out_pure = jax.block_until_ready(f_pure(x, lp))
        out_pal = jax.block_until_ready(f_pal(x, lp))
        lines.append(
            f"{B:>6}  {B*T:>9}  {str(fin(out_pure)):>12}  {str(fin(out_pal)):>14}"
        )
    out = "\n".join(lines)
    print(out)
    return out


@app.function(image=image, gpu="B200", memory=48 * 1024, timeout=20 * 60)
def variants() -> str:
    """At a triggering batch, A/B the interventions against the stock kernel.
    Whichever flips pallas back to finite names the mechanism:
      - materialize_var fixes it  => var32 broadcast/alias is the cause.
      - barrier_inputs fixes it    => fusion-ordering/aliasing (not just var32).
      - only kernel-in-own-jit fixes it => module-level codegen (matches the fp8
        finding in fp8.py:58 — barrier can't help, only a separate program)."""
    import os
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    import jax
    import jax.numpy as jnp
    from kernels import _rms_norm_pure

    B, T = 128, 512   # adjust if sweep_batch shows a different threshold
    cfg, x, lp = _make_inputs(B, T)

    def fin(z):
        return bool(jnp.isfinite(z).all())

    stock = build_rms_norm()
    matvar = build_rms_norm(materialize_var=True)
    barrier = build_rms_norm(barrier_inputs=True)

    def block_jit(norm):
        # Whole block fused in ONE compiled program (the failing config).
        return jax.block_until_ready(
            jax.jit(lambda x, lp: _block_forward(x, lp, cfg, norm))(x, lp)
        )

    def block_eager_norm_jit():
        # Block runs EAGERLY (matmuls dispatch one-op-at-a-time); only the norm
        # is jitted, so each norm call is its OWN compiled program — never in the
        # same module as the matmuls. This is the "separate XLA program" escape
        # hatch from fp8.py:58. Nesting jax.jit-in-jax.jit would just inline, so
        # the outer block must NOT be jitted here.
        return jax.block_until_ready(_block_forward(x, lp, cfg, jax.jit(stock)))

    cases = [
        ("pure-XLA (control)", lambda: block_jit(_rms_norm_pure)),
        ("pallas stock (fused)", lambda: block_jit(stock)),
        ("pallas + materialize_var", lambda: block_jit(matvar)),
        ("pallas + barrier_inputs", lambda: block_jit(barrier)),
        ("pallas norm in own program", block_eager_norm_jit),
    ]
    lines = [f"jax {jax.__version__}   B={B} T={T} (N={B*T})"]
    for name, run in cases:
        lines.append(f"{name:<28} finite={str(fin(run())):>5}")
    out = "\n".join(lines)
    print(out)
    return out


@app.function(image=image, gpu="B200", memory=48 * 1024, timeout=20 * 60)
def hlo() -> str:
    """Dump the compiled custom-call lines for the Pallas norm, isolated vs
    fused into the block. If var32 arrives as a broadcast/stride-0 operand (or
    the custom-call gains an aliasing annotation) only in the fused HLO, that's
    the smoking gun. Uses lower().compile().as_text() (post-XLA passes)."""
    import os
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    import jax

    B, T = 128, 512
    cfg, x, lp = _make_inputs(B, T)
    stock = build_rms_norm()

    iso = jax.jit(lambda x: stock(x, lp["norm1"], cfg.rms_eps))
    fused = jax.jit(lambda x, lp: _block_forward(x, lp, cfg, stock))

    hlo_iso = iso.lower(x).compile().as_text()
    hlo_fused = fused.lower(x, lp).compile().as_text()

    def relevant(label, text):
        out = [f"\n{'='*64}\n  {label}\n{'='*64}"]
        for line in text.splitlines():
            if any(kw in line for kw in (
                "custom-call", "custom_call", "mosaic", "broadcast",
                "bitcast", "copy", "fusion", "rsqrt",
            )):
                out.append(line.rstrip())
        return "\n".join(out)

    out = (relevant("ISOLATED: rms_norm alone", hlo_iso)
           + "\n"
           + relevant("FUSED: rms_norm inside full block", hlo_fused))
    print(out)
    return out


@app.local_entrypoint()
def main():
    print("=== batch sweep: pure vs pallas, finiteness of fused block ===")
    print(sweep_batch.remote())
    print("\n\n=== interventions at the triggering batch ===")
    print(variants.remote())
