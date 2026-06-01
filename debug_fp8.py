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

# Same toolchain but JAX bumped to 0.10 — used only to test whether the
# multi-call scaled_matmul NaN is a 0.9.2 kernel bug fixed upstream. DEBUG ONLY:
# any real use of this version must be verified against the approved-tools list
# (raise an IT/security ticket if 0.10 is not yet approved).
image_010 = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "jax[cuda13]==0.10.0",
        "optax==0.2.8",
        "numpy==2.4.5",
        "chex==0.1.91",
    )
    .add_local_python_source("config", "model", "losses", "data", "kernels", "fp8")
)

app = modal.App("train-gpt-fp8-smoke")


def _run_sweep(cfg_cls, init_params, block_scaled_matmul):
    """Shared body for the layer-count sweep, so it can run under either JAX
    image. Returns a printable report string."""
    import dataclasses
    import math
    import jax
    import jax.numpy as jnp

    cfg = cfg_cls()
    B, T = 2, 512
    key = jax.random.PRNGKey(0)
    params_full = init_params(key, cfg)
    input_ids = jax.random.randint(
        jax.random.PRNGKey(1), (B, T), 0, cfg.vocab_size, dtype=jnp.int32
    )
    mm_fp8 = lambda a, b: block_scaled_matmul(a, b)

    lines = [f"jax {jax.__version__}",
             f"{'n_layers':>9}  {'#scaled_mm':>11}  {'L00/q finite':>13}  {'first NaN op':>14}"]
    for n in [1, 2, 4, 8, 24]:
        cfg_n = dataclasses.replace(cfg, n_layers=n)
        params_n = dict(params_full)
        params_n["layers"] = {k: v[:n] for k, v in params_full["layers"].items()}
        trace = jax.jit(lambda p, ids: _trace_forward(p, ids, cfg_n, mm_fp8))
        rec = jax.block_until_ready(trace(params_n, input_ids))
        q0 = float(rec["q"][0])
        first_nan = "none"
        for i in range(n):
            for op in OP_ORDER:
                if not math.isfinite(float(rec[op][i])):
                    first_nan = f"L{i:02d}/{op}"
                    break
            if first_nan != "none":
                break
        lines.append(f"{n:>9}  {n * 6:>11}  {str(math.isfinite(q0)):>13}  {first_nan:>14}")
    return "\n".join(lines)


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


# Order in which a block's matmul outputs are produced. Used both to drive the
# trace and to report the FIRST point where fp8 magnitude diverges from bf16.
OP_ORDER = ["q", "k", "v", "attn", "o", "res1", "h_up", "act", "d_down", "res2"]


def _trace_forward(params, input_ids, cfg, mm, norm=None):
    """Unrolled (no scan) forward that records amax after every op, for one
    matmul implementation `mm(a, b) -> a@b`. Returns {op_name: (n_layers,) fp32}
    plus a "logits" entry. Forward-only — no grad, no checkpoint. `norm` selects
    the rms_norm implementation (defaults to the Pallas kernel)."""
    import jax
    import jax.numpy as jnp
    from kernels import rms_norm as _pallas_rms_norm
    rms_norm = norm if norm is not None else _pallas_rms_norm

    B, T = input_ids.shape
    H, Dh = cfg.n_heads, cfg.head_dim
    impl = "cudnn" if jax.default_backend() == "gpu" else "xla"

    x = params["embed"][input_ids] + params["pos"][:T]
    layers = params["layers"]
    rec = {name: [] for name in OP_ORDER}

    def amax(z):
        # max|.| in fp32 so inf/nan propagate visibly instead of saturating bf16.
        return jnp.max(jnp.abs(z.astype(jnp.float32)))

    for i in range(cfg.n_layers):
        lp = {k: layers[k][i] for k in layers}

        xn = rms_norm(x, lp["norm1"], cfg.rms_eps)
        q = mm(xn, lp["wq"]); rec["q"].append(amax(q))
        k = mm(xn, lp["wk"]); rec["k"].append(amax(k))
        v = mm(xn, lp["wv"]); rec["v"].append(amax(v))
        q = q.reshape(B, T, H, Dh)
        k = k.reshape(B, T, H, Dh)
        v = v.reshape(B, T, H, Dh)
        attn = jax.nn.dot_product_attention(
            q, k, v, is_causal=True, implementation=impl
        )
        rec["attn"].append(amax(attn))
        o = mm(attn.reshape(B, T, H * Dh), lp["wo"]); rec["o"].append(amax(o))
        x = x + o; rec["res1"].append(amax(x))

        xn2 = rms_norm(x, lp["norm2"], cfg.rms_eps)
        h = mm(xn2, lp["w_up"]); rec["h_up"].append(amax(h))
        act = jax.nn.relu(h) ** 2; rec["act"].append(amax(act))
        d = mm(act, lp["w_down"]); rec["d_down"].append(amax(d))
        x = x + d; rec["res2"].append(amax(x))

    x = rms_norm(x, params["final_norm"], cfg.rms_eps)
    w_out = params["embed"].T if cfg.tie_embeddings else params["unembed"]
    logits = x @ w_out

    out = {name: jnp.stack(rec[name]) for name in OP_ORDER}
    out["logits"] = jnp.stack([amax(logits)])
    return out


@app.function(image=image, gpu="B200", memory=48 * 1024, timeout=20 * 60)
def trace_amax() -> str:
    """Run the FULL-config forward in bf16 and fp8 on identical params/input,
    recording amax after every matmul, and report the first op where fp8
    magnitude blows past bf16. Pinpoints which matmul detonates."""
    import os
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    import math
    import jax
    import jax.numpy as jnp
    from config import Config
    from model import init_params
    from fp8 import block_scaled_matmul

    cfg = Config()
    # Small batch/seq is enough: the fp8 blow-up is per-element, not batch-driven.
    B, T = 2, 512

    key = jax.random.PRNGKey(0)
    params = init_params(key, cfg)
    input_ids = jax.random.randint(
        jax.random.PRNGKey(1), (B, T), 0, cfg.vocab_size, dtype=jnp.int32
    )

    mm_bf16 = lambda a, b: a @ b
    mm_fp8 = lambda a, b: block_scaled_matmul(a, b)

    trace_bf16 = jax.jit(lambda p, ids: _trace_forward(p, ids, cfg, mm_bf16))
    trace_fp8 = jax.jit(lambda p, ids: _trace_forward(p, ids, cfg, mm_fp8))

    print("running bf16 trace...")
    rec_bf = jax.block_until_ready(trace_bf16(params, input_ids))
    print("running fp8 trace...")
    rec_fp = jax.block_until_ready(trace_fp8(params, input_ids))

    # Materialize to host as plain floats.
    bf = {k: [float(z) for z in v] for k, v in rec_bf.items()}
    fp = {k: [float(z) for z in v] for k, v in rec_fp.items()}

    print(f"\n{'layer/op':>14}  {'bf16 amax':>12}  {'fp8 amax':>12}  {'ratio':>10}")
    print("-" * 56)
    first_bad = None
    for i in range(cfg.n_layers):
        for op in OP_ORDER:
            b_val, f_val = bf[op][i], fp[op][i]
            finite = math.isfinite(f_val)
            ratio = (f_val / b_val) if (finite and b_val > 0) else float("inf")
            flag = ""
            if (not finite) or ratio > 2.0 or ratio < 0.5:
                flag = "  <-- DIVERGE"
                if first_bad is None:
                    first_bad = (i, op, b_val, f_val)
            # Print the first 4 layers in full; after that only flagged rows.
            if i < 4 or flag:
                rstr = "inf" if math.isinf(ratio) else f"{ratio:8.2f}x"
                print(f"L{i:02d}/{op:<8}  {b_val:12.3e}  {f_val:12.3e}  "
                      f"{rstr:>10}{flag}")
        if first_bad and i >= first_bad[0] + 1:
            print("  ... (stopping detail after first divergence layer) ...")
            break

    print(f"\n{'logits':>14}  {bf['logits'][0]:12.3e}  {fp['logits'][0]:12.3e}")
    if first_bad:
        i, op, b_val, f_val = first_bad
        print(f"\nFIRST DIVERGENCE: layer {i}, op '{op}': "
              f"bf16={b_val:.3e}  fp8={f_val:.3e}")
    else:
        print("\nNo divergence > 2x detected anywhere — fp8 forward is clean.")
    return "done"


@app.function(image=image, gpu="B200", memory=48 * 1024, timeout=20 * 60)
def isolate() -> str:
    """Isolate the first Q projection. Runs block_scaled_matmul on the model's
    ACTUAL layer-0 inputs in three contexts to separate data/shape failure from
    big-graph context failure:
      (1) eager (no jit)
      (2) standalone small jit
      (3) same inputs, but b given as a fresh contiguous copy (rules out the
          vmap-stacked weight slice being the problem)
    """
    import os
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    import math
    import jax
    import jax.numpy as jnp
    from config import Config
    from model import init_params
    from kernels import rms_norm
    from fp8 import block_scaled_matmul

    cfg = Config()
    B, T = 2, 512
    key = jax.random.PRNGKey(0)
    params = init_params(key, cfg)
    input_ids = jax.random.randint(
        jax.random.PRNGKey(1), (B, T), 0, cfg.vocab_size, dtype=jnp.int32
    )

    # Reproduce layer-0's Q-projection input exactly.
    x = params["embed"][input_ids] + params["pos"][:T]
    xn = rms_norm(x, params["layers"]["norm1"][0], cfg.rms_eps)
    wq = params["layers"]["wq"][0]                       # vmap-stacked slice
    wq_copy = jnp.array(jax.device_get(wq))              # fresh contiguous copy

    print(f"xn:  shape={xn.shape} dtype={xn.dtype} finite={bool(jnp.isfinite(xn).all())} amax={float(jnp.abs(xn).max()):.3e}")
    print(f"wq:  shape={wq.shape} dtype={wq.dtype} finite={bool(jnp.isfinite(wq).all())} amax={float(jnp.abs(wq).max()):.3e}")

    ref = (xn.astype(jnp.float32) @ wq.astype(jnp.float32))
    print(f"bf16 ref: finite={bool(jnp.isfinite(ref).all())} amax={float(jnp.abs(ref).max()):.3e}")

    def report(tag, out):
        finite = bool(jnp.isfinite(out).all())
        amax = float(jnp.abs(jnp.where(jnp.isfinite(out), out, 0.0)).max())
        rel = float(jnp.linalg.norm((out.astype(jnp.float32) - ref))
                    / jnp.linalg.norm(ref))
        print(f"{tag:<28} finite={finite}  amax={amax:.3e}  rel_err={rel:.4f}")

    report("(1) eager", block_scaled_matmul(xn, wq))
    report("(2) small jit", jax.jit(block_scaled_matmul)(xn, wq))
    report("(3) small jit, wq copy", jax.jit(block_scaled_matmul)(xn, wq_copy))
    return "done"


@app.function(image=image, gpu="B200", memory=48 * 1024, timeout=20 * 60)
def sweep_layers() -> str:
    """How many fp8 scaled_matmul calls in ONE graph does it take to NaN?
    Slices the SAME params so layer-0 weights are identical across runs — only
    the graph size changes. If L00/q stays finite for small N and flips to NaN
    at some threshold, the failure is graph-size/workspace-driven, not data."""
    import os
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    import dataclasses
    import math
    import jax
    import jax.numpy as jnp
    from config import Config
    from model import init_params
    from fp8 import block_scaled_matmul

    cfg = Config()
    B, T = 2, 512
    key = jax.random.PRNGKey(0)
    params_full = init_params(key, cfg)
    input_ids = jax.random.randint(
        jax.random.PRNGKey(1), (B, T), 0, cfg.vocab_size, dtype=jnp.int32
    )
    mm_fp8 = lambda a, b: block_scaled_matmul(a, b)

    print(f"{'n_layers':>9}  {'#scaled_mm':>11}  {'L00/q finite':>13}  {'first NaN op':>14}")
    print("-" * 56)
    for n in [1, 2, 3, 4, 6, 8, 12, 24]:
        cfg_n = dataclasses.replace(cfg, n_layers=n)
        params_n = dict(params_full)
        # Slice the stacked layer weights to the first n — layer-0 unchanged.
        params_n["layers"] = {
            k: v[:n] for k, v in params_full["layers"].items()
        }
        trace = jax.jit(lambda p, ids: _trace_forward(p, ids, cfg_n, mm_fp8))
        rec = jax.block_until_ready(trace(params_n, input_ids))
        q0 = float(rec["q"][0])
        first_nan = "none"
        for i in range(n):
            for op in OP_ORDER:
                if not math.isfinite(float(rec[op][i])):
                    first_nan = f"L{i:02d}/{op}"
                    break
            if first_nan != "none":
                break
        print(f"{n:>9}  {n * 6:>11}  {str(math.isfinite(q0)):>13}  {first_nan:>14}")
    return "done"


@app.function(image=image, gpu="B200", memory=48 * 1024, timeout=20 * 60)
def minrepro() -> str:
    """Characterize WHICH combination first triggers the NaN, on random inputs
    of model shapes. Each case is its own jit (one graph). Reports whether the
    first matmul's output is finite — if a case flips finite->NaN vs the prior,
    that added element is the trigger."""
    import os
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    import jax
    import jax.numpy as jnp
    from kernels import rms_norm
    from fp8 import block_scaled_matmul as mm

    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    M, K, N, F = 1024, 2304, 2304, 9216
    a = jax.random.normal(k1, (M, K), dtype=jnp.bfloat16)
    a3 = a.reshape(2, 512, K)
    w = jax.random.normal(k2, (K, N), dtype=jnp.bfloat16) * 0.02
    w_up = jax.random.normal(k3, (K, F), dtype=jnp.bfloat16) * 0.02
    w_down = jax.random.normal(k4, (F, N), dtype=jnp.bfloat16) * 0.02

    def fin(x):
        return bool(jnp.isfinite(x).all())

    cases = {}

    # A: single matmul (baseline — expect finite, matches isolate)
    cases["A: 1 mm"] = jax.jit(lambda: mm(a, w))

    # B: two independent matmuls, same input
    cases["B: 2 mm (indep)"] = jax.jit(lambda: (mm(a, w), mm(a, w)))

    # C: two chained matmuls
    cases["C: 2 mm (chain)"] = jax.jit(lambda: (lambda o: (o, mm(o, w)))(mm(a, w)))

    # D: six matmuls, no attention, no norm (q,k,v,o,up,down pattern)
    def six():
        q = mm(a, w); k = mm(a, w); v = mm(a, w)
        o = mm(q + k + v, w)
        up = jax.nn.relu(mm(o, w_up)) ** 2
        dn = mm(up, w_down)
        return q, k, v, o, up, dn
    cases["D: 6 mm (no attn)"] = jax.jit(six)

    # E: matmul + cuDNN attention in one graph
    def with_attn():
        q = mm(a3, w).reshape(2, 512, 18, 128)
        kk = mm(a3, w).reshape(2, 512, 18, 128)
        vv = mm(a3, w).reshape(2, 512, 18, 128)
        att = jax.nn.dot_product_attention(q, kk, vv, is_causal=True, implementation="cudnn")
        return q, att
    cases["E: 3 mm + cudnn attn"] = jax.jit(with_attn)

    # F: matmul + Pallas rms_norm in one graph
    def with_norm():
        xn = rms_norm(a3, jnp.ones((K,), jnp.bfloat16), 1e-5)
        return mm(xn, w)
    cases["F: rms_norm + 1 mm"] = jax.jit(with_norm)

    lines = []
    for name, fn in cases.items():
        out = jax.block_until_ready(fn())
        leaves = jax.tree_util.tree_leaves(out)
        first_finite = fin(leaves[0])
        all_finite = all(fin(x) for x in leaves)
        lines.append(f"{name:<22} first_finite={str(first_finite):>5}  all_finite={str(all_finite):>5}")
    return "\n".join(lines)


@app.function(image=image_010, gpu="B200", memory=48 * 1024, timeout=20 * 60)
def sweep_010() -> str:
    """The layer-count sweep, but on JAX 0.10. If the multi-call NaN is a 0.9.2
    kernel bug fixed upstream, this stays finite where sweep_layers went NaN."""
    import os
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    from config import Config
    from model import init_params
    from fp8 import block_scaled_matmul
    return _run_sweep(Config, init_params, block_scaled_matmul)


@app.function(image=image, gpu="B200", memory=48 * 1024, timeout=20 * 60)
def bridge() -> str:
    """Bridge isolate (finite) -> sweep n=1 (NaN) using REAL layer-0 data, with
    rms_norm INSIDE the jit (the key difference from isolate). Truncate the
    block at progressive points; the first truncation that NaNs names the
    trigger. Also tests rms_norm in-jit vs xn pre-materialized, and a non-fused
    (XLA) rms_norm, to confirm/deny the Pallas-rms_norm + scaled_matmul theory."""
    import os
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    import jax
    import jax.numpy as jnp
    from config import Config
    from model import init_params
    from kernels import rms_norm
    from kernels import _rms_norm_pure
    from fp8 import block_scaled_matmul as mm

    cfg = Config()
    B, T, H, Dh = 2, 512, cfg.n_heads, cfg.head_dim
    key = jax.random.PRNGKey(0)
    params = init_params(key, cfg)
    input_ids = jax.random.randint(
        jax.random.PRNGKey(1), (B, T), 0, cfg.vocab_size, dtype=jnp.int32
    )
    x0 = params["embed"][input_ids] + params["pos"][:T]          # tiny (~0.02)
    lp = {k: params["layers"][k][0] for k in params["layers"]}
    eps = cfg.rms_eps

    def fin(x):
        return bool(jnp.isfinite(x).all())

    # Each fn returns q first so we can check the first matmul specifically.
    def t1_rmsnorm_in_jit(x):                 # rms_norm fused INTO the jit
        xn = rms_norm(x, lp["norm1"], eps)
        return mm(xn, lp["wq"]),

    def t1_xn_premat(xn):                      # xn precomputed (== isolate)
        return mm(xn, lp["wq"]),

    def t1_pure_rmsnorm(x):                    # XLA (non-Pallas) rms_norm in jit
        xn = _rms_norm_pure(x, lp["norm1"], eps)
        return mm(xn, lp["wq"]),

    def t2_qkv(x):
        xn = rms_norm(x, lp["norm1"], eps)
        return mm(xn, lp["wq"]), mm(xn, lp["wk"]), mm(xn, lp["wv"])

    def t_full_block(x):
        xn = rms_norm(x, lp["norm1"], eps)
        q = mm(xn, lp["wq"]).reshape(B, T, H, Dh)
        k = mm(xn, lp["wk"]).reshape(B, T, H, Dh)
        v = mm(xn, lp["wv"]).reshape(B, T, H, Dh)
        att = jax.nn.dot_product_attention(q, k, v, is_causal=True, implementation="cudnn")
        o = mm(att.reshape(B, T, H * Dh), lp["wo"])
        x = x + o
        xn2 = rms_norm(x, lp["norm2"], eps)
        up = jax.nn.relu(mm(xn2, lp["w_up"])) ** 2
        dn = mm(up, lp["w_down"])
        return (q,)   # report q (the first matmul) — does adding the rest break it?

    xn_pre = rms_norm(x0, lp["norm1"], eps)    # materialize outside jit

    cases = [
        ("t1: rms_norm(pallas) in jit + 1mm", jax.jit(t1_rmsnorm_in_jit), x0),
        ("t1: xn pre-materialized (isolate)", jax.jit(t1_xn_premat), xn_pre),
        ("t1: rms_norm(pure/xla) in jit + 1mm", jax.jit(t1_pure_rmsnorm), x0),
        ("t2: rms_norm + q,k,v", jax.jit(t2_qkv), x0),
        ("t_full: whole block (report q)", jax.jit(t_full_block), x0),
    ]
    lines = []
    for name, fn, arg in cases:
        out = jax.block_until_ready(fn(arg))
        leaves = jax.tree_util.tree_leaves(out)
        lines.append(f"{name:<38} q_finite={str(fin(leaves[0])):>5}")
    return "\n".join(lines)


@app.function(image=image, gpu="B200", memory=48 * 1024, timeout=20 * 60)
def trace_fp8_pure() -> str:
    """Full 24-layer fp8 forward, but with the PURE-XLA rms_norm (not the Pallas
    kernel). If the root cause is the Pallas-rms_norm + scaled_matmul module
    miscompile, this should be finite end-to-end."""
    import os
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    import math
    import jax
    import jax.numpy as jnp
    from config import Config
    from model import init_params
    from kernels import _rms_norm_pure
    from fp8 import block_scaled_matmul

    cfg = Config()
    B, T = 2, 512
    key = jax.random.PRNGKey(0)
    params = init_params(key, cfg)
    input_ids = jax.random.randint(
        jax.random.PRNGKey(1), (B, T), 0, cfg.vocab_size, dtype=jnp.int32
    )
    mm_fp8 = lambda a, b: block_scaled_matmul(a, b)
    trace = jax.jit(
        lambda p, ids: _trace_forward(p, ids, cfg, mm_fp8, norm=_rms_norm_pure)
    )
    rec = jax.block_until_ready(trace(params, input_ids))
    q0 = float(rec["q"][0])
    logit = float(rec["logits"][0])
    all_finite = all(
        math.isfinite(float(z)) for v in rec.values() for z in v
    )
    return (f"fp8 + pure-XLA rms_norm: L00/q={q0:.3e}  logits_amax={logit:.3e}  "
            f"all_ops_finite={all_finite}")


@app.local_entrypoint()
def main():
    print("--- full 24-layer fp8 trace with PURE-XLA rms_norm ---")
    print(trace_fp8_pure.remote())
