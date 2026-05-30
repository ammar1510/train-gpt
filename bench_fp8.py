"""FP8 vs BF16 GEMM microbenchmark + precision comparison on B200.

Times the four matmul shapes used in the model at batch=30, seq=2048
(M = 61,440 tokens) and reports numerical error of each precision against an
fp32 reference. Does NOT run the model — pure cuBLAS/NVJET GEMM kernel time
and per-shape numerical fidelity. The speedup here is the UPPER BOUND of what
FP8 could buy end-to-end; real FP8 training adds quantize / scale-EMA /
dequantize overhead this does not capture.

Accuracy method: fp32 matmul on fp32-cast inputs is the gold reference. Both
bf16 and fp8 results are cast to fp32 and compared via Frobenius relative
error and max absolute error. Inputs are random-normal scaled by 0.1; for
real-distribution numbers you would need to load actual model params and
activations instead.

Run:
    modal run bench_fp8.py
    modal run bench_fp8.py --dump-hlo    # verify FP8 kernels actually fired

Verify FP8 dispatch in the dumped HLO:
    grep -l "f8e4m3fn" hlo-dump/*.txt    # dot ops should reference f8e4m3fn
If a shape shows ~1.0x speedup, XLA is silently falling back to BF16 — check
HLO to confirm and consider jax.nn.scaled_matmul as an alternative path.
"""
from pathlib import Path

import modal

B200_BF16_TFLOPS = 2_250.0  # NVIDIA spec, dense, sm100
B200_FP8_TFLOPS = 4_500.0   # NVIDIA spec, dense, sm100

TRACE_VOLUME_NAME = "train-gpt-traces"
REMOTE_TRACE_DIR = "/traces"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("jax[cuda13]==0.9.2", "numpy==2.4.5")
)

app = modal.App("train-gpt-bench-fp8")
trace_vol = modal.Volume.from_name(TRACE_VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    gpu="B200",
    timeout=10 * 60,
    volumes={REMOTE_TRACE_DIR: trace_vol},
)
def run_microbench(dump_hlo: bool = False) -> dict:
    import os
    import time

    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    xla_flags = ["--xla_gpu_enable_triton_gemm=false"]
    if dump_hlo:
        os.makedirs("/tmp/hlo-dump-fp8", exist_ok=True)
        xla_flags += [
            "--xla_dump_to=/tmp/hlo-dump-fp8",
            "--xla_dump_hlo_as_text",
        ]
    os.environ["XLA_FLAGS"] = " ".join(xla_flags)

    import jax
    import jax.numpy as jnp

    # Shapes the model actually emits. M = batch * seq_len = 30 * 2048.
    M = 30 * 2048
    SHAPES = [
        ("attn proj (q/k/v/o)", M, 2304, 2304),
        ("mlp up",              M, 2304, 9216),
        ("mlp down",            M, 9216, 2304),
        ("logits",              M, 2304, 50257),
    ]

    # e4m3 has range ±448; map amax to ±448 by multiplying by (448 / amax).
    E4M3_MAX = 448.0
    BLOCK_K = 128  # DeepSeek-V3-style per-K-tile block size

    @jax.jit
    def matmul(a, b):
        return a @ b

    @jax.jit
    def matmul_f8(a, b):
        # Explicit accumulator dtype — fp8 inputs have no implicit promotion.
        return jnp.matmul(a, b, preferred_element_type=jnp.bfloat16)

    @jax.jit
    def matmul_f32(a, b):
        return a @ b

    @jax.jit
    def matmul_f8_pt_scaled(a_bf, b_bf):
        # Per-tensor amax scaling — Transformer Engine's default recipe (without
        # the delayed-history part; here we use the current-tensor amax).
        a_f32 = a_bf.astype(jnp.float32)
        b_f32 = b_bf.astype(jnp.float32)
        a_scale = E4M3_MAX / jnp.maximum(jnp.max(jnp.abs(a_f32)), 1e-12)
        b_scale = E4M3_MAX / jnp.maximum(jnp.max(jnp.abs(b_f32)), 1e-12)
        a_f8 = (a_f32 * a_scale).astype(jnp.float8_e4m3fn)
        b_f8 = (b_f32 * b_scale).astype(jnp.float8_e4m3fn)
        out = jnp.matmul(a_f8, b_f8, preferred_element_type=jnp.float32)
        return (out / (a_scale * b_scale)).astype(jnp.bfloat16)

    @jax.jit
    def matmul_f8_block_scaled(a_bf, b_bf):
        # Per-K-tile block scaling: (1, BLOCK_K) blocks for a, (BLOCK_K, 1)
        # for b — same scaling granularity along the contracted axis as
        # DeepSeek-V3's recipe (their full recipe also tiles the non-K axis of
        # weights; this version is finer along that axis, so it's an
        # upper-bound on quality vs the published recipe).
        # NOTE: this simulates the precision via quantize→dequantize→bf16
        # matmul. The matmul runs in bf16, not fp8, so the timing of this path
        # is not meaningful — only the accuracy is.
        M, K = a_bf.shape
        K_b, N = b_bf.shape
        n_tiles = K // BLOCK_K

        a_f32 = a_bf.astype(jnp.float32)
        a_tiled = a_f32.reshape(M, n_tiles, BLOCK_K)
        a_amax = jnp.max(jnp.abs(a_tiled), axis=2, keepdims=True)
        a_scale = E4M3_MAX / jnp.maximum(a_amax, 1e-12)
        # Optimization barrier forces XLA to actually materialize the fp8
        # rounding — without it, XLA folds `(x * s).astype(fp8).astype(fp32) / s`
        # back to `x` since no downstream op requires fp8 values, and the
        # quantization noise (which is the whole point) silently disappears.
        a_q = jax.lax.optimization_barrier(
            (a_tiled * a_scale).astype(jnp.float8_e4m3fn)
        )
        a_dq = (a_q.astype(jnp.float32) / a_scale).reshape(M, K).astype(jnp.bfloat16)

        b_f32 = b_bf.astype(jnp.float32)
        b_tiled = b_f32.reshape(n_tiles, BLOCK_K, N)
        b_amax = jnp.max(jnp.abs(b_tiled), axis=1, keepdims=True)
        b_scale = E4M3_MAX / jnp.maximum(b_amax, 1e-12)
        b_q = jax.lax.optimization_barrier(
            (b_tiled * b_scale).astype(jnp.float8_e4m3fn)
        )
        b_dq = (b_q.astype(jnp.float32) / b_scale).reshape(K_b, N).astype(jnp.bfloat16)

        return a_dq @ b_dq

    @jax.jit
    def errors(out, ref):
        # ref is fp32; cast out to fp32 first so the diff is meaningful.
        diff = out.astype(jnp.float32) - ref
        ref_norm = jnp.linalg.norm(ref)
        return (
            jnp.linalg.norm(diff) / ref_norm,   # Frobenius relative error
            jnp.max(jnp.abs(diff)),             # max absolute error
        )

    def time_call(fn, a, b, n_iter: int = 50, warmup: int = 5) -> float:
        fn(a, b).block_until_ready()
        for _ in range(warmup):
            fn(a, b).block_until_ready()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            out = fn(a, b)
        out.block_until_ready()
        return (time.perf_counter() - t0) / n_iter

    print(f"  device: {jax.devices()[0]}")
    print(f"  bf16 peak ref: {B200_BF16_TFLOPS:>5.0f} TFLOP/s   "
          f"fp8 peak ref: {B200_FP8_TFLOPS:>5.0f} TFLOP/s")

    results = []
    key = jax.random.PRNGKey(0)
    for name, m, k, n in SHAPES:
        ka, kb, key = jax.random.split(key, 3)
        # Scale down so e4m3 (range ±448) doesn't saturate after cast.
        # Does not affect kernel time, only numerical fidelity.
        a_bf = (jax.random.normal(ka, (m, k)) * 0.1).astype(jnp.bfloat16)
        b_bf = (jax.random.normal(kb, (k, n)) * 0.1).astype(jnp.bfloat16)
        a_f8 = a_bf.astype(jnp.float8_e4m3fn)
        b_f8 = b_bf.astype(jnp.float8_e4m3fn)

        flops = 2.0 * m * k * n

        t_bf = time_call(matmul, a_bf, b_bf)
        t_f8 = time_call(matmul_f8, a_f8, b_f8)

        # Accuracy: fp32 matmul on fp32-cast inputs is gold; compare all
        # lower-precision variants against it.
        ref = matmul_f32(a_bf.astype(jnp.float32), b_bf.astype(jnp.float32))
        out_bf = matmul(a_bf, b_bf)
        out_f8 = matmul_f8(a_f8, b_f8)
        out_pt = matmul_f8_pt_scaled(a_bf, b_bf)
        out_block = matmul_f8_block_scaled(a_bf, b_bf)
        rel_bf, max_bf = (float(x) for x in errors(out_bf, ref))
        rel_f8, max_f8 = (float(x) for x in errors(out_f8, ref))
        rel_pt, max_pt = (float(x) for x in errors(out_pt, ref))
        rel_block, max_block = (float(x) for x in errors(out_block, ref))

        results.append({
            "name": name, "m": m, "k": k, "n": n,
            "t_bf": t_bf, "t_f8": t_f8, "flops": flops,
            "rel_bf": rel_bf, "max_bf": max_bf,
            "rel_f8": rel_f8, "max_f8": max_f8,
            "rel_pt": rel_pt, "max_pt": max_pt,
            "rel_block": rel_block, "max_block": max_block,
        })

    print()
    print("THROUGHPUT")
    print("-" * 108)
    print(f"  {'shape':<22}  {'M':>6}  {'K':>5}  {'N':>5}  "
          f"{'bf16 ms':>8}  {'bf16 TF/s':>10}  "
          f"{'fp8 ms':>8}  {'fp8 TF/s':>10}  {'speedup':>8}")
    print("-" * 108)
    for r in results:
        print(f"  {r['name']:<22}  {r['m']:>6}  {r['k']:>5}  {r['n']:>5}  "
              f"{r['t_bf'] * 1e3:>8.2f}  {r['flops'] / r['t_bf'] / 1e12:>10.0f}  "
              f"{r['t_f8'] * 1e3:>8.2f}  {r['flops'] / r['t_f8'] / 1e12:>10.0f}  "
              f"{r['t_bf'] / r['t_f8']:>7.2f}x")

    print()
    print("ACCURACY — naked cast (vs fp32 reference; rel_err = ||diff|| / ||ref||)")
    print("-" * 108)
    print(f"  {'shape':<22}  "
          f"{'bf16 rel_err':>13}  {'bf16 max_abs':>13}  "
          f"{'fp8  rel_err':>13}  {'fp8  max_abs':>13}  {'fp8 / bf16':>11}")
    print("-" * 108)
    for r in results:
        ratio = r["rel_f8"] / r["rel_bf"] if r["rel_bf"] > 0 else float("inf")
        print(f"  {r['name']:<22}  "
              f"{r['rel_bf']:>13.2e}  {r['max_bf']:>13.2e}  "
              f"{r['rel_f8']:>13.2e}  {r['max_f8']:>13.2e}  "
              f"{ratio:>10.1f}x")

    print()
    print("ACCURACY — scaled FP8 (vs fp32 reference; ratios compared to bf16 baseline)")
    print("-" * 108)
    print(f"  {'shape':<22}  "
          f"{'pt-scaled rel':>13}  {'pt / bf16':>11}  "
          f"{'block-scaled':>13}  {'block / bf16':>13}")
    print("-" * 108)
    for r in results:
        ratio_pt = r["rel_pt"] / r["rel_bf"] if r["rel_bf"] > 0 else float("inf")
        ratio_bl = r["rel_block"] / r["rel_bf"] if r["rel_bf"] > 0 else float("inf")
        print(f"  {r['name']:<22}  "
              f"{r['rel_pt']:>13.2e}  {ratio_pt:>10.1f}x  "
              f"{r['rel_block']:>13.2e}  {ratio_bl:>12.1f}x")
    print("-" * 108)
    print("  pt-scaled    : real FP8 matmul, scale = E4M3_MAX / amax(tensor).")
    print("  block-scaled : per-K-tile (1, 128) / (128, 1) scaling, simulated via")
    print("                 quantize→dequantize→bf16 matmul (precision only, not speed).")
    print("  Inputs are random-normal * 0.1; real model tensors have different")
    print("  (per-layer) distributions — replace inputs with saved params and")
    print("  activations from a real forward pass for a true-distribution test.")
    result: dict = {}
    if dump_hlo:
        import shutil
        hlo_src = Path("/tmp/hlo-dump-fp8")
        if hlo_src.exists() and any(hlo_src.iterdir()):
            tar_path = Path(REMOTE_TRACE_DIR) / "hlo_dump_fp8.tar.gz"
            shutil.make_archive(
                str(tar_path).removesuffix(".tar.gz"), "gztar", hlo_src,
            )
            trace_vol.commit()
            size_kb = tar_path.stat().st_size / 1024
            print(f"  hlo dump saved to volume: {tar_path.name}  ({size_kb:.1f} KB)")
            result["hlo_dump_name"] = tar_path.name
        else:
            print("  warning: no hlo dump files found at /tmp/hlo-dump-fp8")
    return result


@app.local_entrypoint()
def main(dump_hlo: bool = False) -> None:
    r = run_microbench.remote(dump_hlo=dump_hlo) or {}

    if "hlo_dump_name" in r:
        fname = r["hlo_dump_name"]
        local_path = Path(fname)
        print(f"downloading hlo dump from volume ...")
        with open(local_path, "wb") as f:
            for chunk in trace_vol.read_file(fname):
                f.write(chunk)
        print(f"HLO dump saved to: {local_path.resolve()}")
        print(f"Extract with: tar -xzf {local_path.name} -C hlo-dump-fp8/")
