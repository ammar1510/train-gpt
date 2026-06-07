"""End-to-end throughput: bf16 vs fp8 GEMM x Pallas vs pure-XLA rms_norm.

Motivation: on B200 the Pallas rms_norm kernel and jax.nn.scaled_matmul cannot
share an XLA program without miscompiling to NaN (see memory:
fp8-pallas-rmsnorm-nan). The only working fp8 path uses the pure-XLA rms_norm,
which gives up the custom kernel's speed. This bench quantifies that trade-off
on the FULL model by timing a 2x2 matrix:

                Pallas rms_norm     pure-XLA rms_norm
    bf16 GEMM   A (ships today)     B
    fp8  GEMM   C (NaN; time only)  D (the working fp8 path)

  C vs D : throughput hit of pure rms_norm under fp8 (the direct question)
  A vs D : working fp8 vs the bf16 we ship (the decision)
  A vs B : how much the Pallas rms_norm kernel buys on its own (sanity)

C is numerically broken (NaN) but its STEP TIME is valid — it is the ceiling fp8
could reach if the Pallas+scaled_matmul bug were fixed upstream.

Caveats:
  - The fp8 path runs the FORWARD GEMMs in fp8 and the BACKWARD GEMMs in bf16
    (straight-through custom_vjp in fp8.py). So fwd+bwd shows a smaller fp8 win
    than fwd-only by construction — that is the real training cost, reported
    honestly, not a measurement artifact.
  - B200 is power-capped at ~1000W (see memory: modal-b200-power-capped), so
    absolute MFU is power-bound. All four configs hit the SAME cap, so the
    relative comparisons here remain valid. Per-repeat spread is printed so a
    transient throttle is visible rather than hidden.

Run:
    modal run bench_fp8_norm.py
    modal run bench_fp8_norm.py --batch-size 16 --steps 30
"""
import modal

# B200 SXM dense tensor-core peaks (NVIDIA spec, sm100). MFU vs these.
B200_BF16_TFLOPS = 2_250.0
B200_FP8_TFLOPS = 4_500.0

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

app = modal.App("train-gpt-bench-fp8-norm")


@app.function(image=image, gpu="B200", memory=48 * 1024, timeout=30 * 60)
def run_bench(
    batch_size: int = 8,
    warmup: int = 5,
    steps: int = 20,
    repeats: int = 3,
) -> dict:
    import os
    import time

    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    # Match bench.py: keep XLA's own Triton GEMM autotuner out of the picture so
    # the bf16 path uses the same cuBLAS/NVJET kernels we benchmark elsewhere.
    os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

    import jax
    import jax.numpy as jnp
    from jax import lax

    from config import Config
    from model import init_params, param_count
    from kernels import rms_norm as rms_norm_pallas
    from kernels import _rms_norm_pure
    from fp8 import block_scaled_matmul

    cfg = Config()

    def build_loss(mm, norm):
        """Full-model cross-entropy loss parametrized by the matmul impl `mm`
        and the rms_norm impl `norm`. Mirrors model.py exactly (3 separate qkv
        projections, squared-ReLU MLP, tied embeddings, logit cap, scan over
        layers with rematerialization) but swaps `@` -> mm and rms_norm -> norm.
        The logits projection stays a plain bf16 matmul, as in model.py."""
        H, Dh = cfg.n_heads, cfg.head_dim

        def attention(x, p):
            B, T, _ = x.shape
            q = mm(x, p["wq"]).reshape(B, T, H, Dh)
            k = mm(x, p["wk"]).reshape(B, T, H, Dh)
            v = mm(x, p["wv"]).reshape(B, T, H, Dh)
            out = jax.nn.dot_product_attention(
                q, k, v, is_causal=True, implementation="cudnn"
            )
            return mm(out.reshape(B, T, H * Dh), p["wo"])

        def mlp(x, p):
            return mm(jax.nn.relu(mm(x, p["w_up"])) ** 2, p["w_down"])

        def block(x, p):
            x = x + attention(norm(x, p["norm1"], cfg.rms_eps), p)
            x = x + mlp(norm(x, p["norm2"], cfg.rms_eps), p)
            return x

        def forward(params, ids):
            B, T = ids.shape
            x = params["embed"][ids] + params["pos"][:T]

            @jax.checkpoint
            def step(carry, lp):
                return block(carry, lp), None

            x, _ = lax.scan(step, x, params["layers"])
            x = norm(x, params["final_norm"], cfg.rms_eps)
            w_out = params["embed"].T if cfg.tie_embeddings else params["unembed"]
            logits = x @ w_out
            logits = cfg.logit_cap * jnp.tanh(logits / cfg.logit_cap)
            return logits

        def loss(params, ids, tgt):
            logits = forward(params, ids).astype(jnp.float32)
            logp = jax.nn.log_softmax(logits, axis=-1)
            nll = -jnp.take_along_axis(logp, tgt[..., None], axis=-1).squeeze(-1)
            return nll.mean()

        return loss

    mm_bf16 = lambda a, b: a @ b
    mm_fp8 = lambda a, b: block_scaled_matmul(a, b)

    configs = [
        ("bf16 + pallas (A)", mm_bf16, rms_norm_pallas),
        ("bf16 + pure   (B)", mm_bf16, _rms_norm_pure),
        ("fp8  + pallas (C)", mm_fp8, rms_norm_pallas),   # NaN; time only
        ("fp8  + pure   (D)", mm_fp8, _rms_norm_pure),
    ]

    key = jax.random.PRNGKey(0)
    params = init_params(key, cfg)
    n_params = param_count(params)
    ids = jax.random.randint(
        jax.random.PRNGKey(1), (batch_size, cfg.seq_len), 0,
        cfg.vocab_size, dtype=jnp.int32,
    )
    tgt = jax.random.randint(
        jax.random.PRNGKey(2), (batch_size, cfg.seq_len), 0,
        cfg.vocab_size, dtype=jnp.int32,
    )
    tokens_per_step = batch_size * cfg.seq_len

    def timed(fn):
        """Min-of-repeats averaged step time, plus per-repeat spread. The first
        call triggers compilation; the rest ramp the clocks before timing."""
        out = fn()
        jax.block_until_ready(out)
        for _ in range(warmup - 1):
            out = fn()
        jax.block_until_ready(out)
        per = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for _ in range(steps):
                out = fn()
            jax.block_until_ready(out)
            per.append((time.perf_counter() - t0) / steps)
        return min(per), (max(per) - min(per)) / min(per)

    # FLOPs: forward 2*N*T, fwd+bwd 6*N*T over the matmul params (per bench.py).
    d, h, ff, L = cfg.d_model, cfg.n_heads * cfg.head_dim, cfg.d_ff, cfg.n_layers
    layer_params = 3 * d * h + h * d + 2 * d * ff + ff * d
    fwd_flops = 2.0 * L * layer_params * tokens_per_step
    bwd_flops = 6.0 * L * layer_params * tokens_per_step

    print(f"device: {jax.devices()[0]}")
    print(f"params: {n_params:,}  batch={batch_size}  seq={cfg.seq_len}  "
          f"tokens/step={tokens_per_step:,}")
    print(f"timing: warmup={warmup} steps={steps} repeats={repeats} (min reported)\n")

    results = {}
    for name, mm, norm in configs:
        loss = build_loss(mm, norm)
        fwd = jax.jit(loss)
        grad = jax.jit(jax.value_and_grad(loss))
        # Sanity: record finiteness of the forward loss (C is expected NaN).
        loss_val = float(fwd(params, ids, tgt))
        t_fwd, sp_fwd = timed(lambda: fwd(params, ids, tgt))
        t_bwd, sp_bwd = timed(lambda: grad(params, ids, tgt))
        results[name] = {
            "loss": loss_val,
            "t_fwd": t_fwd, "sp_fwd": sp_fwd,
            "t_bwd": t_bwd, "sp_bwd": sp_bwd,
        }
        print(f"{name}: loss={loss_val:.3f} finite={loss_val == loss_val}  "
              f"fwd={t_fwd*1e3:.1f}ms  fwd+bwd={t_bwd*1e3:.1f}ms")

    base = results["bf16 + pallas (A)"]

    def row(name, r):
        fwd_tps = tokens_per_step / r["t_fwd"]
        bwd_tps = tokens_per_step / r["t_bwd"]
        fwd_mfu = fwd_flops / r["t_fwd"] / 1e12 / B200_BF16_TFLOPS
        bwd_mfu = bwd_flops / r["t_bwd"] / 1e12 / B200_BF16_TFLOPS
        rel_fwd = base["t_fwd"] / r["t_fwd"]    # >1 = faster than baseline A
        rel_bwd = base["t_bwd"] / r["t_bwd"]
        return (f"  {name:<20}  "
                f"{r['t_fwd']*1e3:>7.1f}  {fwd_tps:>10,.0f}  {fwd_mfu*100:>5.1f}%  "
                f"{rel_fwd:>6.2f}x   |  "
                f"{r['t_bwd']*1e3:>7.1f}  {bwd_tps:>10,.0f}  {bwd_mfu*100:>5.1f}%  "
                f"{rel_bwd:>6.2f}x")

    print("\n" + "=" * 104)
    print(f"  {'config':<20}  {'fwd ms':>7}  {'fwd tok/s':>10}  {'MFU':>6}  "
          f"{'vs A':>7}  |  {'f+b ms':>7}  {'f+b tok/s':>10}  {'MFU':>6}  {'vs A':>7}")
    print("  " + "-" * 100)
    for name in results:
        print(row(name, results[name]))
    print("=" * 104)
    print("  'vs A' > 1.00x = faster than bf16+pallas baseline. MFU vs bf16 peak "
          f"({B200_BF16_TFLOPS:.0f} TF/s).")
    print("  C is NaN (timing only). Backward GEMMs are bf16 in both fp8 configs, "
          "so fwd+bwd fp8 gains < fwd-only by design.")

    # Headline numbers the decision hinges on.
    A, B, C, D = (results[k] for k in [
        "bf16 + pallas (A)", "bf16 + pure   (B)",
        "fp8  + pallas (C)", "fp8  + pure   (D)",
    ])
    print("\nHEADLINES (fwd+bwd, the training regime):")
    print(f"  pure-rms_norm hit under fp8 (C->D):   "
          f"{(C['t_bwd']/D['t_bwd'] - 1)*100:+.1f}%  "
          f"({'D slower' if D['t_bwd'] > C['t_bwd'] else 'D faster'})")
    print(f"  working fp8 vs shipped bf16 (A->D):   "
          f"{(A['t_bwd']/D['t_bwd'] - 1)*100:+.1f}%  "
          f"({'fp8 faster' if D['t_bwd'] < A['t_bwd'] else 'fp8 SLOWER'})")
    print(f"  Pallas kernel value in bf16 (A->B):   "
          f"{(B['t_bwd']/A['t_bwd'] - 1)*100:+.1f}%  "
          f"(B is the no-kernel bf16 cost)")

    return {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()}


@app.local_entrypoint()
def main(batch_size: int = 8, warmup: int = 5, steps: int = 20, repeats: int = 3):
    run_bench.remote(
        batch_size=batch_size, warmup=warmup, steps=steps, repeats=repeats
    )
