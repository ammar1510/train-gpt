"""Throughput benchmark: forward-only and fwd+bwd+step at target model dims.

Runs on a Modal B200. Reports step time (ms), tokens/sec, and MFU.

Usage:
    modal run bench.py
    modal run bench.py --config small --batch-size 4 --steps 50
    modal run bench.py --fwd-only
    modal run bench.py --batch-size 16 --profile
    modal volume get train-gpt-traces perfetto_trace.json.gz ./perfetto_trace.json.gz
"""
from pathlib import Path

import modal

# B200 SXM BF16 dense tensor-core peak (TFLOP/s).
# Verify against NVIDIA product page before trusting MFU numbers.
B200_BF16_TFLOPS = 2_250.0

TRACE_VOLUME_NAME = "train-gpt-traces"
REMOTE_TRACE_DIR = "/traces"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "jax[cuda13]==0.9.2",
        "optax==0.2.8",
        "numpy==2.4.5",
        "chex==0.1.91",
    )
    .add_local_python_source("config", "model", "losses", "train", "data")
)

app = modal.App("train-gpt-bench")
trace_vol = modal.Volume.from_name(TRACE_VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    gpu="B200",
    memory=48 * 1024,
    timeout=30 * 60,
    volumes={REMOTE_TRACE_DIR: trace_vol},
)
def run_bench(
    config: str = "full",
    batch_size: int = 4,
    warmup: int = 5,
    steps: int = 20,
    fwd_only: bool = False,
    peak_tflops: float = B200_BF16_TFLOPS,
    profile: bool = False,
) -> dict:
    import time

    import jax
    import jax.numpy as jnp
    import optax

    from config import SMALL, Config
    from losses import cross_entropy_loss
    from model import init_params, param_count
    from train import make_train_step

    FULL_CONFIG = Config()
    cfg = SMALL if config == "small" else FULL_CONFIG

    def estimate_flops(cfg: Config, batch_size: int, fwd_only: bool) -> float:
        d, h, ff, L = cfg.d_model, cfg.n_heads * cfg.head_dim, cfg.d_ff, cfg.n_layers
        layer_params = 3 * d * h + h * d + 2 * d * ff + ff * d
        tokens = batch_size * cfg.seq_len
        # fwd: 2N*T, fwd+bwd: 6N*T
        multiplier = 2.0 if fwd_only else 6.0
        return multiplier * L * layer_params * tokens

    def mfu(flops: float, step_time_s: float) -> float:
        achieved_tflops = flops / step_time_s / 1e12
        return achieved_tflops / peak_tflops

    def memory_gb() -> tuple[float, float]:
        try:
            stats = jax.devices()[0].memory_stats()
            return (
                stats["peak_bytes_in_use"] / 2**30,
                stats["bytes_limit"] / 2**30,
            )
        except Exception:
            return (0.0, 0.0)

    def timed_run(fn, warmup, steps):
        out = fn()
        for _ in range(warmup - 1):
            out = fn()
        jax.block_until_ready(out)
        t0 = time.perf_counter()
        for _ in range(steps):
            out = fn()
        jax.block_until_ready(out)
        return (time.perf_counter() - t0) / steps

    key = jax.random.PRNGKey(0)
    params = init_params(key, cfg)
    n_params = param_count(params)
    dummy = jnp.zeros((batch_size, cfg.seq_len), dtype=jnp.int32)

    import subprocess
    from pathlib import Path as _Path

    def _cuda_version() -> str:
        # Prefer the version file shipped with the CUDA runtime.
        for p in ("/usr/local/cuda/version.json", "/usr/local/cuda/version.txt"):
            try:
                txt = _Path(p).read_text()
                if p.endswith(".json"):
                    import json as _json
                    return _json.loads(txt).get("cuda", {}).get("version", txt.strip())
                return txt.strip().split("\n")[0]
            except FileNotFoundError:
                pass
        # Fall back to nvidia-smi.
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        return f"driver {r.stdout.strip()}" if r.returncode == 0 else "unknown"

    cuda_ver = _cuda_version()

    results: dict = {
        "config": config,
        "n_params": n_params,
        "batch_size": batch_size,
        "seq_len": cfg.seq_len,
        "device": str(jax.devices()[0]),
        "device_kind": jax.devices()[0].device_kind,
        "backend": jax.default_backend(),
        "cuda_version": cuda_ver,
    }

    fwd_jit = jax.jit(lambda p, x: cross_entropy_loss(p, x, x, cfg))
    print("compiling fwd ...")
    fwd_jit(params, dummy)

    dt_fwd = timed_run(lambda: fwd_jit(params, dummy), warmup, steps)
    flops_fwd = estimate_flops(cfg, batch_size, fwd_only=True)
    results["fwd"] = {
        "ms_per_step": dt_fwd * 1e3,
        "tok_per_sec": batch_size * cfg.seq_len / dt_fwd,
        "mfu": mfu(flops_fwd, dt_fwd),
    }

    if not fwd_only:
        optimizer = optax.adamw(learning_rate=3e-4)
        opt_state = optimizer.init(params)
        train_step = make_train_step(cfg, optimizer)
        print("compiling fwd+bwd ...")
        params, opt_state, _ = train_step(params, opt_state, dummy, dummy)

        dt_step = timed_run(
            lambda: train_step(params, opt_state, dummy, dummy),
            warmup, steps,
        )
        flops_step = estimate_flops(cfg, batch_size, fwd_only=False)
        results["fwd_bwd"] = {
            "ms_per_step": dt_step * 1e3,
            "tok_per_sec": batch_size * cfg.seq_len / dt_step,
            "mfu": mfu(flops_step, dt_step),
        }

    used_gb, limit_gb = memory_gb()
    results["hbm_peak_gb"] = used_gb
    results["hbm_limit_gb"] = limit_gb

    if profile:
        from pathlib import Path as _Path

        tmp_trace_dir = "/tmp/jax-trace"
        profile_fn = (
            (lambda: train_step(params, opt_state, dummy, dummy))
            if not fwd_only
            else (lambda: fwd_jit(params, dummy))
        )
        print("capturing perfetto trace (5 steps) ...")
        with jax.profiler.trace(tmp_trace_dir, create_perfetto_trace=True):
            for _ in range(5):
                out = profile_fn()
            jax.block_until_ready(out)

        # Copy trace file to the persistent volume so main() can download it.
        import shutil
        src = _Path(tmp_trace_dir) / "perfetto_trace.json.gz"
        if not src.exists():
            candidates = (
                list(_Path(tmp_trace_dir).rglob("*.gz"))
                + list(_Path(tmp_trace_dir).rglob("*.json"))
            )
            src = candidates[0] if candidates else None

        if src:
            dst = _Path(REMOTE_TRACE_DIR) / src.name
            shutil.copy2(src, dst)
            trace_vol.commit()
            results["trace_remote_name"] = src.name
            print(f"trace saved to volume: {src.name}  ({src.stat().st_size / 1024:.1f} KB)")
        else:
            print("warning: no trace file found")

    return results


@app.local_entrypoint()
def main(
    config: str = "full",
    batch_size: int = 4,
    warmup: int = 5,
    steps: int = 20,
    fwd_only: bool = False,
    peak_tflops: float = B200_BF16_TFLOPS,
    profile: bool = False,
):
    r = run_bench.remote(
        config=config,
        batch_size=batch_size,
        warmup=warmup,
        steps=steps,
        fwd_only=fwd_only,
        peak_tflops=peak_tflops,
        profile=profile,
    )

    print(f"\n{'─' * 55}")
    print(f"  config     : {r['config']}  ({r['n_params']:,} params)")
    print(f"  batch      : {r['batch_size']}  seq_len {r['seq_len']}")
    print(f"  device     : {r['device']}")
    print(f"  peak ref   : {peak_tflops:,.0f} TFLOP/s  (BF16 dense)")
    print(f"  jax backend: {r['backend']}")
    print(f"  device kind: {r['device_kind']}")
    print(f"  cuda       : {r['cuda_version']}")
    print(f"{'─' * 55}")

    def row(label, d):
        print(f"  {label:<12}  {d['ms_per_step']:7.1f} ms/step  "
              f"{d['tok_per_sec']:>12,.0f} tok/s  MFU {d['mfu'] * 100:.1f}%")

    row("fwd-only", r["fwd"])
    if "fwd_bwd" in r:
        row("fwd+bwd", r["fwd_bwd"])

    print(f"{'─' * 55}")
    print(f"  HBM peak   : {r['hbm_peak_gb']:.1f} / {r['hbm_limit_gb']:.1f} GB")
    print()

    if "trace_remote_name" in r:
        fname = r["trace_remote_name"]
        local_path = Path(fname)
        print(f"downloading trace from volume ...")
        with open(local_path, "wb") as f:
            for chunk in trace_vol.read_file(fname):
                f.write(chunk)
        print(f"Perfetto trace saved to: {local_path.resolve()}")
        print("Open at: https://ui.perfetto.dev  (drag-and-drop the file)")
