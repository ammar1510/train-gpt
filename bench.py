"""Throughput benchmark: forward-only and fwd+bwd+step at target model dims.

Runs on a Modal B200. Reports step time (ms), tokens/sec, and MFU.

Usage:
    modal run bench.py
    modal run bench.py --config small --batch-size 4 --steps 50
    modal run bench.py --fwd-only
    modal run bench.py --batch-size 16 --profile
    modal run bench.py --batch-size 128 --repeats 5   # min-of-5 + throttle check
    modal volume get train-gpt-traces perfetto_trace.json.gz ./perfetto_trace.json.gz

Timing reports the *best* (min) of `repeats` averaged windows plus the
run-to-run spread, and samples nvidia-smi clocks/temp/power/throttle flags
during timing so a throttled GPU is visible rather than silently skewing MFU.
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
    .add_local_python_source("config", "model", "losses", "train", "data", "kernels")
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
    batch_size: int = 8,
    warmup: int = 5,
    steps: int = 20,
    repeats: int = 3,
    fwd_only: bool = False,
    peak_tflops: float = B200_BF16_TFLOPS,
    profile: bool = False,
    use_remat: bool = True,
    use_pallas_norm: bool = True,
) -> dict:
    import os
    import time

    hlo_dump_dir = "/tmp/hlo-dump"
    # Set before JAX initializes so XLA picks it up.
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    xla_flags = ["--xla_gpu_enable_triton_gemm=false"]
    if profile:
        os.makedirs(hlo_dump_dir, exist_ok=True)
        xla_flags += [
            f"--xla_dump_to={hlo_dump_dir}",
            "--xla_dump_hlo_as_text",
            "--xla_dump_hlo_pass_re=.*",
        ]
    os.environ["XLA_FLAGS"] = " ".join(xla_flags)

    import jax
    import jax.numpy as jnp
    import optax

    from config import SMALL, Config
    from losses import cross_entropy_loss
    from model import init_params, param_count
    from train import make_train_step

    import dataclasses

    FULL_CONFIG = Config()
    cfg = SMALL if config == "small" else FULL_CONFIG
    # Toggle activation checkpointing + rms_norm impl so their cost is measurable.
    cfg = dataclasses.replace(
        cfg, use_remat=use_remat, use_pallas_norm=use_pallas_norm
    )

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

    def timed_run(fn, warmup, steps, repeats):
        # Warm up once (compilation already done by caller); also ramps the
        # GPU clocks before the first measured window.
        out = fn()
        for _ in range(warmup - 1):
            out = fn()
        jax.block_until_ready(out)
        # Each repeat is an independent averaged window. Returning the full list
        # lets the caller report min/median/spread instead of a single number
        # that a transient throttle can corrupt.
        per_step = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for _ in range(steps):
                out = fn()
            jax.block_until_ready(out)
            per_step.append((time.perf_counter() - t0) / steps)
        return per_step

    key = jax.random.PRNGKey(0)
    params = init_params(key, cfg)
    n_params = param_count(params)
    dummy = jnp.zeros((batch_size, cfg.seq_len), dtype=jnp.int32)

    import statistics
    import subprocess
    import tempfile
    from pathlib import Path as _Path

    class _GpuSampler:
        """Polls nvidia-smi in the background to detect clock throttling.

        Runs as a separate `nvidia-smi -lms` subprocess that queries NVML — not
        the CUDA compute path — so it does not perturb the timed JAX loop. If
        nvidia-smi is unavailable the sampler degrades to a no-op.
        """

        # Order matters: parsing indexes into this list.
        _FIELDS = [
            "clocks.sm",                                       # 0 current SM clock
            "clocks.max.sm",                                   # 1 max supported
            "temperature.gpu",                                 # 2
            "power.draw",                                      # 3
            "power.limit",                                     # 4 enforced cap
            "clocks_throttle_reasons.sw_power_cap",            # 5
            "clocks_throttle_reasons.sw_thermal_slowdown",     # 6
            "clocks_throttle_reasons.hw_thermal_slowdown",     # 7
            "clocks_throttle_reasons.hw_power_brake_slowdown",  # 8
        ]

        def __init__(self, interval_ms: int = 200):
            self._interval_ms = interval_ms
            self._proc = None
            self._tmp = None

        def start(self) -> None:
            try:
                self._tmp = tempfile.NamedTemporaryFile(
                    mode="w+", suffix=".csv", delete=False
                )
                self._proc = subprocess.Popen(
                    [
                        "nvidia-smi",
                        "--query-gpu=" + ",".join(self._FIELDS),
                        "--format=csv,noheader,nounits",
                        "-lms", str(self._interval_ms),
                    ],
                    stdout=self._tmp,
                    stderr=subprocess.DEVNULL,
                )
            except (FileNotFoundError, OSError):
                # No nvidia-smi (e.g. CPU host) — sampling silently disabled.
                self._proc = None

        def stop(self) -> dict:
            if self._proc is None or self._tmp is None:
                return {}
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()

            self._tmp.flush()
            self._tmp.seek(0)
            rows = []
            for line in self._tmp:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == len(self._FIELDS):
                    rows.append(parts)
            self._tmp.close()
            try:
                os.unlink(self._tmp.name)
            except OSError:
                pass
            if not rows:
                return {}

            def col_floats(idx):
                out = []
                for r in rows:
                    try:
                        out.append(float(r[idx]))
                    except ValueError:
                        pass
                return out

            sm = col_floats(0)
            sm_cap = col_floats(1)
            temp = col_floats(2)
            draw = col_floats(3)
            lim = col_floats(4)

            # Fraction of samples in which each throttle reason was Active.
            reasons = {}
            for i, name in enumerate(self._FIELDS):
                if not name.startswith("clocks_throttle_reasons."):
                    continue
                active = sum(1 for r in rows if r[i].lower().startswith("active"))
                frac = active / len(rows)
                if frac > 0:
                    reasons[name.split(".")[-1]] = round(frac, 3)

            return {
                "samples": len(rows),
                "sm_clock_mhz": (
                    {"min": min(sm), "max": max(sm),
                     "mean": round(statistics.fmean(sm), 1)}
                    if sm else {}
                ),
                "sm_clock_cap_mhz": max(sm_cap) if sm_cap else None,
                "temp_c_max": max(temp) if temp else None,
                "power_w": (
                    {"max_draw": max(draw), "limit": max(lim)}
                    if draw and lim else {}
                ),
                "throttle_active_frac": reasons,
            }

    def measure(fn, flops):
        """Run min-of-N timing with concurrent GPU telemetry."""
        sampler = _GpuSampler()
        sampler.start()
        times = timed_run(fn, warmup, steps, repeats)
        telemetry = sampler.stop()
        t_min = min(times)
        return {
            "ms_min": t_min * 1e3,
            "ms_median": statistics.median(times) * 1e3,
            "ms_max": max(times) * 1e3,
            "spread_pct": (max(times) - t_min) / t_min * 100.0 if t_min else 0.0,
            # Headline tok/s and MFU use the best (min) window: the run least
            # corrupted by throttling, i.e. closest to the kernels' true speed.
            "tok_per_sec": batch_size * cfg.seq_len / t_min,
            "mfu": mfu(flops, t_min),
            "n_repeats": len(times),
            "telemetry": telemetry,
        }

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
        "use_remat": use_remat,
        "use_pallas_norm": use_pallas_norm,
        "device": str(jax.devices()[0]),
        "device_kind": jax.devices()[0].device_kind,
        "backend": jax.default_backend(),
        "cuda_version": cuda_ver,
    }

    fwd_jit = jax.jit(lambda p, x: cross_entropy_loss(p, x, x, cfg))
    print("compiling fwd ...")
    fwd_jit(params, dummy)

    results["fwd"] = measure(
        lambda: fwd_jit(params, dummy),
        estimate_flops(cfg, batch_size, fwd_only=True),
    )

    if not fwd_only:
        optimizer = optax.adamw(learning_rate=3e-4)
        opt_state = optimizer.init(params)
        train_step = make_train_step(cfg, optimizer)
        print("compiling fwd+bwd ...")
        params, opt_state, _, _ = train_step(params, opt_state, dummy, dummy)

        results["fwd_bwd"] = measure(
            lambda: train_step(params, opt_state, dummy, dummy),
            estimate_flops(cfg, batch_size, fwd_only=False),
        )

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
        # Sample clocks/power/throttle during the capture too, so every trace
        # self-reports the GPU state it was taken under — a throttled capture
        # otherwise looks like a code regression (kernels uniformly slower).
        prof_sampler = _GpuSampler()
        prof_sampler.start()
        with jax.profiler.trace(tmp_trace_dir, create_perfetto_trace=True):
            for _ in range(5):
                out = profile_fn()
            jax.block_until_ready(out)
        results["profile_telemetry"] = prof_sampler.stop()

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

        # Bundle the HLO dumps into a tarball next to the trace.
        hlo_src = _Path(hlo_dump_dir)
        if hlo_src.exists() and any(hlo_src.iterdir()):
            tar_path = _Path(REMOTE_TRACE_DIR) / "hlo_dump.tar.gz"
            shutil.make_archive(str(tar_path).removesuffix(".tar.gz"), "gztar", hlo_src)
            trace_vol.commit()
            results["hlo_dump_name"] = tar_path.name
            print(f"hlo dump saved to volume: {tar_path.name}  ({tar_path.stat().st_size / 1024:.1f} KB)")
        else:
            print("warning: no hlo dump files found")

    return results


@app.local_entrypoint()
def main(
    config: str = "full",
    batch_size: int = 8,
    warmup: int = 5,
    steps: int = 20,
    repeats: int = 3,
    fwd_only: bool = False,
    peak_tflops: float = B200_BF16_TFLOPS,
    profile: bool = False,
    use_remat: bool = True,
    use_pallas_norm: bool = True,
):
    r = run_bench.remote(
        config=config,
        batch_size=batch_size,
        warmup=warmup,
        steps=steps,
        repeats=repeats,
        fwd_only=fwd_only,
        peak_tflops=peak_tflops,
        profile=profile,
        use_remat=use_remat,
        use_pallas_norm=use_pallas_norm,
    )

    print(f"\n{'─' * 55}")
    print(f"  config     : {r['config']}  ({r['n_params']:,} params)")
    print(f"  batch      : {r['batch_size']}  seq_len {r['seq_len']}")
    print(f"  remat      : {r.get('use_remat', True)}")
    print(f"  pallas norm: {r.get('use_pallas_norm', True)}")
    print(f"  device     : {r['device']}")
    print(f"  peak ref   : {peak_tflops:,.0f} TFLOP/s  (BF16 dense)")
    print(f"  jax backend: {r['backend']}")
    print(f"  device kind: {r['device_kind']}")
    print(f"  cuda       : {r['cuda_version']}")
    print(f"  gemm backend: cuBLAS (triton disabled)")
    print(f"{'─' * 55}")

    def telem_lines(t):
        if not t:
            print(f"  {'':<12}  (no GPU telemetry — nvidia-smi unavailable)")
            return
        sm = t.get("sm_clock_mhz", {})
        pw = t.get("power_w", {})
        cap = t.get("sm_clock_cap_mhz")
        clk = (f"SM {sm['min']:.0f}–{sm['max']:.0f} MHz (cap {cap:.0f})"
               if sm and cap else "SM clock n/a")
        therm = (f"{t.get('temp_c_max', '?')}°C, "
                 f"{pw.get('max_draw', '?')}/{pw.get('limit', '?')} W"
                 if pw else f"{t.get('temp_c_max', '?')}°C")
        print(f"  {'':<12}  {clk}   {therm}   ({t.get('samples', 0)} samples)")
        thr = t.get("throttle_active_frac", {})
        if thr:
            parts = ", ".join(f"{k} {v * 100:.0f}% of samples"
                              for k, v in thr.items())
            print(f"  {'':<12}  ⚠ THROTTLED: {parts}")
        else:
            print(f"  {'':<12}  ✓ no throttle flags raised")

    def row(label, d):
        print(f"  {label:<12}  {d['ms_min']:7.1f} ms/step (best)  "
              f"{d['tok_per_sec']:>12,.0f} tok/s  MFU {d['mfu'] * 100:.1f}%")
        print(f"  {'':<12}  median {d['ms_median']:.1f}  max {d['ms_max']:.1f} ms  "
              f"·  run-to-run spread {d['spread_pct']:.1f}%  (n={d['n_repeats']})")
        telem_lines(d.get("telemetry", {}))

    row("fwd-only", r["fwd"])
    if "fwd_bwd" in r:
        row("fwd+bwd", r["fwd_bwd"])

    print(f"{'─' * 55}")
    print(f"  HBM peak   : {r['hbm_peak_gb']:.1f} / {r['hbm_limit_gb']:.1f} GB")
    if "profile_telemetry" in r:
        print(f"  trace capture GPU state:")
        telem_lines(r["profile_telemetry"])
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

    if "hlo_dump_name" in r:
        fname = r["hlo_dump_name"]
        local_path = Path(fname)
        print(f"downloading hlo dump from volume ...")
        with open(local_path, "wb") as f:
            for chunk in trace_vol.read_file(fname):
                f.write(chunk)
        print(f"HLO dump saved to: {local_path.resolve()}")
        print(f"Extract with: tar -xzf {local_path.name} -C hlo-dump/")

