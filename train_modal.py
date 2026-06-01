"""Modal B200 launcher for the bf16 training run on FineWeb-Edu.

Runs the pure-bf16 loop (model.py / train.py — no fp8 anywhere) against the
real corpus on the `fineweb-edu-data` volume, and reports the same throughput
metrics as bench.py: tok/s, MFU, and peak HBM. Checkpoints are written to the
`train-gpt-checkpoints` volume (final + every --ckpt-every steps).

The MFU/FLOP helpers are copied from bench.py verbatim so the numbers are
directly comparable to a `modal run bench.py` reading. Like bench.py, the FLOP
estimate is the parameter-based 6*N*T approximation: it omits attention
score/value einsums and the embedding+logit matmul, so reported MFU is a
slight under-count of the true tensor-core utilisation. Good enough for
tracking run health and comparing against the benchmark.

Run (does NOT execute automatically — you launch it):
    modal run train_modal.py
    modal run train_modal.py --n-steps 5000 --batch-size 8 --lr 1e-4
    modal run train_modal.py --config small        # synth-data smoke test

Resume from a checkpoint on the volume:
    modal run train_modal.py --resume-from step_2000.pkl --n-steps 5000

Stable defaults (see train.py docstring): grad clip 1.0, lr 1e-4, batch >= 8.
"""
import pickle
from pathlib import Path

import modal

# B200 SXM BF16 dense tensor-core peak (TFLOP/s). Same reference as bench.py —
# verify against the NVIDIA product page before trusting MFU numbers.
B200_BF16_TFLOPS = 2_250.0

DATA_VOLUME_NAME = "fineweb-edu-data"
REMOTE_DATA_DIR = "/data"
REAL_TOKENS_FILE = "fineweb-edu-10BT-train.bin"

CKPT_VOLUME_NAME = "train-gpt-checkpoints"
REMOTE_CKPT_DIR = "/checkpoints"

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

app = modal.App("train-gpt-train")
data_vol = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
ckpt_vol = modal.Volume.from_name(CKPT_VOLUME_NAME, create_if_missing=True)


def _tree_to_numpy(tree):
    """Host-side numpy copy of a param pytree, for dependency-free pickling.
    Structure (plain dicts + stacked layer arrays) survives a pickle round-trip
    because it contains only dicts and arrays — no JAX-specific objects."""
    import numpy as np
    import jax

    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)


@app.function(
    image=image,
    gpu="B200",
    memory=48 * 1024,
    timeout=24 * 60 * 60,  # long-running training; cap at 24h per Modal limits
    volumes={REMOTE_DATA_DIR: data_vol, REMOTE_CKPT_DIR: ckpt_vol},
)
def run_train(
    config: str = "full",
    batch_size: int = 8,
    n_steps: int = 5000,
    total_steps: int = 0,
    lr: float = 1e-4,
    warmup_steps: int = 200,
    lr_schedule: str = "cosine",
    lr_end_frac: float = 0.1,
    weight_decay: float = 0.1,
    clip_norm: float = 1.0,
    log_every: int = 50,
    ckpt_every: int = 1000,
    seed: int = 0,
    peak_tflops: float = B200_BF16_TFLOPS,
    resume_from: str = "",
    use_remat: bool = True,
    use_pallas_norm: bool = False,
    gpu_sample_ms: int = 500,
) -> dict:
    import os
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

    import dataclasses
    import math
    import statistics
    import subprocess
    import tempfile
    import time

    import jax
    import jax.numpy as jnp
    import numpy as np

    from config import SMALL, Config
    from data import batch_iter, load_tokens, synth_bin
    from losses import cross_entropy_loss
    from model import init_params, param_count
    from train import make_lr_schedule, make_optimizer, make_train_step

    cfg = SMALL if config == "small" else Config()
    # use_remat: low HBM (+recompute) vs faster/higher-memory.
    # use_pallas_norm defaults False: the Pallas rms_norm kernel NaNs at batch
    # >=64 on B200 (see memory: batch128-bf16-divergence) — pure-XLA is safe.
    cfg = dataclasses.replace(
        cfg, use_remat=use_remat, use_pallas_norm=use_pallas_norm
    )

    # --- FLOP / MFU helpers (copied from bench.py for comparable numbers) ---
    def estimate_flops(cfg: Config, batch_size: int) -> float:
        d, h, ff, L = cfg.d_model, cfg.n_heads * cfg.head_dim, cfg.d_ff, cfg.n_layers
        layer_params = 3 * d * h + h * d + 2 * d * ff + ff * d
        tokens = batch_size * cfg.seq_len
        return 6.0 * L * layer_params * tokens  # fwd+bwd ≈ 6 * N * T

    def mfu(flops_per_step: float, step_time_s: float) -> float:
        achieved_tflops = flops_per_step / step_time_s / 1e12
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

    # --- GPU telemetry sampler (copied verbatim from bench.py so the SM-clock /
    # power readings are directly comparable to a `modal run bench.py` run). The
    # point of running it here: bench sits clamped at the ~1155 MHz base clock
    # and draws ~590 W, while training is observed near the ~1000 W cap. Sampling
    # the *training* SM clock settles whether training boosts (power-bound) or is
    # also clock-clamped. nvidia-smi queries NVML on a separate process, not the
    # CUDA path, so it does not perturb the JAX loop. ---
    class _GpuSampler:
        """Polls nvidia-smi in the background for clock/power/throttle state.

        Degrades to a no-op (returns {}) if nvidia-smi is unavailable.
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

        def __init__(self, interval_ms: int = 500):
            self._interval_ms = interval_ms
            self._proc = None
            self._tmp = None

        def start(self) -> None:
            if self._interval_ms <= 0:
                return
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
                    {"max_draw": max(draw), "limit": max(lim),
                     "mean_draw": round(statistics.fmean(draw), 1)}
                    if draw and lim else {}
                ),
                "throttle_active_frac": reasons,
            }

    # --- corpus selection: real text for FULL, synth fallback for SMALL ---
    real_data_path = Path(REMOTE_DATA_DIR) / REAL_TOKENS_FILE
    if config != "small":
        if not real_data_path.exists():
            raise FileNotFoundError(
                f"expected real corpus at {real_data_path} on volume "
                f"'{DATA_VOLUME_NAME}', but it is missing. Run prepare_data first "
                f"or pass --config small for a synth-data smoke test."
            )
        data_path = real_data_path
        print(f"corpus: {data_path} ({data_path.stat().st_size:,} bytes)")
    else:
        data_path = Path("/tmp/train_tokens.bin")
        if not data_path.exists():
            synth_bin(data_path, cfg.vocab_size, n_tokens=2_000_000, seed=seed)
        print(f"corpus (synth): {data_path}")

    tokens = load_tokens(data_path)
    print(f"loaded {len(tokens):,} tokens")

    # --- params: fresh init or resume from a checkpoint on the volume ---
    key = jax.random.PRNGKey(seed)
    params = init_params(key, cfg)
    start_step = 0
    if resume_from:
        ckpt_path = Path(REMOTE_CKPT_DIR) / resume_from
        if not ckpt_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {ckpt_path}")
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        # Restore as device arrays in the model dtype.
        params = jax.tree_util.tree_map(
            lambda x: jnp.asarray(x, dtype=cfg.dtype), ckpt["params"]
        )
        start_step = int(ckpt.get("step", 0))
        print(f"resumed from {resume_from} at step {start_step}")

    n_params = param_count(params)
    print(f"params: {n_params:,}  (bf16)  use_remat={use_remat}")

    # LR schedule: warmup -> decay (default cosine to 10% of peak). A constant
    # LR only orbits the minimum and plateaus; see make_lr_schedule docstring.
    #
    # `total_steps` is the FULL decay horizon, which may span multiple resumed
    # chunks; `n_steps` is only what THIS process runs. They differ when a run
    # longer than Modal's 24h timeout is split into chunks: the cosine then
    # spans the whole corpus, not each chunk. Defaults to n_steps (single run).
    horizon = total_steps if total_steps > 0 else n_steps
    if horizon < start_step + n_steps:
        raise ValueError(
            f"total_steps ({horizon}) must be >= start_step + n_steps "
            f"({start_step + n_steps}); the schedule would end before the run does"
        )
    base_sched = make_lr_schedule(
        lr, horizon, warmup_steps=warmup_steps,
        end_lr_frac=lr_end_frac, kind=lr_schedule,
    )
    # adamw indexes the schedule by THIS process's optimizer count (0-based), so
    # on resume we shift by start_step to keep one continuous curve across
    # chunks. Off by the single pre-loop warmup update — negligible over 1e4+
    # steps. `base_sched` (un-shifted, global) drives the logged LR readout.
    if callable(base_sched) and start_step > 0:
        def opt_sched(local_count):
            return base_sched(local_count + start_step)
    else:
        opt_sched = base_sched

    def lr_at(global_step: int) -> float:
        return float(base_sched(global_step)) if callable(base_sched) else float(base_sched)

    print(
        f"lr schedule: {lr_schedule} peak={lr:.2e} warmup={warmup_steps} "
        f"end={lr * lr_end_frac:.2e} over {horizon} steps "
        f"(this chunk: {start_step + 1}..{start_step + n_steps})"
    )

    optimizer = make_optimizer(opt_sched, weight_decay, clip_norm)
    opt_state = optimizer.init(params)
    train_step = make_train_step(cfg, optimizer)

    batches = batch_iter(tokens, batch_size, cfg.seq_len, seed=seed + start_step)
    tokens_per_step = batch_size * cfg.seq_len
    flops_per_step = estimate_flops(cfg, batch_size)

    def save_ckpt(params, step: int) -> str:
        """Pickle a host-side numpy copy of params to the checkpoint volume."""
        name = f"step_{step}.pkl"
        path = Path(REMOTE_CKPT_DIR) / name
        payload = {
            "step": step,
            "config": config,
            "params": _tree_to_numpy(params),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        ckpt_vol.commit()
        print(f"  checkpoint saved: {name}")
        return name

    # --- warmup step: compile + fail-fast on a broken setup ---
    inputs, targets = next(batches)
    params, opt_state, loss, grad_norm = train_step(
        params, opt_state, inputs, targets
    )
    jax.block_until_ready((loss, grad_norm))
    warmup_loss = float(loss)
    print(f"compile + warmup done, loss={warmup_loss:.4f}  "
          f"grad_norm={float(grad_norm):.3f}")
    if not math.isfinite(warmup_loss):
        raise RuntimeError(
            f"warmup loss is non-finite ({warmup_loss}); aborting before training"
        )

    history = []  # (step, loss, grad_norm) for the returned summary
    # Start GPU telemetry only now, after compile+warmup, so the clock/power
    # stats reflect steady-state training and not the (idle-ish) compile phase.
    gpu_sampler = _GpuSampler(gpu_sample_ms)
    gpu_sampler.start()
    t0 = time.perf_counter()
    last_log_t, last_log_step = t0, 0
    peak_mfu = 0.0

    for step in range(start_step + 1, start_step + n_steps + 1):
        inputs, targets = next(batches)
        params, opt_state, loss, grad_norm = train_step(
            params, opt_state, inputs, targets
        )

        if step % log_every == 0:
            jax.block_until_ready((loss, grad_norm))
            loss_val = float(loss)
            gn_val = float(grad_norm)
            if not math.isfinite(loss_val):
                # Save what we have so the run isn't a total loss, then abort.
                save_ckpt(params, step)
                raise RuntimeError(
                    f"loss went non-finite at step {step} ({loss_val}); aborting. "
                    f"Try a lower lr or a larger batch_size. (checkpoint saved)"
                )
            now = time.perf_counter()
            dt = now - last_log_t
            steps_done = step - last_log_step
            tps = steps_done * tokens_per_step / dt
            step_time = dt / steps_done
            cur_mfu = mfu(flops_per_step, step_time)
            peak_mfu = max(peak_mfu, cur_mfu)
            done = step - start_step
            eta_s = (n_steps - done) * step_time
            print(f"step {step:6d}  loss {loss_val:7.4f}  "
                  f"grad_norm {gn_val:7.3f}  lr {lr_at(step):.2e}  "
                  f"{tps:>10,.0f} tok/s  "
                  f"MFU {cur_mfu * 100:4.1f}%  ETA {eta_s / 60:5.1f}m")
            history.append((step, loss_val, gn_val))
            last_log_t, last_log_step = now, step

        if ckpt_every > 0 and step % ckpt_every == 0:
            jax.block_until_ready((loss, grad_norm))
            save_ckpt(params, step)

    jax.block_until_ready(loss)
    total_dt = time.perf_counter() - t0
    gpu_telemetry = gpu_sampler.stop()
    final_step = start_step + n_steps

    final_ckpt = ""
    if ckpt_every > 0:
        final_ckpt = save_ckpt(params, final_step)

    avg_tps = n_steps * tokens_per_step / total_dt
    avg_step_time = total_dt / n_steps
    avg_mfu = mfu(flops_per_step, avg_step_time)
    used_gb, limit_gb = memory_gb()

    return {
        "config": config,
        "n_params": n_params,
        "batch_size": batch_size,
        "seq_len": cfg.seq_len,
        "device": str(jax.devices()[0]),
        "device_kind": jax.devices()[0].device_kind,
        "n_steps": n_steps,
        "start_step": start_step,
        "final_step": final_step,
        "tokens_trained": n_steps * tokens_per_step,
        "wall_seconds": total_dt,
        "avg_tok_per_sec": avg_tps,
        "avg_mfu": avg_mfu,
        "peak_mfu": peak_mfu,
        "hbm_peak_gb": used_gb,
        "hbm_limit_gb": limit_gb,
        "gpu_telemetry": gpu_telemetry,
        "final_loss": history[-1][1] if history else warmup_loss,
        "final_grad_norm": history[-1][2] if history else None,
        "final_checkpoint": final_ckpt,
        "peak_tflops": peak_tflops,
    }


@app.local_entrypoint()
def main(
    config: str = "full",
    batch_size: int = 8,
    n_steps: int = 5000,
    total_steps: int = 0,
    lr: float = 1e-4,
    warmup_steps: int = 200,
    lr_schedule: str = "cosine",
    lr_end_frac: float = 0.1,
    weight_decay: float = 0.1,
    clip_norm: float = 1.0,
    log_every: int = 50,
    ckpt_every: int = 1000,
    seed: int = 0,
    peak_tflops: float = B200_BF16_TFLOPS,
    resume_from: str = "",
    use_remat: bool = True,
    use_pallas_norm: bool = False,
    gpu_sample_ms: int = 500,
):
    r = run_train.remote(
        config=config,
        batch_size=batch_size,
        n_steps=n_steps,
        total_steps=total_steps,
        lr=lr,
        warmup_steps=warmup_steps,
        lr_schedule=lr_schedule,
        lr_end_frac=lr_end_frac,
        weight_decay=weight_decay,
        clip_norm=clip_norm,
        log_every=log_every,
        ckpt_every=ckpt_every,
        seed=seed,
        peak_tflops=peak_tflops,
        resume_from=resume_from,
        use_remat=use_remat,
        use_pallas_norm=use_pallas_norm,
        gpu_sample_ms=gpu_sample_ms,
    )

    print(f"\n{'─' * 60}")
    print(f"  config      : {r['config']}  ({r['n_params']:,} params)")
    print(f"  batch       : {r['batch_size']}  seq_len {r['seq_len']}")
    print(f"  device      : {r['device']}  ({r['device_kind']})")
    print(f"  peak ref    : {r['peak_tflops']:,.0f} TFLOP/s  (BF16 dense)")
    print(f"  steps       : {r['start_step']} → {r['final_step']}  "
          f"({r['n_steps']} this run)")
    print(f"{'─' * 60}")
    print(f"  tokens      : {r['tokens_trained']:,}")
    print(f"  wall time   : {r['wall_seconds'] / 60:.1f} min")
    print(f"  throughput  : {r['avg_tok_per_sec']:,.0f} tok/s avg")
    print(f"  MFU         : {r['avg_mfu'] * 100:.1f}% avg   "
          f"{r['peak_mfu'] * 100:.1f}% peak")
    print(f"  HBM peak    : {r['hbm_peak_gb']:.1f} / {r['hbm_limit_gb']:.1f} GB")
    t = r.get("gpu_telemetry") or {}
    sm = t.get("sm_clock_mhz") or {}
    pw = t.get("power_w") or {}
    cap = t.get("sm_clock_cap_mhz")
    if sm and pw:
        clk = (f"SM {sm['min']:.0f}–{sm['max']:.0f} MHz (mean {sm['mean']:.0f}, "
               f"cap {cap:.0f})" if cap else f"SM {sm['min']:.0f}–{sm['max']:.0f} MHz")
        print(f"  gpu state   : {clk}")
        print(f"                {t.get('temp_c_max', 0):.0f}°C, "
              f"mean {pw['mean_draw']:.0f} / max {pw['max_draw']:.0f} / "
              f"limit {pw['limit']:.0f} W   ({t['samples']} samples)")
        reasons = t.get("throttle_active_frac") or {}
        if reasons:
            flags = "  ".join(f"{k} {v:.0%}" for k, v in reasons.items())
            print(f"                ⚠ throttle active: {flags}")
        else:
            print(f"                ✓ no throttle flags raised")
    print(f"  final loss  : {r['final_loss']:.4f}")
    if r["final_grad_norm"] is not None:
        print(f"  final |g|   : {r['final_grad_norm']:.3f}")
    if r["final_checkpoint"]:
        print(f"  checkpoint  : {r['final_checkpoint']}  "
              f"(volume '{CKPT_VOLUME_NAME}')")
    print(f"{'─' * 60}")
    print(f"\n  download checkpoint with:")
    print(f"    modal volume get {CKPT_VOLUME_NAME} "
          f"{r['final_checkpoint'] or 'step_<N>.pkl'} ./")
