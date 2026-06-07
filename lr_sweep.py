"""Learning-rate sweep at batch 128 to pick the LR for the full run.

lr=1e-4 was tuned at batch 8. At batch 128 (16x larger) the gradient is far
less noisy, so the optimal LR is higher — the Adam/AdamW rule of thumb is
LR ∝ √batch, i.e. √16 = 4x, pointing at ~4e-4. This sweep confirms that
empirically instead of trusting the heuristic blind: it finds the largest LR
that is still stable and makes the most loss progress per token. Cheap insurance
(4 x ~300 steps) on an 11k+ step real run.

Design:
  - Every LR starts from the SAME init (same seed) and consumes the SAME data
    batches (deterministic batch_iter), so loss differences are purely the LR.
  - Short linear warmup -> constant peak LR, so a high peak does not diverge on
    the cold-init grad spike (we saw |g|~1016 at step 0). Grad clipping at 1.0
    stays on, as in the production recipe.
  - Each LR runs on its own B200 container in parallel (spawn/get).
  - A run that goes non-finite is recorded and stopped early — no wasted compute.

Run:
    modal run lr_sweep.py
    modal run lr_sweep.py --n-steps 400 --warmup-steps 60

Reads the real corpus from the `fineweb-edu-data` volume. Writes nothing.
"""
from pathlib import Path

import modal

B200_BF16_TFLOPS = 2_250.0

DATA_VOLUME_NAME = "fineweb-edu-data"
REMOTE_DATA_DIR = "/data"
REAL_TOKENS_FILE = "fineweb-edu-10BT-train.bin"

# √batch heuristic from the batch-8 lr=1e-4 baseline points at ~4e-4; bracket it.
DEFAULT_LRS = [1e-4, 2e-4, 4e-4, 6e-4]

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

app = modal.App("train-gpt-lr-sweep")
data_vol = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    gpu="B200",
    memory=48 * 1024,
    timeout=60 * 60,
    volumes={REMOTE_DATA_DIR: data_vol},
)
def run_lr(
    peak_lr: float,
    n_steps: int = 300,
    warmup_steps: int = 40,
    batch_size: int = 128,
    weight_decay: float = 0.1,
    clip_norm: float = 1.0,
    log_every: int = 20,
    seed: int = 0,
    use_pallas_norm: bool = False,  # Pallas rms_norm NaNs at batch >=64 — default off
) -> dict:
    import os
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

    import dataclasses
    import math

    import jax
    import optax

    from config import Config
    from data import batch_iter, load_tokens
    from losses import cross_entropy_loss  # noqa: F401  (used via make_train_step)
    from model import init_params
    from train import make_optimizer, make_train_step

    cfg = Config()  # use_remat=True by default — required to fit batch 128
    cfg = dataclasses.replace(cfg, use_pallas_norm=use_pallas_norm)

    data_path = Path(REMOTE_DATA_DIR) / REAL_TOKENS_FILE
    if not data_path.exists():
        raise FileNotFoundError(
            f"real corpus missing at {data_path} on volume '{DATA_VOLUME_NAME}'"
        )
    tokens = load_tokens(data_path)

    key = jax.random.PRNGKey(seed)
    params = init_params(key, cfg)

    # Warmup -> constant schedule. adamw accepts a schedule callable directly.
    if warmup_steps > 0:
        schedule = optax.join_schedules(
            [
                optax.linear_schedule(0.0, peak_lr, warmup_steps),
                optax.constant_schedule(peak_lr),
            ],
            boundaries=[warmup_steps],
        )
    else:
        schedule = peak_lr

    optimizer = make_optimizer(schedule, weight_decay, clip_norm)
    opt_state = optimizer.init(params)
    train_step = make_train_step(cfg, optimizer)

    batches = batch_iter(tokens, batch_size, cfg.seq_len, seed=seed)

    # Warmup step: compile + record the cold-init pre-clip grad norm.
    inputs, targets = next(batches)
    params, opt_state, loss, grad_norm = train_step(
        params, opt_state, inputs, targets
    )
    jax.block_until_ready((loss, grad_norm))
    warmup_loss = float(loss)
    warmup_gn = float(grad_norm)

    traj = []  # list of (step, loss, pre_clip_grad_norm) at logged steps
    diverged_at = None
    max_gn = warmup_gn

    for step in range(1, n_steps + 1):
        inputs, targets = next(batches)
        params, opt_state, loss, grad_norm = train_step(
            params, opt_state, inputs, targets
        )
        if step % log_every == 0:
            jax.block_until_ready((loss, grad_norm))
            lv, gnv = float(loss), float(grad_norm)
            max_gn = max(max_gn, gnv)
            traj.append((step, lv, gnv))
            if not math.isfinite(lv):
                diverged_at = step
                break  # stop a diverged run early; don't burn the container

    finite_losses = [lv for (_, lv, _) in traj if math.isfinite(lv)]
    return {
        "peak_lr": peak_lr,
        "warmup_steps": warmup_steps,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "warmup_loss": warmup_loss,
        "warmup_grad_norm": warmup_gn,
        "trajectory": traj,
        "diverged_at": diverged_at,
        "final_loss": finite_losses[-1] if finite_losses else float("nan"),
        "min_loss": min(finite_losses) if finite_losses else float("nan"),
        "max_grad_norm": max_gn,
    }


@app.local_entrypoint()
def main(
    n_steps: int = 300,
    warmup_steps: int = 40,
    batch_size: int = 128,
    seed: int = 0,
    lrs: str = "",
):
    import math

    sweep_lrs = (
        [float(x) for x in lrs.split(",")] if lrs else DEFAULT_LRS
    )
    print(f"LR sweep: {sweep_lrs}")
    print(f"batch={batch_size}  steps={n_steps}  warmup={warmup_steps}  seed={seed}")
    print(f"launching {len(sweep_lrs)} B200 containers in parallel ...\n")

    # spawn() returns immediately; the containers run concurrently. get() blocks.
    handles = [
        run_lr.spawn(
            lr, n_steps=n_steps, warmup_steps=warmup_steps,
            batch_size=batch_size, seed=seed,
        )
        for lr in sweep_lrs
    ]
    results = [h.get() for h in handles]
    results.sort(key=lambda r: r["peak_lr"])

    # Per-LR loss trajectory at aligned step boundaries.
    log_steps = sorted({s for r in results for (s, _, _) in r["trajectory"]})
    show = [s for s in log_steps if s in (log_steps[:1] + log_steps[-1:])
            or s % 60 == 0]
    print(f"\n{'LOSS TRAJECTORY':-^72}")
    header = "  " + f"{'peak LR':>9}  " + "  ".join(f"@{s:<5}" for s in show)
    print(header)
    for r in results:
        d = {s: lv for (s, lv, _) in r["trajectory"]}
        cells = []
        for s in show:
            cells.append(f"{d[s]:6.3f}" if s in d and math.isfinite(d[s])
                         else "  -  ")
        print(f"  {r['peak_lr']:>9.1e}  " + "  ".join(f"{c:>6}" for c in cells))

    print(f"\n{'SUMMARY':-^72}")
    print(f"  {'peak LR':>9}  {'final':>7}  {'min':>7}  {'max |g|':>9}  {'status':>12}")
    for r in results:
        status = (f"DIVERGED@{r['diverged_at']}" if r["diverged_at"]
                  else "ok")
        print(f"  {r['peak_lr']:>9.1e}  {r['final_loss']:>7.3f}  "
              f"{r['min_loss']:>7.3f}  {r['max_grad_norm']:>9.1f}  {status:>12}")

    # Recommendation: largest LR that did not diverge AND has the lowest final
    # loss. Lowest final loss among stable runs is the most token-efficient.
    stable = [r for r in results if not r["diverged_at"]
              and math.isfinite(r["final_loss"])]
    if stable:
        best = min(stable, key=lambda r: r["final_loss"])
        print(f"\n  -> best by final loss: lr={best['peak_lr']:.1e} "
              f"(loss {best['final_loss']:.3f}, max|g| {best['max_grad_norm']:.1f})")
        if best["peak_lr"] == max(r["peak_lr"] for r in stable):
            print("     NOTE: best is the LARGEST LR tested and stable — the true "
                  "optimum may be higher. Consider extending the sweep upward.")
    else:
        print("\n  -> all runs diverged; lower the LR range or lengthen warmup.")


@app.function(
    image=image,
    gpu="B200",
    memory=48 * 1024,
    timeout=30 * 60,
    volumes={REMOTE_DATA_DIR: data_vol},
)
def probe_nan(batch_size: int = 64, data_mode: str = "real", n_steps: int = 12,
              lr: float = 1e-4, seed: int = 0) -> dict:
    """Pinpoint the first non-finite at a diverging batch: is it the FORWARD
    (loss) or a GRADIENT, and which param group? data_mode 'real' uses the
    corpus; 'random' uses random token ids of the same shape — comparing the two
    tells us whether the NaN is data-triggered or batch/op-driven."""
    import os
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

    import math

    import jax
    import jax.numpy as jnp
    import optax

    from config import Config
    from data import batch_iter, load_tokens
    from losses import cross_entropy_loss
    from model import init_params
    from train import make_optimizer

    cfg = Config()

    def make_batches():
        if data_mode == "real":
            path = Path(REMOTE_DATA_DIR) / REAL_TOKENS_FILE
            tokens = load_tokens(path)
            return batch_iter(tokens, batch_size, cfg.seq_len, seed=seed)
        # random: fresh uniform token ids each step, same shapes.
        def gen():
            k = jax.random.PRNGKey(seed)
            i = 0
            while True:
                k, ka, kb = jax.random.split(k, 3)
                shape = (batch_size, cfg.seq_len)
                yield (jax.random.randint(ka, shape, 0, cfg.vocab_size, jnp.int32),
                       jax.random.randint(kb, shape, 0, cfg.vocab_size, jnp.int32))
                i += 1
        return gen()

    batches = make_batches()
    key = jax.random.PRNGKey(seed)
    params = init_params(key, cfg)
    optimizer = make_optimizer(lr, 0.1, 1.0)
    opt_state = optimizer.init(params)

    # Mirror make_train_step's graph EXACTLY (optimizer ops fused in) and just
    # additionally return the raw grads. A bare jit(value_and_grad) jitted in
    # isolation trips the scan+checkpoint XLA artifact (NaN even on batches that
    # train fine), so we must inspect grads through the real training graph.
    @jax.jit
    def step_with_grads(params, opt_state, inputs, targets):
        loss, grads = jax.value_and_grad(cross_entropy_loss)(
            params, inputs, targets, cfg
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss, grads

    def grad_report(grads):
        """Per-leaf (finite, max|.|). Non-finite leaves name the culprit op."""
        out = []
        for path, g in jax.tree_util.tree_leaves_with_path(grads):
            ga = g.astype(jnp.float32)
            finite = bool(jnp.isfinite(ga).all())
            mx = float(jnp.abs(jnp.where(jnp.isfinite(ga), ga, 0.0)).max())
            out.append((jax.tree_util.keystr(path), finite, mx))
        return out

    history = []
    culprit = None
    for step in range(1, n_steps + 1):
        inputs, targets = next(batches)
        new_params, opt_state, loss, grads = step_with_grads(
            params, opt_state, inputs, targets
        )
        loss_v = float(loss)
        rep = grad_report(grads)
        bad = [(name, mx) for (name, finite, mx) in rep if not finite]
        history.append((step, loss_v, len(bad)))
        if (not math.isfinite(loss_v)) or bad:
            culprit = {
                "step": step,
                "loss_finite": math.isfinite(loss_v),
                "loss": loss_v,
                "bad_leaves": [n for (n, _) in bad],
                # Top finite grads by magnitude — what was largest just before NaN.
                "top_mag": sorted(
                    [(n, mx) for (n, f, mx) in rep if f],
                    key=lambda t: -t[1])[:5],
            }
            break
        params = new_params

    return {
        "batch_size": batch_size, "data_mode": data_mode, "lr": lr,
        "history": history, "culprit": culprit,
    }


@app.local_entrypoint()
def probe(n_steps: int = 12, seed: int = 0):
    """Data-vs-op fork + first-non-finite localization at a diverging batch.
    Run as: modal run lr_sweep.py::probe
    """
    cases = [
        (64, "real"),    # reproduce divergence on real data
        (64, "random"),  # same shape, random tokens — finite => it's the DATA
        (32, "real"),    # known-stable control
    ]
    handles = [(bs, dm, probe_nan.spawn(batch_size=bs, data_mode=dm,
                                        n_steps=n_steps, seed=seed))
               for (bs, dm) in cases]
    for bs, dm, h in handles:
        r = h.get()
        print(f"\n=== batch {bs}, {dm} ===")
        for (s, lv, nbad) in r["history"]:
            print(f"   step {s:2d}  loss {lv:10.4f}  bad_grad_leaves={nbad}")
        c = r["culprit"]
        if c is None:
            print("   -> FINITE throughout (no divergence)")
        else:
            print(f"   -> first non-finite @ step {c['step']}: "
                  f"loss_finite={c['loss_finite']}")
            if c["bad_leaves"]:
                print(f"      non-finite grads: {c['bad_leaves']}")
            print(f"      largest finite grads just before: "
                  + ", ".join(f"{n}={mx:.2e}" for n, mx in c["top_mag"]))


@app.local_entrypoint()
def norm_ab(batch_size: int = 64, n_steps: int = 30, lr: float = 1e-4,
            seed: int = 0):
    """2x2: {warmup, constant} x {Pallas rms_norm, pure-XLA rms_norm} at a fixed
    batch known to NaN. Isolates whether the trigger is the warmup schedule, the
    Pallas kernel, or their interaction. Run as: modal run lr_sweep.py::norm_ab
    """
    import math

    cases = [
        ("warmup20 + pallas", 20, True),
        ("warmup20 + pure",   20, False),
        ("constant + pallas",  0, True),
        ("constant + pure",    0, False),
    ]
    print(f"norm A/B at batch {batch_size}, lr {lr:.1e}, {n_steps} steps\n")
    handles = [
        (label, run_lr.spawn(
            lr, n_steps=n_steps, warmup_steps=wu, batch_size=batch_size,
            log_every=2, seed=seed, use_pallas_norm=pallas,
        ))
        for (label, wu, pallas) in cases
    ]
    print(f"{'config':>22}  {'final loss':>10}  {'max |g|':>10}  {'status':>14}")
    for label, h in handles:
        r = h.get()
        status = f"DIVERGED@{r['diverged_at']}" if r["diverged_at"] else "ok"
        print(f"{label:>22}  {r['final_loss']:>10.3f}  "
              f"{r['max_grad_norm']:>10.1f}  {status:>14}")


@app.local_entrypoint()
def batch_sweep(n_steps: int = 60, warmup_steps: int = 20, seed: int = 0,
                lr: float = 1e-4, batches: str = "16,32,64,96"):
    """Find the largest TRUE batch that trains stably in bf16 on real data, at
    the known-good lr=1e-4. Each batch runs to peak LR (after warmup) and beyond
    so survival reflects the real operating point, not just the warmup ramp.
    Run as: modal run lr_sweep.py::batch_sweep
    """
    import math

    bs_list = [int(x) for x in batches.split(",")]
    print(f"batch-ceiling sweep: batches={bs_list}  lr={lr:.1e}  "
          f"steps={n_steps}  warmup={warmup_steps}\n")
    handles = [
        (bs, run_lr.spawn(
            lr, n_steps=n_steps, warmup_steps=warmup_steps,
            batch_size=bs, log_every=4, seed=seed,
        ))
        for bs in bs_list
    ]
    results = [(bs, h.get()) for bs, h in handles]

    print(f"\n{'SUMMARY':-^64}")
    print(f"  {'batch':>6}  {'final loss':>10}  {'max |g|':>10}  {'status':>14}")
    largest_stable = None
    for bs, r in results:
        status = f"DIVERGED@{r['diverged_at']}" if r["diverged_at"] else "ok"
        if not r["diverged_at"] and math.isfinite(r["final_loss"]):
            largest_stable = bs
        print(f"  {bs:>6}  {r['final_loss']:>10.3f}  "
              f"{r['max_grad_norm']:>10.1f}  {status:>14}")

    if largest_stable:
        print(f"\n  -> largest stable batch: {largest_stable} "
              f"(of {bs_list}); re-tune LR there next.")
    else:
        print(f"\n  -> none of {bs_list} stable; the ceiling is below "
              f"{min(bs_list)} — fall back to batch 8 or fix the overflow.")


@app.local_entrypoint()
def diagnose(n_steps: int = 30, seed: int = 0):
    """Isolate the uniform divergence: known-good control (batch 8) vs batch 128,
    every step logged. Run as: modal run lr_sweep.py::diagnose

    A finite control + diverging batch-128 => batch size is the trigger.
    A diverging control => the config change (tie_embeddings=False) is the cause.
    """
    import math

    # (label, lr, batch, warmup)
    cases = [
        ("A b8  lr1e-4 wu0",   1e-4,   8, 0),    # old known-good recipe
        ("B b128 lr1e-4 wu40", 1e-4, 128, 40),   # reproduce divergence
        ("C b128 lr1e-5 wu100", 1e-5, 128, 100),  # much gentler — survivable?
    ]
    print(f"divergence diagnostic: {n_steps} steps, every-step logging\n")
    handles = [
        (label, run_lr.spawn(
            lr, n_steps=n_steps, warmup_steps=wu, batch_size=bs,
            log_every=1, seed=seed,
        ))
        for (label, lr, bs, wu) in cases
    ]
    for label, h in handles:
        r = h.get()
        status = f"DIVERGED@{r['diverged_at']}" if r["diverged_at"] else "ok"
        print(f"\n=== {label} ===  warmup_loss={r['warmup_loss']:.3f}  "
              f"warmup|g|={r['warmup_grad_norm']:.1f}  [{status}]")
        for (s, lv, gn) in r["trajectory"]:
            tag = "  <-- non-finite" if not math.isfinite(lv) else ""
            print(f"   step {s:3d}  loss {lv:9.4f}  |g| {gn:11.1f}{tag}")
