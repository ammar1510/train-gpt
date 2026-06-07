"""Autoregressive text generation from a training checkpoint.

Loads a checkpoint pickle written by `train_modal.py` (a dict
`{"step", "config", "params"}` whose `params` is a host-side numpy pytree),
rebuilds the model config, and samples tokens with the same GPT-2 BPE tokenizer
the corpus was prepared with (see prepare_data.py: tiktoken "gpt2", eot per doc).

This is the reusable, backend-agnostic core. It runs on whatever JAX backend is
available locally (CPU/Metal/GPU). For the FULL 1.76B config, CPU generation is
slow (seconds-to-minutes per token) — prefer the Modal GPU path in
`generate_modal.py`, which reads the checkpoint straight off the volume.

Local usage (after downloading a checkpoint to ./):
    modal volume get train-gpt-checkpoints step_10000.pkl ./
    uv run python generate.py --checkpoint step_10000.pkl \
        --prompt "The mitochondria is" --max-new-tokens 100

Greedy decoding:
    uv run python generate.py --checkpoint step_10000.pkl --prompt "Once upon" \
        --temperature 0
"""
from __future__ import annotations

import argparse
import dataclasses
import pickle
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from config import SMALL, Config
from model import forward
from sft_data import format_chat_prompt

# Padding bucket for the fixed-shape forward pass. The context is right-padded
# up to the next multiple of this so the jitted forward only recompiles at
# bucket boundaries (<= seq_len/BUCKET distinct shapes) instead of once per
# generated token. Causal masking guarantees the real tokens never attend to
# trailing padding, so the padded positions cannot corrupt the logits we read.
_BUCKET = 128


def load_checkpoint(path: Path) -> tuple[dict, Config, int]:
    """Read a checkpoint pickle and return (params_pytree, cfg, step).

    The returned config forces the inference-safe variant: the pure-XLA rms_norm
    (the Pallas kernel has miscompiled on B200 and isn't needed here) and no
    activation remat (remat only matters for the backward pass; forward-only
    inference doesn't run one).
    """
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    with open(path, "rb") as f:
        ckpt = pickle.load(f)

    for key in ("params", "config"):
        if key not in ckpt:
            raise ValueError(
                f"checkpoint {path} is missing '{key}'; expected a dict written "
                f"by train_modal.py with keys step/config/params"
            )

    base = SMALL if ckpt["config"] == "small" else Config()
    cfg = dataclasses.replace(base, use_pallas_norm=False, use_remat=False)

    # Materialise the numpy pytree as device arrays in the model dtype so the
    # forward pass matches training numerics.
    params = jax.tree_util.tree_map(
        lambda x: jnp.asarray(x, dtype=cfg.dtype), ckpt["params"]
    )
    step = int(ckpt.get("step", 0))
    return params, cfg, step


@partial(jax.jit, static_argnames=("cfg",))
def _forward_jit(params, input_ids, cfg: Config):
    # Config is a frozen (hashable) dataclass, so it's a valid static arg; the
    # cache keys on (param shapes/dtypes, input shape, cfg).
    return forward(params, input_ids, cfg)


def _sample_next(logits: np.ndarray, temperature: float, top_k: int,
                 top_p: float, rng: np.random.Generator) -> int:
    """Sample one token id from a (vocab,) logit vector.

    temperature == 0 is greedy (argmax). top_k <= 0 disables top-k; top_p >= 1
    disables nucleus filtering. Filters compose: top-k first, then top-p.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if temperature == 0.0:
        return int(logits.argmax())

    logits = logits / temperature

    if top_k and top_k > 0:
        k = min(top_k, logits.shape[-1])
        kth = np.partition(logits, -k)[-k]  # k-th largest logit
        logits = np.where(logits < kth, -np.inf, logits)

    # Numerically stable softmax.
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()

    if top_p < 1.0:
        order = np.argsort(probs)[::-1]
        cum = np.cumsum(probs[order])
        # Keep the minimal prefix whose cumulative mass reaches top_p (a token is
        # kept when the mass *before* it is still under the threshold), so the
        # top token is always retained.
        keep = cum - probs[order] < top_p
        removed = order[~keep]
        probs[removed] = 0.0
        probs /= probs.sum()

    return int(rng.choice(probs.shape[-1], p=probs))


def generate(
    params,
    cfg: Config,
    prompt_ids: list[int],
    max_new_tokens: int,
    *,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 1.0,
    seed: int = 0,
    eot_token: int | None = None,
    stop_on_eot: bool = True,
) -> list[int]:
    """Autoregressively sample up to `max_new_tokens` token ids after the prompt.

    Returns only the newly generated ids (not the prompt). Stops early if the
    end-of-text token is sampled and `stop_on_eot` is set.
    """
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be > 0, got {max_new_tokens}")
    if temperature < 0.0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if not prompt_ids:
        raise ValueError("prompt_ids is empty; pass at least one token")

    rng = np.random.default_rng(seed)
    seq_len = cfg.seq_len
    ids = list(prompt_ids)
    new: list[int] = []

    for _ in range(max_new_tokens):
        # The learned positional embedding only covers `seq_len` positions, so
        # keep a sliding window of the most recent tokens as context.
        context = ids[-seq_len:]
        length = len(context)

        padded_len = min(seq_len, ((length + _BUCKET - 1) // _BUCKET) * _BUCKET)
        buf = np.zeros((1, padded_len), dtype=np.int32)
        buf[0, :length] = context

        logits = _forward_jit(params, jnp.asarray(buf), cfg)
        # Read the prediction at the last real position; padding is ignored.
        next_logits = np.asarray(logits[0, length - 1], dtype=np.float32)

        tok = _sample_next(next_logits, temperature, top_k, top_p, rng)
        ids.append(tok)
        new.append(tok)

        if stop_on_eot and eot_token is not None and tok == eot_token:
            break

    return new


def _build_tokenizer():
    """Return the tiktoken GPT-2 encoder used to prepare the corpus.

    Imported lazily so the core generate()/load_checkpoint() functions stay
    usable in environments where only token ids are passed (e.g. tests).
    """
    import tiktoken  # approved dep (pyproject.toml); GPT-2 BPE matches prepare_data.py

    return tiktoken.get_encoding("gpt2")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="path to a step_<N>.pkl checkpoint")
    parser.add_argument("--prompt", default="",
                        help="text prompt (empty -> start from end-of-text)")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="0 = greedy/argmax")
    parser.add_argument("--top-k", type=int, default=50, help="0 disables")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="nucleus threshold; 1.0 disables")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-prepend-eot", action="store_true",
                        help="do not prepend the end-of-text token to the prompt "
                             "(training prepended it to every document)")
    parser.add_argument("--chat", action="store_true",
                        help="wrap the prompt in the instruction chat template "
                             "(<eot>User: ...\\nAssistant:) for a fine-tuned "
                             "model; prints only the assistant response")
    args = parser.parse_args()

    if args.num_samples <= 0:
        parser.error("--num-samples must be > 0")

    params, cfg, step = load_checkpoint(args.checkpoint)
    enc = _build_tokenizer()
    eot = enc.eot_token

    print(f"loaded {args.checkpoint} (step {step}, config seq_len={cfg.seq_len}) "
          f"on {jax.default_backend()}")
    if cfg.vocab_size != enc.n_vocab:
        # Mismatch would mean the checkpoint wasn't trained with this tokenizer.
        print(f"  WARNING: cfg.vocab_size={cfg.vocab_size} != tokenizer "
              f"n_vocab={enc.n_vocab}; output may be garbage")

    if args.chat:
        if not args.prompt:
            parser.error("--chat requires a non-empty --prompt")
        # Chat template owns its leading <eot>; --no-prepend-eot is irrelevant here.
        prompt_ids = format_chat_prompt(args.prompt, enc, eot)
    else:
        prompt_ids = enc.encode_ordinary(args.prompt) if args.prompt else []
        if not args.no_prepend_eot:
            prompt_ids = [eot] + prompt_ids
        if not prompt_ids:  # empty prompt + --no-prepend-eot
            prompt_ids = [eot]

    for i in range(args.num_samples):
        new_ids = generate(
            params, cfg, prompt_ids, args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            seed=args.seed + i, eot_token=eot, stop_on_eot=True,
        )
        text = enc.decode(new_ids)
        header = f"sample {i + 1}/{args.num_samples}" if args.num_samples > 1 else "sample"
        print(f"\n{'─' * 60}\n{header}\n{'─' * 60}")
        if args.chat:
            # The response carries a leading space from the template; strip it
            # so the printed answer reads naturally under the prompt.
            print(f"User: {args.prompt}\nAssistant: {text.strip()}")
        else:
            print(args.prompt + text)


if __name__ == "__main__":
    main()
