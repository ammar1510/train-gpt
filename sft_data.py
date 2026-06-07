"""Instruction-tuning (SFT) data: chat template, loss masking, batch loader.

This module is the single source of truth for the chat format, so training
(prepare_sft_data.py) and inference (generate.py --chat) tokenize identically —
any drift between the two silently degrades the fine-tuned model.

Format (plain text + the GPT-2 end-of-text token as turn separator; the base
vocab has no dedicated chat tokens):

    <eot>User: {user}\nAssistant: {assistant}<eot>User: {user2}\nAssistant: ...

Loss is supervised only on the assistant responses (and each one's closing
<eot>, so the model learns to stop) — see `build_example`. The "Assistant:"
priming carries NO trailing space; the leading space lives with the response
(" {content}") so the BPE of the first response token matches what the model
sees at inference, where the prompt ends at "Assistant:".

Only numpy + a tiktoken-style encoder are needed here (no JAX), so the template
builder is importable from the dataset-prep image as well as the trainer.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def build_example(messages, enc, eot: int) -> tuple[list[int], list[int]] | None:
    """Tokenize one conversation into (token_ids, loss_mask) of equal length.

    `messages` is a list of {"role": "user"|"assistant"|"system"|..., "content":
    str} in chronological order (the HuggingFace No Robots schema). `enc` is a
    tiktoken encoder; `eot` its end-of-text id.

    `loss_mask[j] == 1` marks token j as an assistant-response token (or the
    <eot> that closes a response) — i.e. a token we want the model to learn to
    generate. Everything else (leading <eot>, "User:"/"System:" turns, the
    "Assistant:" priming) is masked to 0.

    Returns None when the conversation contributes no supervised tokens (no
    assistant turn, or only empty ones) — such examples are useless for SFT.
    """
    ids: list[int] = [eot]
    mask: list[int] = [0]
    prev_role: str | None = None

    for i, m in enumerate(messages):
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if role == "assistant":
            # Leading space belongs to the response so its first BPE token
            # matches inference (prompt ends at "Assistant:" with no space).
            seg = enc.encode_ordinary(" " + content) if content else []
            ids += seg
            mask += [1] * len(seg)
            ids.append(eot)          # learn to emit the stop token
            mask.append(1)
        else:
            label = "User:" if role == "user" else f"{role.capitalize()}:"
            # Consecutive context turns aren't separated by an <eot> (only
            # assistant turns emit one), so insert a newline between them.
            sep = "\n" if (prev_role is not None and prev_role != "assistant") else ""
            text = f"{sep}{label} {content}"
            # Prime the response only when an assistant turn actually follows.
            if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                text += "\nAssistant:"
            seg = enc.encode_ordinary(text)
            ids += seg
            mask += [0] * len(seg)
        prev_role = role

    if sum(mask) == 0:
        return None
    return ids, mask


def format_chat_prompt(prompt: str, enc, eot: int) -> list[int]:
    """Token ids for a single-turn inference prompt, matching `build_example`'s
    first turn exactly. Generation should sample after these ids and stop at the
    first <eot>."""
    return [eot] + enc.encode_ordinary(f"User: {prompt}\nAssistant:")


def load_sft(ids_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the padded (N, L) input-id and loss-mask arrays written by
    prepare_sft_data.py. Memory-mapped: the SFT set is tiny but this keeps the
    loader uniform with data.load_tokens."""
    ids = np.load(ids_path, mmap_mode="r")
    mask = np.load(mask_path, mmap_mode="r")
    if ids.shape != mask.shape:
        raise ValueError(
            f"shape mismatch: input_ids {ids.shape} != loss_mask {mask.shape}"
        )
    return ids, mask


def steps_per_epoch(n_examples: int, batch_size: int) -> int:
    """Full batches per epoch (the trailing partial batch is dropped to keep a
    single static (B, L) shape, so jit never recompiles mid-run)."""
    return n_examples // batch_size


def aligned_seq_len(row_len: int, align: int = 128) -> int:
    """The model input length after the next-token shift (row_len - 1), rounded
    UP to a multiple of `align`. cuDNN flash-attention rejects arbitrary
    sequence lengths (e.g. 1023), so SFT batches are padded to this width."""
    t0 = row_len - 1
    return ((t0 + align - 1) // align) * align


def sft_batch_iter(
    input_ids: np.ndarray,
    loss_mask: np.ndarray,
    batch_size: int,
    seed: int = 0,
    align: int = 128,
):
    """Yield (inputs, targets, loss_mask) int32/int32/float32 arrays of shape
    (B, T) forever, reshuffling every epoch.

    Each stored row is a full padded sequence of length L; the usual next-token
    shift gives inputs = row[:-1] and targets = row[1:], and the loss mask is
    shifted to align with targets (mask[1:]) so it marks which *target* tokens
    are supervised. The trailing partial batch each epoch is dropped to keep the
    shape static.

    The sequence dimension is then right-padded from L-1 up to the next multiple
    of `align` (see `aligned_seq_len`) because cuDNN flash-attention only accepts
    certain lengths. The pad columns are token id 0 with loss_mask 0: under
    causal attention the real tokens never attend to trailing positions, and the
    zero mask means the padded targets contribute no loss — so the pad value is
    immaterial to both logits and loss.
    """
    n = input_ids.shape[0]
    if n < batch_size:
        raise ValueError(f"only {n} examples for batch_size {batch_size}")
    pad = aligned_seq_len(input_ids.shape[1], align) - (input_ids.shape[1] - 1)
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(n)
        for start in range(0, n - batch_size + 1, batch_size):
            idx = perm[start : start + batch_size]
            ids = np.asarray(input_ids[idx], dtype=np.int32)
            msk = np.asarray(loss_mask[idx], dtype=np.float32)
            inp, tgt, m = ids[:, :-1], ids[:, 1:], msk[:, 1:]
            if pad:
                width = ((0, 0), (0, pad))
                inp = np.pad(inp, width)
                tgt = np.pad(tgt, width)
                m = np.pad(m, width)
            yield inp, tgt, m
