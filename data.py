"""Token batch loader. Reads a flat uint16 .bin of token IDs via memmap."""
from pathlib import Path

import numpy as np


def synth_bin(path: Path, vocab_size: int, n_tokens: int, seed: int = 0) -> None:
    """Generate a synthetic .bin for plumbing tests when no real corpus exists."""
    assert vocab_size <= 2**16, "uint16 requires vocab_size <= 65536"
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, vocab_size, size=n_tokens, dtype=np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)


def load_tokens(path: Path) -> np.ndarray:
    return np.memmap(path, dtype=np.uint16, mode="r")


def batch_iter(tokens: np.ndarray, batch_size: int, seq_len: int, seed: int = 0):
    """Yields (input_ids, target_ids) int32 arrays of shape (B, T) forever."""
    rng = np.random.default_rng(seed)
    max_start = len(tokens) - seq_len - 1
    assert max_start > 0, "corpus too small for the requested seq_len"
    while True:
        starts = rng.integers(0, max_start, size=batch_size)
        chunks = np.stack([np.asarray(tokens[s : s + seq_len + 1]) for s in starts])
        chunks = chunks.astype(np.int32)
        yield chunks[:, :-1], chunks[:, 1:]
