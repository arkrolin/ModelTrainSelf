"""Datasets for the training backends.

`synthetic_lm` is a hidden-Markov token stream: it has real learnable structure
(a good model must beat the unigram baseline) but needs no download, so the whole
system works offline. `text_file` reads raw bytes, which makes the vocab exactly
256 and keeps the setup trivial.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def make_synthetic_tokens(
    vocab_size: int,
    length: int = 2_000_000,
    n_states: int = 64,
    branch: int = 4,
    emission_width: int = 10,
    seed: int = 0,
) -> np.ndarray:
    """Generate a token stream from a sparse hidden Markov source."""
    rng = np.random.default_rng(seed)
    vocab_size = int(min(vocab_size, 256))

    states = np.arange(n_states)
    transitions = np.zeros((n_states, n_states), dtype=np.float64)
    for i in range(n_states):
        targets = rng.choice(states[(states != i)], size=min(branch, n_states - 1), replace=False)
        weights = rng.dirichlet(np.ones(len(targets)))
        transitions[i, targets] = weights
    transitions /= transitions.sum(axis=1, keepdims=True)

    emissions = np.zeros((n_states, vocab_size), dtype=np.float64)
    for i in range(n_states):
        support = rng.choice(vocab_size, size=min(emission_width, vocab_size), replace=False)
        emissions[i, support] = rng.dirichlet(np.ones(len(support)))

    tokens = np.empty(length, dtype=np.int64)
    state = 0
    chunk = 20_000
    start = 0
    while start < length:
        n = min(chunk, length - start)
        for t in range(n):
            state = rng.choice(n_states, p=transitions[state])
            tokens[start + t] = rng.choice(vocab_size, p=emissions[state])
        start += n
    return tokens


@dataclass
class DataBundle:
    train: np.ndarray
    val: np.ndarray
    vocab_size: int

    def get_batch(self, split: str, batch_size: int, seq_len: int, rng: np.random.Generator):
        source = self.train if split == "train" else self.val
        if source.size <= seq_len + 1:
            raise ValueError(f"{split} split is too small for seq_len={seq_len}")
        high = source.size - seq_len - 1
        starts = rng.integers(0, high, size=batch_size)
        x = np.stack([source[s : s + seq_len] for s in starts])
        y = np.stack([source[s + 1 : s + 1 + seq_len] for s in starts])
        return x, y


def load_bundle(spec, seed: int = 0) -> DataBundle:
    """Build the train/val arrays described by `spec.data`."""
    d = spec.data
    if d.dataset == "text_file":
        if not d.path:
            raise ValueError("data.dataset=text_file requires data.path")
        raw = Path(d.path).read_bytes()
        tokens = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
        vocab = min(d.vocab_size, 256)
        tokens = tokens % vocab
    else:
        tokens = make_synthetic_tokens(d.vocab_size, length=1_200_000, seed=seed)
        vocab = int(min(d.vocab_size, 256))

    if tokens.size < 10_000:
        raise ValueError(f"dataset too small: {tokens.size} tokens")
    cut = int(tokens.size * 0.97)
    return DataBundle(train=tokens[:cut], val=tokens[cut:], vocab_size=vocab)
