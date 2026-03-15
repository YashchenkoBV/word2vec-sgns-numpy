"""word2vec skip-gram with negative sampling in pure NumPy."""

import numpy as np


def build_vocab(tokens, min_count=5):
    """Return (word2id, id2word, counts) filtering words below min_count."""
    freq = {}
    for w in tokens:
        freq[w] = freq.get(w, 0) + 1

    # Sort descending by count, then alphabetically for reproducibility.
    pairs = sorted(
        ((w, c) for w, c in freq.items() if c >= min_count),
        key=lambda wc: (-wc[1], wc[0]),
    )

    id2word = [w for w, _ in pairs]
    word2id = {w: i for i, w in enumerate(id2word)}
    counts = np.array([c for _, c in pairs], dtype=np.int64)
    return word2id, id2word, counts


def subsample(token_ids, counts, t=1e-5, rng=None):
    """Probabilistically discard frequent words (Mikolov C-code formula)."""
    rng = rng or np.random.default_rng()
    freq = counts / counts.sum()
    keep_prob = np.minimum(np.sqrt(t / freq) + t / freq, 1.0)
    mask = rng.random(len(token_ids)) < keep_prob[token_ids]
    return token_ids[mask]


def build_noise_table(counts, exponent=0.75, table_size=10_000_000):
    """Flat lookup table for O(1) negative sampling.  p(w) ∝ count(w)^0.75."""
    powered = np.float64(counts) ** exponent
    probs = powered / powered.sum()

    table = np.zeros(table_size, dtype=np.int32)
    cumulative = np.cumsum(probs) * table_size
    idx = 0
    for i in range(table_size):
        table[i] = idx
        if i >= cumulative[idx]:
            idx = min(idx + 1, len(counts) - 1)
    return table


def sample_negatives(table, shape, rng):
    """Draw word ids from the noise table."""
    return table[rng.integers(0, len(table), size=shape)]


def load_corpus(path, min_count=5, subsample_t=1e-5, seed=42):
    """Load text8-style corpus.  Returns (token_ids, word2id, id2word, counts, noise_table)."""
    rng = np.random.default_rng(seed)

    with open(path, "r") as f:
        tokens = f.read().split()
    print(f"Raw tokens: {len(tokens):,}")

    word2id, id2word, counts = build_vocab(tokens, min_count)
    V = len(id2word)
    print(f"Vocab size: {V:,} (min_count={min_count})")

    # Encode, drop OOV.
    ids = np.array([word2id[w] for w in tokens if w in word2id], dtype=np.int32)
    ids = subsample(ids, counts, t=subsample_t, rng=rng)
    print(f"After subsampling: {len(ids):,} tokens")

    noise_table = build_noise_table(counts)
    return ids, word2id, id2word, counts, noise_table