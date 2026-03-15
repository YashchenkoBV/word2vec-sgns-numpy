"""word2vec skip-gram with negative sampling in pure NumPy."""

import numpy as np


def build_vocab(tokens, min_count=5):
    """Return (word2id, id2word, counts) keeping only words with count >= min_count."""
    freq = {}
    for w in tokens:
        freq[w] = freq.get(w, 0) + 1

    # Sort descending by count, then alphabetically
    pairs = sorted(
        ((w, c) for w, c in freq.items() if c >= min_count),
        key=lambda wc: (-wc[1], wc[0]),
    )

    id2word = [w for w, _ in pairs]
    word2id = {w: i for i, w in enumerate(id2word)}
    counts = np.array([c for _, c in pairs], dtype=np.int64)
    return word2id, id2word, counts


def subsample(token_ids: np.ndarray, counts, t=1e-5, rng=None):
    """Probabilistically discard frequent words (Mikolov C-code formula)."""
    assert isinstance(token_ids, np.ndarray), "token_ids must be a numpy array"
    rng = rng or np.random.default_rng()
    freq = counts / counts.sum()
    keep_prob = np.minimum(np.sqrt(t / freq) + t / freq, 1.0)  # Mikolov's subsampling formula
    mask = rng.random(len(token_ids)) < keep_prob[token_ids]
    return token_ids[mask]


def build_unigram_table(counts, exponent=0.75, table_size=10_000_000):
    """Flat lookup table for O(1) negative sampling.  p(w) ∝ count(w)^0.75."""
    powered = np.float64(counts) ** exponent
    probs = powered / powered.sum()
    cumulative = np.cumsum(probs)
    table = np.searchsorted(cumulative, np.arange(table_size) / table_size).astype(np.int32)
    return table


def sample_negatives(table, shape, rng):
    """Draw word ids from the noise table."""
    return table[rng.integers(0, len(table), size=shape)]


def load_corpus(path, min_count=5, subsample_t=1e-5, seed=42):
    """Load text8-style corpus.  Returns (token_ids, word2id, id2word, counts, unigram_table)."""
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

    noise_table = build_unigram_table(counts)
    return ids, word2id, id2word, counts, noise_table


def log_sigmoid(x):
    return -np.logaddexp(0, -x)


class SGNS:
    """Skip-gram with negative sampling, temperature-scaled scores."""

    def __init__(self, V, d, tau=1.0, seed=42):
        rng = np.random.default_rng(seed)
        scale = 0.5 / d
        self.W_in = rng.uniform(-scale, scale, (V, d))  # center embeddings
        self.W_out = rng.uniform(-scale, scale, (V, d))  # context embeddings
        self.tau = tau
        self.V = V
        self.d = d

    def forward(self, center, context, negatives):
        """Compute SGNS loss for a batch.

        Args:
            center:    (B,)    int array — center word ids
            context:   (B,)    int array — true context word ids
            negatives: (B, K)  int array — negative sample ids

        Returns:
            Scalar mean loss over the batch.
        """
        v_c = self.W_in[center]  # (B, d)
        u_o = self.W_out[context]  # (B, d)
        u_neg = self.W_out[negatives]  # (B, K, d)

        # Temperature-scaled dot products.
        pos_score = np.sum(v_c * u_o, axis=1) / self.tau  # (B,)
        neg_scores = np.sum(u_neg * v_c[:, None, :], axis=2) / self.tau  # (B, K)

        # L = -log σ(pos) - Σ log σ(-neg)
        loss = -log_sigmoid(pos_score) - log_sigmoid(-neg_scores).sum(axis=1)
        return loss.mean()
