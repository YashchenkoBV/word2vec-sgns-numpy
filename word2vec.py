"""word2vec skip-gram with negative sampling in pure NumPy."""

import numpy as np


# ── Vocabulary ───────────────────────────────────────────────────────────

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


# ── Subsampling ──────────────────────────────────────────────────────────

def subsample(token_ids: np.ndarray, counts, t=1e-5, rng=None):
    """Probabilistically discard frequent words.  P(keep) = sqrt(t / f(w))."""
    assert isinstance(token_ids, np.ndarray), "token_ids must be a numpy array"
    rng = rng or np.random.default_rng()
    freq = counts / counts.sum()
    keep_prob = np.minimum(np.sqrt(t / freq), 1.0)
    mask = rng.random(len(token_ids)) < keep_prob[token_ids]
    return token_ids[mask]


# ── Noise distribution ───────────────────────────────────────────────────

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


# ── Corpus loading ───────────────────────────────────────────────────────

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

    unigram_table = build_unigram_table(counts)
    return ids, word2id, id2word, counts, unigram_table


# ── Numerics ─────────────────────────────────────────────────────────────

def sigmoid(x):
    """Numerically stable σ(x)."""
    pos = x >= 0
    z = np.exp(-np.abs(x))
    out = np.where(pos, 1.0 / (1.0 + z), z / (1.0 + z))
    return out


def log_sigmoid(x):
    """Numerically stable log(σ(x)) = -log(1 + exp(-x))."""
    return -np.logaddexp(0, -x)


# ── Model ────────────────────────────────────────────────────────────────

def _scatter_add(ids, grads, d):
    """Sum per-example grads that share a row id.  Returns (unique_ids, summed_grads)."""
    unique = np.unique(ids)
    local = np.searchsorted(unique, ids)
    out = np.zeros((len(unique), d))
    np.add.at(out, local, grads)
    return unique, out


class SGNS:
    """Skip-gram with negative sampling, temperature-scaled scores.

    Loss for one (center, context, neg_1..neg_K) example:
        L = -log σ(v_c · u_o / τ) - Σ_k log σ(-v_c · u_k / τ)
    """

    def __init__(self, V, d, tau=1.0, seed=42):
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(d)
        self.W_in = rng.uniform(-scale, scale, (V, d))
        self.W_out = np.zeros((V, d))  # standard: zero-init context embeddings
        self.tau = tau
        self.V = V
        self.d = d

    def train_step(self, center, context, negatives, grad_clip=5.0):
        """Forward + backward for a batch.

        Args:
            center:    (B,)    int — center word ids
            context:   (B,)    int — positive context word ids
            negatives: (B, K)  int — negative sample ids
            grad_clip: float   — element-wise gradient clamp bound

        Returns:
            loss:     scalar (mean over batch)
            in_grad:  (unique_in_ids, grad_array)  — sparse gradient for W_in
            out_grad: (unique_out_ids, grad_array)  — sparse gradient for W_out
        """
        B, K = negatives.shape
        tau = self.tau

        # ── forward ──
        v_c = self.W_in[center]                                      # (B, d)
        u_o = self.W_out[context]                                    # (B, d)
        u_neg = self.W_out[negatives]                                # (B, K, d)

        pos_score = np.sum(v_c * u_o, axis=1) / tau                 # (B,)
        neg_scores = np.sum(u_neg * v_c[:, None, :], axis=2) / tau  # (B, K)

        loss = (-log_sigmoid(pos_score) - log_sigmoid(-neg_scores).sum(axis=1)).mean()

        # ── backward (sum gradient; mean loss is for logging only) ──
        # dL/ds_pos = σ(s) - 1,  dL/ds_neg_k = σ(s_k)
        sig_pos = sigmoid(pos_score)     # (B,)
        sig_neg = sigmoid(neg_scores)    # (B, K)

        g_vc = ((sig_pos - 1)[:, None] * u_o
                + (sig_neg[:, :, None] * u_neg).sum(axis=1)) / tau   # (B, d)
        g_uo = (sig_pos - 1)[:, None] * v_c / tau                    # (B, d)
        g_un = sig_neg[:, :, None] * v_c[:, None, :] / tau           # (B, K, d)

        np.clip(g_vc, -grad_clip, grad_clip, out=g_vc)
        np.clip(g_uo, -grad_clip, grad_clip, out=g_uo)
        np.clip(g_un, -grad_clip, grad_clip, out=g_un)

        # Scatter-sum into sparse gradient arrays.
        in_grad = _scatter_add(center, g_vc, self.d)

        out_ids = np.concatenate([context, negatives.ravel()])
        out_g = np.concatenate([g_uo, g_un.reshape(-1, self.d)])
        out_grad = _scatter_add(out_ids, out_g, self.d)

        return loss, in_grad, out_grad


# ── Optimizers ───────────────────────────────────────────────────────────

class SGD:
    """SGD with linearly decaying learning rate."""

    def __init__(self, lr_start=0.025, lr_end=1e-4, total_steps=1):
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.total_steps = total_steps
        self.lr = lr_start

    def set_step(self, step):
        progress = min(step / max(self.total_steps, 1), 1.0)
        self.lr = self.lr_start + (self.lr_end - self.lr_start) * progress

    def update(self, param, grad, indices):
        param[indices] -= self.lr * grad


class AdaGrad:
    """AdaGrad with per-row accumulators, only touching active rows."""

    def __init__(self, lr=0.05, eps=1e-8):
        self.lr = lr
        self.eps = eps
        self._acc = {}

    def update(self, param, grad, indices):
        key = id(param)
        if key not in self._acc:
            self._acc[key] = np.zeros_like(param)
        acc = self._acc[key]
        acc[indices] += grad ** 2
        param[indices] -= self.lr * grad / (np.sqrt(acc[indices]) + self.eps)
