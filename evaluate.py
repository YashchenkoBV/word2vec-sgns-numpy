"""Evaluation script for word2vec embeddings."""

import argparse
import json
import os
import urllib.request
import numpy as np


def load_embeddings(path, use_both=False):
    """Load checkpoint.  Returns (embedding_matrix, id2word, word2id)."""
    ckpt = np.load(path, allow_pickle=True)
    W = ckpt["W_in"]
    if use_both:
        W = (W + ckpt["W_out"]) / 2
    id2word = list(ckpt["id2word"])
    word2id = {w: i for i, w in enumerate(id2word)}
    return W, id2word, word2id


def normalize(W):
    """L2-normalize rows in place, return the matrix."""
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    W /= norms
    return W


# ── Nearest neighbors ───────────────────────────────────────────────────

def nearest_neighbors(W_norm, word2id, id2word, query, k=10):
    """Return k nearest neighbors by cosine similarity."""
    if query not in word2id:
        return None
    vec = W_norm[word2id[query]]
    sims = W_norm @ vec
    # Exclude the query word itself.
    sims[word2id[query]] = -1.0
    top = np.argsort(sims)[-k:][::-1]
    return [(id2word[i], sims[i]) for i in top]


def print_neighbors(W_norm, word2id, id2word, words, k=10):
    """Print nearest neighbors for a list of query words."""
    for w in words:
        result = nearest_neighbors(W_norm, word2id, id2word, w, k)
        if result is None:
            print(f"  {w}: not in vocabulary")
        else:
            neighbors = ", ".join(f"{word} ({sim:.3f})" for word, sim in result)
            print(f"  {w}: {neighbors}")


# ── Google analogy task ──────────────────────────────────────────────────

ANALOGY_URL = "https://raw.githubusercontent.com/tmikolov/word2vec/master/questions-words.txt"


def download_analogies(path="data/questions-words.txt"):
    """Download the Google analogy dataset if not present."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading analogy dataset to {path} ...")
    urllib.request.urlretrieve(ANALOGY_URL, path)
    return path


def load_analogies(path, word2id):
    """Parse the analogy file.  Returns dict: category -> list of (a, b, c, d) tuples.

    Only includes questions where all four words are in the vocabulary.
    """
    categories = {}
    current = None
    total, covered = 0, 0

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(":"):
                current = line[2:]
                continue
            parts = line.lower().split()
            if len(parts) != 4:
                continue
            total += 1
            a, b, c, d = parts
            if all(w in word2id for w in (a, b, c, d)):
                categories.setdefault(current, []).append((a, b, c, d))
                covered += 1

    print(f"Analogies: {covered}/{total} covered by vocabulary")
    return categories


def eval_analogies(W_norm, word2id, categories):
    """Evaluate analogy accuracy: a is to b as c is to ?  (should be d).

    Uses the 3CosAdd method: argmax cos(x, b - a + c), excluding a, b, c.
    Returns dict: category -> (correct, total, accuracy).
    """
    results = {}

    for cat, quads in categories.items():
        correct = 0
        for a, b, c, d in quads:
            ia, ib, ic, idx_d = word2id[a], word2id[b], word2id[c], word2id[d]
            query = W_norm[ib] - W_norm[ia] + W_norm[ic]
            sims = W_norm @ query
            # Exclude input words.
            sims[ia] = -1.0
            sims[ib] = -1.0
            sims[ic] = -1.0
            if np.argmax(sims) == idx_d:
                correct += 1
        total = len(quads)
        results[cat] = (correct, total, correct / total if total > 0 else 0.0)

    return results


def print_analogy_results(results):
    """Print per-category and aggregate analogy results.  Returns summary dict."""
    semantic_correct, semantic_total = 0, 0
    syntactic_correct, syntactic_total = 0, 0

    # Categories starting with "gram" are syntactic; the rest are semantic.
    for cat, (c, t, acc) in sorted(results.items()):
        print(f"  {cat:40s}  {c:>5}/{t:<5}  {acc:.1%}")
        if cat.startswith("gram"):
            syntactic_correct += c
            syntactic_total += t
        else:
            semantic_correct += c
            semantic_total += t

    total_c = semantic_correct + syntactic_correct
    total_t = semantic_total + syntactic_total

    print(f"\n  {'Semantic':40s}  {semantic_correct:>5}/{semantic_total:<5}  "
          f"{semantic_correct / max(semantic_total, 1):.1%}")
    print(f"  {'Syntactic':40s}  {syntactic_correct:>5}/{syntactic_total:<5}  "
          f"{syntactic_correct / max(syntactic_total, 1):.1%}")
    print(f"  {'Total':40s}  {total_c:>5}/{total_t:<5}  "
          f"{total_c / max(total_t, 1):.1%}")

    return {
        "semantic": semantic_correct / max(semantic_total, 1),
        "syntactic": syntactic_correct / max(syntactic_total, 1),
        "total": total_c / max(total_t, 1),
        "per_category": {cat: acc for cat, (_, _, acc) in results.items()},
    }


def main():
    p = argparse.ArgumentParser(description="Evaluate word2vec embeddings")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--use-both", action="store_true",
                   help="Average W_in and W_out instead of using W_in only")
    p.add_argument("--analogy-path", type=str, default="data/questions-words.txt")
    p.add_argument("--neighbors", type=str, nargs="*",
                   default=["king", "france", "computer", "good", "dog"])
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--output-json", type=str, default=None,
                   help="Path to write evaluation results as JSON")
    args = p.parse_args()

    W, id2word, word2id = load_embeddings(args.checkpoint, use_both=args.use_both)
    W_norm = normalize(W.copy())
    print(f"Loaded {W.shape[0]} embeddings, dim={W.shape[1]}")

    # ── Nearest neighbors ──
    print(f"\nNearest neighbors (k={args.top_k}):")
    print_neighbors(W_norm, word2id, id2word, args.neighbors, k=args.top_k)

    # ── Analogies ──
    analogy_path = download_analogies(args.analogy_path)
    categories = load_analogies(analogy_path, word2id)

    if not categories:
        print("No analogy questions covered by vocabulary.")
        return

    print("\nAnalogy accuracy:")
    results = eval_analogies(W_norm, word2id, categories)
    summary = print_analogy_results(results)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved results to {args.output_json}")


if __name__ == "__main__":
    main()
