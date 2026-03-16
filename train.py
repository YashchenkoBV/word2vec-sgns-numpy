"""Training script for SGNS word2vec."""

import argparse
import time
import os
import numpy as np

from word2vec import (
    load_corpus, sample_negatives, SGNS, SGD, AdaGrad,
)


def generate_batches(corpus, window, neg_samples, unigram_table, batch_size, rng):
    """Yield (center, context, negatives) batches from the corpus.

    For each center position, a window size is sampled from U[1, window].
    Each (center, context) pair within that window becomes one training example.
    """
    N = len(corpus)
    centers, contexts = [], []

    for i in range(N):
        w = rng.integers(1, window + 1)
        lo = max(0, i - w)
        hi = min(N, i + w + 1)
        for j in range(lo, hi):
            if j == i:
                continue
            centers.append(corpus[i])
            contexts.append(corpus[j])

            if len(centers) == batch_size:
                c = np.array(centers, dtype=np.int32)
                o = np.array(contexts, dtype=np.int32)
                neg = sample_negatives(unigram_table, (batch_size, neg_samples), rng)
                centers, contexts = [], []
                yield c, o, neg

    # Drop the last incomplete batch — no padding needed.


def estimate_total_steps(corpus_len, window, batch_size):
    """Estimate steps per epoch.  E[pairs per position] = window + 1."""
    pairs_per_epoch = corpus_len * (window + 1)
    return pairs_per_epoch // batch_size


def main():
    p = argparse.ArgumentParser(description="Train word2vec SGNS")
    p.add_argument("--corpus", type=str, default="data/text8")
    p.add_argument("--dim", type=int, default=100)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--neg-samples", type=int, default=5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adagrad"])
    p.add_argument("--lr", type=float, default=None,
                   help="Learning rate (default: 0.025 for sgd, 0.05 for adagrad)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10_000)
    p.add_argument("--save-dir", type=str, default="checkpoints")
    p.add_argument("--loss-log", type=str, default=None,
                   help="Path to write step,loss CSV for plotting")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    # ── Data ──
    corpus, word2id, id2word, counts, unigram_table = load_corpus(
        args.corpus, seed=args.seed,
    )

    V = len(id2word)
    steps_per_epoch = estimate_total_steps(len(corpus), args.window, args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    print(f"Estimated steps/epoch: {steps_per_epoch:,}  total: {total_steps:,}")

    # ── Model ──
    model = SGNS(V, args.dim, tau=args.temperature, seed=args.seed)

    # ── Optimizer ──
    if args.optimizer == "sgd":
        lr = args.lr if args.lr is not None else 0.025
        opt = SGD(lr_start=lr, lr_end=1e-4, total_steps=total_steps)
    else:
        lr = args.lr if args.lr is not None else 0.05
        opt = AdaGrad(lr=lr)

    # ── Training ──
    os.makedirs(args.save_dir, exist_ok=True)
    global_step = 0
    smoothed_loss = None
    alpha = 0.01  # EMA smoothing factor

    loss_log_f = None
    if args.loss_log:
        os.makedirs(os.path.dirname(args.loss_log) or ".", exist_ok=True)
        loss_log_f = open(args.loss_log, "w")
        loss_log_f.write("step,loss\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        epoch_loss = 0.0
        epoch_steps = 0

        for center, context, negatives in generate_batches(
            corpus, args.window, args.neg_samples, unigram_table, args.batch_size, rng,
        ):
            if hasattr(opt, "set_step"):
                opt.set_step(global_step)

            loss, (in_ids, in_g), (out_ids, out_g) = model.train_step(
                center, context, negatives, grad_clip=args.grad_clip,
            )

            opt.update(model.W_in, in_g, in_ids)
            opt.update(model.W_out, out_g, out_ids)

            smoothed_loss = loss if smoothed_loss is None else (1 - alpha) * smoothed_loss + alpha * loss
            epoch_loss += loss
            epoch_steps += 1
            global_step += 1

            if global_step % args.log_every == 0:
                lr_now = opt.lr
                print(f"  step {global_step:>8,}  loss {smoothed_loss:.4f}  lr {lr_now:.6f}")
                if loss_log_f:
                    loss_log_f.write(f"{global_step},{smoothed_loss:.6f}\n")
                    loss_log_f.flush()

        dt = time.time() - t0
        avg = epoch_loss / max(epoch_steps, 1)
        print(f"Epoch {epoch}/{args.epochs}  avg_loss {avg:.4f}  "
              f"steps {epoch_steps:,}  time {dt:.0f}s")

        # Save checkpoint.
        path = os.path.join(args.save_dir, f"epoch{epoch}.npz")
        np.savez(path, W_in=model.W_in, W_out=model.W_out,
                 id2word=np.array(id2word, dtype=object))
        print(f"  saved {path}")

    if loss_log_f:
        loss_log_f.close()


if __name__ == "__main__":
    main()
