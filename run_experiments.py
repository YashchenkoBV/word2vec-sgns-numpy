"""Run the full experiment grid: 2 optimizers × 3 temperatures.

Usage:
    python run_experiments.py

Results are saved to experiments/<run_name>/ with:
    - loss.csv       — step,loss for plotting
    - epoch5.npz     — final checkpoint
    - eval.json      — analogy accuracy breakdown
    - stdout.log     — full training + eval output
"""

import subprocess
import sys
import os
import time

EXPERIMENTS = [
    {"name": "sgd_tau0.5",  "optimizer": "sgd",     "lr": "0.025", "temperature": "0.5"},
    {"name": "sgd_tau1.0",  "optimizer": "sgd",     "lr": "0.025", "temperature": "1.0"},
    {"name": "sgd_tau2.0",  "optimizer": "sgd",     "lr": "0.025", "temperature": "2.0"},
    {"name": "ada_tau0.5",  "optimizer": "adagrad",  "lr": "0.05",  "temperature": "0.5"},
    {"name": "ada_tau1.0",  "optimizer": "adagrad",  "lr": "0.05",  "temperature": "1.0"},
    {"name": "ada_tau2.0",  "optimizer": "adagrad",  "lr": "0.05",  "temperature": "2.0"},
]

COMMON_ARGS = [
    "--corpus", "data/text8",
    "--dim", "100",
    "--window", "5",
    "--neg-samples", "5",
    "--epochs", "5",
    "--batch-size", "512",
    "--seed", "42",
    "--log-every", "5000",
]


def run_one(exp):
    name = exp["name"]
    exp_dir = os.path.join("experiments", name)
    os.makedirs(exp_dir, exist_ok=True)

    log_path = os.path.join(exp_dir, "stdout.log")
    loss_log = os.path.join(exp_dir, "loss.csv")
    save_dir = exp_dir
    ckpt_path = os.path.join(exp_dir, "epoch5.npz")
    eval_json = os.path.join(exp_dir, "eval.json")

    # ── Train ──
    train_cmd = [
        sys.executable, "train.py",
        *COMMON_ARGS,
        "--optimizer", exp["optimizer"],
        "--lr", exp["lr"],
        "--temperature", exp["temperature"],
        "--save-dir", save_dir,
        "--loss-log", loss_log,
    ]

    print(f"\n{'='*60}")
    print(f"  START: {name}  (optimizer={exp['optimizer']}, tau={exp['temperature']})")
    print(f"{'='*60}")
    t0 = time.time()

    with open(log_path, "w") as log_f:
        proc = subprocess.run(
            train_cmd, stdout=log_f, stderr=subprocess.STDOUT, text=True,
        )

    train_time = time.time() - t0

    if proc.returncode != 0:
        print(f"  FAILED (exit code {proc.returncode}). See {log_path}")
        return

    print(f"  Training done in {train_time/60:.1f} min")

    # ── Evaluate ──
    eval_cmd = [
        sys.executable, "evaluate.py",
        "--checkpoint", ckpt_path,
        "--output-json", eval_json,
    ]

    with open(log_path, "a") as log_f:
        log_f.write("\n\n--- EVALUATION ---\n\n")
        proc = subprocess.run(
            eval_cmd, stdout=log_f, stderr=subprocess.STDOUT, text=True,
        )

    if proc.returncode != 0:
        print(f"  Evaluation FAILED. See {log_path}")
        return

    # Print summary.
    import json
    with open(eval_json) as f:
        res = json.load(f)
    print(f"  Analogy accuracy: {res['total']:.1%} "
          f"(semantic {res['semantic']:.1%}, syntactic {res['syntactic']:.1%})")


def main():
    total_t0 = time.time()

    for exp in EXPERIMENTS:
        run_one(exp)

    total = (time.time() - total_t0) / 3600
    print(f"\n{'='*60}")
    print(f"  ALL DONE — total time: {total:.1f} hours")
    print(f"  Results in experiments/*/eval.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

