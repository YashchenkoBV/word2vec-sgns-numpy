# word2vec-numpy

Skip-gram with negative sampling (SGNS) implemented in pure NumPy. Includes temperature-scaled contrastive loss and experiments comparing SGD vs AdaGrad across temperatures.

## How to run

```bash
pip install numpy

# Download corpus
python download_data.py

# Train (default: SGD, τ=1.0, dim=100, 5 epochs)
python train.py

# Evaluate
python evaluate.py --checkpoint checkpoints/epoch5.npz

# Run full experiment grid (2 optimizers; 3 temperatures)
python run_experiments.py
```

## Project structure

```
word2vec.py          # Data pipeline, model, optimizers
train.py             # Training loop with CLI
evaluate.py          # Nearest neighbors, Google analogy task
run_experiments.py   # Experiment grid runner
download_data.py     # Downloads text8 corpus
DERIVATION.md        # Gradient derivation for temperature-scaled SGNS
```

## Method

**Model.** Standard skip-gram with negative sampling. For each (center, context) pair, the loss is:

```
L = −log σ(v_c · u_o / τ) − Σ_k log σ(−v_c · u_k / τ)
```

where τ is a temperature parameter (τ = 1 corresponds to standard SGNS). See ```DERIVATION.md``` for the full gradient derivation.

**Data pipeline.** text8 corpus (~17M tokens), vocabulary filtered at min_count=5 (~71K words), frequent-word subsampling with $\mathbb{P}_{keep} = \sqrt{t/f}$, and a 10M-entry unigram table (freq^0.75) for O(1) negative sampling.

**Training.** Batched (B=512) with sum gradients and per-element gradient clipping. Dynamic context window sampled from Unif[1, W]. Two optimizers: SGD with linear LR decay (0.025 → 1e-4), and AdaGrad (lr=0.05).

**Evaluation.** Cosine nearest neighbors and the Google analogy task (3CosAdd method, 19,544 questions across 14 categories).

## Results

All runs: dim=100, window=5, K=5 negatives, 5 epochs, batch_size=512, text8.

| Optimizer | τ   | Semantic | Syntactic | Total  |
|-----------|-----|----------|-----------|--------|
| SGD       | 0.5 | 15.7%    | 15.5%     | 15.6%  |
| SGD       | 1.0 | 13.7%    | 15.4%     | 14.7%  |
| SGD       | 2.0 | 8.3%     | 13.4%     | 11.3%  |
| AdaGrad   | 0.5 | 17.8%    | 13.7%     | 15.4%  |
| AdaGrad   | 1.0 | 14.3%    | 12.4%     | 13.2%  |
| AdaGrad   | 2.0 | 10.8%    | 10.9%     | 10.8%  |

Sample nearest neighbors for SGD, τ=1.0, epoch 5:

| Query    | Top neighbors                                  |
|----------|------------------------------------------------|
| king     | pepin, lancastrian, pretender, thrones, reigned |
| computer | computing, hardware, computers, wozniak, pda    |
| dog      | dogs, hound, keeshond, breeds, hounds           |
| france   | belgium, nantes, netherlands, spain, alsace      |

## Discussion

**Temperature improves SGNS.** Lower temperature (τ < 1) consistently improves analogy accuracy across both optimizers. At τ = 0.5, both SGD and AdaGrad reach ~15.5%, compared to ~14% at τ = 1.0 and ~11% at τ = 2.0. The effect seems to be monotonic.

This is expected from the contrastive learning perspective. SGNS is a contrastive objective: each training example contrasts one positive pair against K negative pairs. Temperature controls the sharpness of this contrast: lower τ amplifies score differences, producing stronger gradients for misclassified pairs and forcing the model toward more explicit positive/negative separation.

**AdaGrad favors semantic quality.** AdaGrad achieves the highest semantic accuracy (17.8% at τ = 0.5) while SGD leads on syntactic analogies (15.5%). AdaGrad gives effectively larger updates to rare words (whose accumulators grow slower), and rare words tend to carry semantic rather than syntactic information, e.g. country names, domain-specific words. Syntactic patterns are dominated by frequent words that both optimizers handle similarly.

**AdaGrad partially absorbs temperature.** In theory, AdaGrad's per-parameter accumulator normalizes gradient magnitudes, which should reduce the effect of temperature scaling. In practice, the total accuracy spread across temperatures is similar for both optimizers (~4.5 %), so the absorption effect is limited at the implementation level.

**Absolute accuracy.** The numbers (~15%) are below open source word2vec baselines on text8 (~30% at dim=300). Two factors explain this: (1) batched training with B=512 makes weights more "stale" compared to the original per-pair updates, and (2) dim=100 vs the typical dim=300 reduces model capacity.

## Requirements

- Python 3.8+
- NumPy
