# Gradient Derivation: Temperature-Scaled Skip-Gram with Negative Sampling

## Objective

Standard SGNS loss for one (center, context, negative_1...negative_k) example:

$$
L = -\log\sigma(\mathbf{v}_c \cdot \mathbf{u}_o) - \sum_{k=1}^{K} \log\sigma(-\mathbf{v}_c \cdot \mathbf{u}_k)
$$

We introduce a temperature parameter τ > 0 that scales the dot-product scores before the sigmoid:

$$
L = -\log\sigma\!\left(\frac{\mathbf{v}_c \cdot \mathbf{u}_o}{\tau}\right) - \sum_{k=1}^{K} \log\sigma\!\left(\frac{-\mathbf{v}_c \cdot \mathbf{u}_k}{\tau}\right)
$$

Setting τ = 1 corresponds standard SGNS.

## Notation

| Symbol  | Meaning |
|---------|---------|
| v_c     | Input (center) embedding, row of W_in |
| u_o     | Output (context) embedding, row of W_out |
| u_k     | Output embedding for k-th negative sample |
| σ(x)    | Sigmoid: 1/(1 + exp(−x)) |
| τ       | Temperature scalar |
| $s^+$   | Positive score: v_c · u_o / τ |
| $s_k^-$ | Negative score: v_c · u_k / τ |

## Gradients with respect to scores

From the loss:

$$
\frac{\partial L}{\partial s^+} = -(1 - \sigma(s^+)) = \sigma(s^+) - 1
$$

$$
\frac{\partial L}{\partial s_k^-} = \sigma(s_k^-)
$$

## Chain rule to parameters

Since $s^+ = v_c · u_o / τ$ and $s_k^- = v_c · u_k / τ$:

**Center embedding v_c** receives gradients from all terms:

$$
\frac{\partial L}{\partial \mathbf{v}_c} = \frac{1}{\tau}\left[(\sigma(s^+) - 1)\,\mathbf{u}_o + \sum_{k=1}^{K} \sigma(s_k^-)\,\mathbf{u}_k\right]
$$

**Positive context embedding u_o:**

$$
\frac{\partial L}{\partial \mathbf{u}_o} = \frac{1}{\tau}(\sigma(s^+) - 1)\,\mathbf{v}_c
$$

**Negative sample embedding u_k:**

$$
\frac{\partial L}{\partial \mathbf{u}_k} = \frac{1}{\tau}\,\sigma(s_k^-)\,\mathbf{v}_c
$$

## Effect of temperature

When τ < 1, gradients are scaled up by 1/τ > 1, sharpening the contrastive signal. When τ > 1, gradients are damped. 

 The temperature plays the role of controlling the sharpness of the similarity distribution.

## Interaction with AdaGrad

AdaGrad accumulates squared gradients per parameter and scales updates by 1/√(accumulated). Since gradients scale as 1/τ, the accumulator grows as 1/τ². The effective AdaGrad update scales as:

$$
\Delta\theta \propto \frac{g}{\sqrt{\sum g^2}} \propto \frac{1/\tau}{\sqrt{\sum 1/\tau^2}} = \frac{1/\tau}{1/\tau \cdot \sqrt{n}} = \frac{1}{\sqrt{n}}
$$

In the idealized case, AdaGrad completely absorbs the temperature scaling. In practice, the absorption is partial because gradients vary across steps and parameters. This means temperature has a weaker effect under AdaGrad than under SGD.

## Batching

For a batch of B examples, we compute the sum of per-example gradients (not the mean). The learning rate is calibrated for per-example updates following the original word2vec implementation. The returned loss is the mean over the batch for stable logging.
