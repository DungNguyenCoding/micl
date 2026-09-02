# Algorithm notes

This repository intentionally separates three algorithms:

1. **FedAvg**: deterministic local SGD and sample-weighted parameter averaging.
2. **BBB**: Bayes by Backprop with diagonal Gaussian variational weights and the two-component zero-mean Gaussian scale-mixture prior from Blundell et al. (2015).
3. **FOLA**: the online diagonal Laplace method, prior iteration, and Gaussian-product aggregation from Liu et al. (2023/2024).

## Bayes by Backprop

For every Bayesian weight tensor, `bayesian-torch` supplies trainable `mu` and `rho`, with

```
sigma = softplus(rho)
w = mu + sigma * epsilon, epsilon ~ N(0, I)
```

The repository does **not** use `bayesian_torch.models.dnn_to_bnn.get_kl_loss` for BBB because the requested prior is a Gaussian scale mixture, not a single Gaussian. Instead it reconstructs the exact last sampled weight from the layer's epsilon buffer and evaluates

```
log q(w | mu, sigma) - log[pi N(w;0,sigma1^2) + (1-pi) N(w;0,sigma2^2)]
```

with `logaddexp` for stability.

The paper's equal-minibatch scheme is preserved through `beta_i = 1/M`. The project-specific configuration additionally applies `kl_weight = 1/d` when `kl_weight: null`. For CIFAR-10, `d=851,514`, so this resolves to approximately `1.17438e-6`. This `1/d` factor is a project normalization layered on top of BBB; it is not the equal-minibatch rule itself.

### Federated aggregation caveat

The original BBB paper is not a federated-learning paper and does not define a server aggregation rule. The default `bbb.aggregation: gaussian_product` is therefore a **project extension**: each client's diagonal Gaussian variational posterior is combined with the same sample-ratio-tempered Gaussian-product equation used in the FOLA paper. `bbb.aggregation: fedavg_variational` is included as an ablation that directly averages all variational parameters.

## Federated Online Laplace Approximation

The client objective is

```
L = L_task + lambda * 0.5 * sum_j precision_global[j] * (theta[j]-mu_global[j])^2
```

and the diagonal task curvature is accumulated online from squared task gradients at every optimization step. The local precision is then

```
precision_local = (1/r) * fisher_local + ((r-1)/r) * precision_global
```

for one-based round `r`. The implementation uses squared **minibatch** task gradients, matching the operational structure of Algorithm 1 and avoiding the prohibitive per-example Jacobian materialization of a large ResNet.

Server aggregation is the diagonal form of the paper's multivariate Gaussian product:

```
precision_global = sum_k alpha_k * precision_k
mu_global = sum_k alpha_k * precision_k * mu_k / precision_global
alpha_k = |D_k| / sum_j |D_j|
```

## Variance floor

The requested stabilization rule is implemented for both Bayesian methods but is not presented as part of either paper:

```
sigma_local >= variance_floor_ratio * sigma_global
```

For FOLA this is equivalent to

```
precision_local <= precision_global / variance_floor_ratio^2
```

The fraction of affected parameters is logged each round.

## CIFAR-10 model dimension

The CIFAR model is a 6n+2 ResNet-56 (`n=9`) with GroupNorm-8. All convolution weights plus the final linear weight and bias are Bayesian under BBB; GroupNorm affine parameters remain deterministic. With 1x1 projection shortcuts at the two stage transitions, the Bayesian random-variable count is exactly **851,514**. A unit test enforces this.

## Source-vs-project choices

The online-Laplace paper uses an MLP for MNIST and a smaller CNN for CIFAR-10. This repository follows the user's requested `ResNet56 + GroupNorm-8`, 100 clients, sparse CIFAR partition, disabled augmentation, and cosine LR schedule instead. Therefore the implementation is algorithm-derived from the paper but is not a reproduction of its CIFAR experiment table.
