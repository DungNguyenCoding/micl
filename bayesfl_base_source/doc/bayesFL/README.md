# Bayesian FL Mathematical Notes and Source Mapping

This document explains the mathematical meaning of the implemented methods and maps each important formula to the source code. The line numbers below refer to the current validated v1 source package. If code is edited later, search by the listed function names.

## Table of contents

1. [Notation](#1-notation)
2. [FedAvg](#2-fedavg)
   - [2.1. Mathematical formulation](#21-mathematical-formulation)
   - [2.2. Source-code mapping](#22-source-code-mapping)
3. [Variational Inference Bayesian FL](#3-variational-inference-bayesian-fl)
   - [3.1. Local posterior](#31-local-posterior)
   - [3.2. Local VI objective](#32-local-vi-objective)
   - [3.3. Posterior aggregation](#33-posterior-aggregation)
   - [3.4. Stabilization](#34-stabilization)
   - [3.5. Source-code mapping](#35-source-code-mapping)
4. [Online Laplace Approximation / FOLA](#4-online-laplace-approximation--fola)
   - [4.1. Local objective](#41-local-objective)
   - [4.2. Fisher and precision update](#42-fisher-and-precision-update)
   - [4.3. Posterior aggregation](#43-posterior-aggregation)
   - [4.4. Source-code mapping](#44-source-code-mapping)
5. [Sparse Bayesian communication](#5-sparse-bayesian-communication)
   - [5.1. Importance scores](#51-importance-scores)
   - [5.2. Top-k sparse transmission](#52-top-k-sparse-transmission)
   - [5.3. Sparse Bayesian aggregation](#53-sparse-bayesian-aggregation)
   - [5.4. Communication cost approximation](#54-communication-cost-approximation)
   - [5.5. Source-code mapping](#55-source-code-mapping)
6. [Posterior evaluation](#6-posterior-evaluation)
7. [Method comparison interpretation](#7-method-comparison-interpretation)

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| $K$ | Number of selected physical devices in one communication round. |
| $n_k$ | Number of training examples on client/device $k$. |
| $N_t = \sum_{k \in S_t} n_k$ | Total examples from selected devices in round $t$. |
| $\theta$ | Deterministic neural network parameter vector. |
| $\mu$ | Posterior mean / deterministic model mean. |
| $\sigma$ | Posterior standard deviation. |
| $\Lambda$ | Posterior precision, usually $\Lambda = 1 / \sigma^2$. |
| $q_k(w)$ | Local client posterior approximation. |
| $q_G(w)$ | Server/global posterior approximation. |
| $S_t$ | Selected physical-device set in round $t$. |

The implementation stores all model parameters as flat NumPy arrays for Flower communication. Shape metadata is stored separately in posterior snapshots.

---

## 2. FedAvg

### 2.1. Mathematical formulation

In FedAvg, the server sends the current global model $\theta_t$ to selected clients. Each client performs local deterministic optimization and returns a local model $\theta_{t+1}^{(k)}$.

The server aggregates by example-weighted averaging:

$$
\theta_{t+1}
= \sum_{k \in S_t} \frac{n_k}{N_t}\,\theta_{t+1}^{(k)}.
$$

Local deterministic training minimizes the cross-entropy objective:

$$
\mathcal{L}_{\text{task}}(\theta)
= \frac{1}{|D_k|}\sum_{(x,y)\in D_k}\operatorname{CE}(f_\theta(x), y).
$$

### 2.2. Source-code mapping

| Formula / concept | Source mapping |
|---|---|
| Initial FedAvg payload `[theta]` | `main.py`, `initial_payload()`, lines 48-59. |
| Client local deterministic training | `client.py`, `_fit_fedavg()`, lines 185-228. |
| Cross-entropy task loss | `model.py`, `train_deterministic()`, lines 209-215. |
| Local optimizer step | `model.py`, lines 231-234. |
| Client weighted model contribution $n_k\theta_k$ | `client.py`, lines 203-206. |
| Server FedAvg aggregation $\sum n_k\theta_k/N_t$ | `strategy.py`, `_aggregate_fedavg()`, lines 409-417. |

---

## 3. Variational Inference Bayesian FL

### 3.1. Local posterior

The VI method represents each client's local posterior with a diagonal Gaussian:

$$
q_k(w) = \mathcal{N}(w; \mu_k, \operatorname{diag}(\sigma_k^2)).
$$

The incoming global posterior is also diagonal Gaussian:

$$
q_G(w) = \mathcal{N}(w; \mu_G, \operatorname{diag}(\sigma_G^2)).
$$

In the source, the VI Flower payload is:

```text
[global_loc, global_scale]
```

where `global_loc` is $\mu_G$ and `global_scale` is $\sigma_G$.

### 3.2. Local VI objective

The local VI objective is a Bayes-by-Backprop-style variational objective:

$$
\mathcal{F}_k(q)
= \mathbb{E}_{q_k(w)}[-\log p(D_k \mid w)]
+ \operatorname{KL}(q_k(w)\,\|\,q_G(w)).
$$

Pyro SVI minimizes a stochastic ELBO/free-energy proxy. The implementation also logs a closed-form diagonal Gaussian KL:

$$
\operatorname{KL}(q_k\|q_G)
= \frac{1}{2}\sum_i
\left[
\frac{\sigma_{k,i}^2 + (\mu_{k,i}-\mu_{G,i})^2}{\sigma_{G,i}^2}
- 1
+ 2\log \frac{\sigma_{G,i}}{\sigma_{k,i}}
\right].
$$

### 3.3. Posterior aggregation

The main VI aggregation mode is precision-product aggregation. For each coordinate $i$:

$$
\Lambda_{G,i}^{t+1}
= \frac{1}{N_t}\sum_{k\in S_t} n_k\Lambda_{k,i},
$$

$$
\mu_{G,i}^{t+1}
= \frac{\sum_{k\in S_t} n_k\Lambda_{k,i}\mu_{k,i}}{\sum_{k\in S_t} n_k\Lambda_{k,i}}.
$$

The resulting standard deviation is:

$$
\sigma_{G,i}^{t+1}=\sqrt{\frac{1}{\Lambda_{G,i}^{t+1}}}.
$$

The optional moment-matching mode computes:

$$
\mu_G = \frac{1}{N_t}\sum_k n_k\mu_k,
$$

$$
\operatorname{Var}_G = \frac{1}{N_t}\sum_k n_k(\sigma_k^2 + \mu_k^2) - \mu_G^2.
$$

### 3.4. Stabilization

Long non-IID VI runs can suffer posterior drift. The current code supports two stabilization mechanisms.

Round-based learning-rate decay:

$$
\eta_t = \eta_0 \gamma^{m(t)},
$$

where $m(t)$ is the number of configured milestones reached by round $t$.

Posterior-scale clipping:

$$
\sigma_{k,i} \leftarrow \min(\sigma_{k,i}, \sigma_{\max}).
$$

### 3.5. Source-code mapping

| Formula / concept | Source mapping |
|---|---|
| VI initial payload `[mu, scale]` | `main.py`, `initial_payload()`, lines 56-58. |
| Pyro Gaussian weight prior | `bayes_vi.py`, `_make_pyro_model()`, lines 96-117. |
| Pyro SVI and ELBO setup | `bayes_vi.py`, `train_vi_local()`, lines 181-186. |
| Local SVI training loop | `bayes_vi.py`, lines 188-195. |
| Extract posterior `loc` and `scale` | `bayes_vi.py`, lines 197-210. |
| Scale upper clipping | `bayes_vi.py`, lines 211-212. |
| Closed-form diagonal Gaussian KL | `bayes_vi.py`, lines 214-222. |
| Likelihood/NLL metric | `bayes_vi.py`, line 223. |
| VI metric dictionary | `bayes_vi.py`, lines 225-242. |
| Client calls `train_vi_local()` | `client.py`, `_fit_vi()`, lines 391-404. |
| VI precision $1/\sigma^2$ | `client.py`, line 407. |
| Dense product contribution $n\Lambda$ and $n\Lambda\mu$ | `client.py`, lines 408-410. |
| Server precision-product aggregation | `strategy.py`, `_aggregate_product_precision()`, lines 419-496. |
| Optional moment matching | `strategy.py`, `_aggregate_moment_match()`, lines 498-511. |
| LR decay config | `config.py`, lines 100-106 and CLI lines 207-209. |
| Effective VI LR computation | `client.py`, `_effective_vi_lr()`, lines 159-171. |

---

## 4. Online Laplace Approximation / FOLA

### 4.1. Local objective

OLA/FOLA uses deterministic local training with a quadratic prior-iteration penalty around the current global posterior mean:

$$
\mathcal{L}_k(\theta)
= \mathcal{L}_{\text{task}}(\theta; D_k)
+ \lambda_{\text{OLA}}\,\frac{1}{P}\frac{1}{2}\sum_{i=1}^{P}
\Lambda_{G,i}(\theta_i - \mu_{G,i})^2.
$$

The normalized prior loss is used in the optimized objective. The raw prior loss is also logged for interpretation.

### 4.2. Fisher and precision update

The diagonal Fisher approximation is collected from squared task-loss gradients:

$$
F_{k,i} \approx \mathbb{E}_{(x,y)\in D_k}
\left[\left(\frac{\partial \mathcal{L}_{\text{task}}}{\partial \theta_i}\right)^2\right].
$$

The online local precision update is:

$$
\Lambda_{k,i}^{t}
= \frac{1}{t}F_{k,i}^{t}
+ \frac{t-1}{t}\Lambda_{G,i}^{t-1}
+ \frac{1}{t}\gamma_i,
$$

where $\gamma_i$ is initialized by `precision_init`.

The local standard deviation is:

$$
\sigma_{k,i}=\sqrt{1/\Lambda_{k,i}}.
$$

### 4.3. Posterior aggregation

OLA uses the same precision-weighted product-style server aggregation as VI, but the payload is `[mu, precision]` rather than `[mu, scale]`:

$$
\Lambda_{G,i}^{t+1}
= \frac{1}{N_t}\sum_{k\in S_t} n_k\Lambda_{k,i},
$$

$$
\mu_{G,i}^{t+1}
= \frac{\sum_{k\in S_t} n_k\Lambda_{k,i}\mu_{k,i}}{\sum_{k\in S_t} n_k\Lambda_{k,i}}.
$$

### 4.4. Source-code mapping

| Formula / concept | Source mapping |
|---|---|
| OLA initial payload `[mu, precision]` | `main.py`, `initial_payload()`, lines 53-55. |
| Task cross-entropy loss | `model.py`, `train_deterministic()`, lines 213-215. |
| Fisher diagonal from squared gradients | `model.py`, lines 216-222 and 249-254. |
| Prior loss $0.5\sum \Lambda(\theta-\mu)^2$ | `model.py`, lines 224-230. |
| Optimized OLA objective | `model.py`, line 231. |
| OLA client training call with prior and Fisher | `client.py`, `_fit_ola()`, lines 253-263. |
| OLA precision update | `client.py`, line 268. |
| Local sigma and SNR | `client.py`, lines 270-271. |
| Dense precision payload $n\Lambda$ and $n\Lambda\mu$ | `client.py`, lines 275-277. |
| Server precision-product aggregation | `strategy.py`, `_aggregate_product_precision()`, lines 419-496. |
| OLA prior/task/Fisher metrics | `client.py`, lines 313-328 and `observability.py`, lines 183-188. |

---

## 5. Sparse Bayesian communication

Sparse Bayesian communication sends only the most informative coordinates of the local posterior contribution.

### 5.1. Importance scores

The implementation supports four coordinate scores.

Bayes-by-Backprop-style weight SNR:

$$
\operatorname{SNR}_i = \frac{|\mu_i|}{\sigma_i + \epsilon}.
$$

Federated update SNR:

$$
\operatorname{UpdateSNR}_i = \frac{|\mu_{k,i} - \mu_{G,i}|}{\sigma_{k,i} + \epsilon}.
$$

OLA/FOLA precision-update score:

$$
\operatorname{PrecisionUpdate}_i = |\mu_{k,i} - \mu_{G,i}|\,\Lambda_{k,i}.
$$

Per-coordinate diagonal Gaussian KL:

$$
\operatorname{KL}_i(q_k\|q_G)
= \frac{1}{2}
\left[
\frac{\sigma_{k,i}^2 + (\mu_{k,i}-\mu_{G,i})^2}{\sigma_{G,i}^2}
-1
+2\log\frac{\sigma_{G,i}}{\sigma_{k,i}}
\right].
$$

### 5.2. Top-k sparse transmission

For a configured keep ratio $r$, the client keeps the top:

$$
k = \lceil rP \rceil
$$

coordinates by score. It sends:

```text
indices
first_values
second_values
count_values
```

where `first_values` and `second_values` are method-specific Bayesian aggregation contributions.

### 5.3. Sparse Bayesian aggregation

For sent coordinates, the server uses the same precision-weighted aggregation formula. For unsent coordinates, missing values mean:

```text
no new posterior evidence
```

Therefore, the server keeps the previous global posterior for those coordinates instead of setting them to zero.

### 5.4. Communication cost approximation

The dense Bayesian payload assumes two float32 arrays per selected physical client:

$$
B_{\text{dense}} = 2P \times 4 \text{ bytes}.
$$

The sparse payload assumes one int64 index and three float32 values per sent coordinate:

$$
B_{\text{sparse}} = k(8 + 3\times4) \text{ bytes}.
$$

The saving ratio is:

$$
1 - \frac{B_{\text{sparse}}}{B_{\text{dense}}}.
$$

### 5.5. Source-code mapping

| Formula / concept | Source mapping |
|---|---|
| Sparse CLI options | `config.py`, `RunConfig`, lines 112-118 and CLI lines 214-219. |
| Weight SNR | `compression.py`, `weight_snr()`, lines 55-57. |
| Update SNR | `compression.py`, `update_snr()`, lines 60-64. |
| Precision-update score | `compression.py`, `precision_update_score()`, lines 67-71. |
| KL score | `compression.py`, `diag_gaussian_kl_score()`, lines 74-86. |
| Metric dispatch | `compression.py`, `score_for_sparse_metric()`, lines 89-115. |
| Top-k mask | `compression.py`, `topk_mask()`, lines 118-142. |
| Sparse payload packing | `compression.py`, `pack_sparse_contribution()`, lines 145-174. |
| Per-client sparse metrics | `compression.py`, `sparse_row_metrics()`, lines 197-248. |
| Sparse VI client path | `client.py`, `_fit_vi()`, lines 417-441. |
| Sparse OLA client path | `client.py`, `_fit_ola()`, lines 279-303. |
| Sparse server aggregation rule | `strategy.py`, `_aggregate_product_precision()`, lines 425-496. |
| Keep previous global posterior for unsent coordinates | `strategy.py`, lines 433-435 and 469-481. |
| Communication byte estimates | `strategy.py`, lines 345-360. |

---

## 6. Posterior evaluation

Bayesian methods are evaluated in two ways.

Posterior-mean deterministic evaluation:

$$
\theta = \mu.
$$

Monte Carlo posterior-predictive evaluation:

$$
\hat{p}(y\mid x)
= \frac{1}{M}\sum_{m=1}^{M}p(y\mid x,w^{(m)}),
\quad
w^{(m)} = \mu + \alpha \sigma \odot \epsilon^{(m)},
$$

where $\alpha$ is `posterior_sample_scale`.

Source mapping:

| Concept | Source mapping |
|---|---|
| Mean evaluation | `strategy.py`, `evaluate()`, lines 521-535. |
| MC evaluation | `strategy.py`, lines 537-555. |
| Stable global metric aliases use mean evaluation | `strategy.py`, lines 557-559 and 578-617. |
| Posterior MC metric fields | `strategy.py`, lines 598-614. |
| Evaluation implementation | `observability.py`, `evaluate_payload()`, lines 580-760. |

---

## 7. Method comparison interpretation

The current implementation supports two complementary comparison types.

| Comparison type | Meaning |
|---|---|
| Dense FedAvg vs dense VI vs dense OLA | Baseline learning behavior under the same dataset/client selection setting. |
| Dense vs sparse Bayesian method | Communication-efficiency and Bayesian filtering effect. |
| Sparse VI + LR decay + scale clamp | Stability-enhanced VI for long non-IID training. |
| Posterior mean vs MC evaluation | Separates deterministic model quality from posterior uncertainty calibration. |
| Best-vs-final comparison | Detects late-round degradation, especially dense VI posterior drift. |

Important limitation: current validated evidence is single-seed MNIST non-IID unbalanced. Multi-seed and CIFAR-10 validation are needed before publication-level claims.
