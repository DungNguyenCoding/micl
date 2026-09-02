# BayesFL Base

A research-oriented Flower/Ray package supporting six experiment modes:

- FedAvg / MNIST
- FedAvg / CIFAR-10
- Bayes by Backprop (BBB) / MNIST
- Bayes by Backprop (BBB) / CIFAR-10
- Federated Online Laplace Approximation (FOLA) / MNIST
- Federated Online Laplace Approximation (FOLA) / CIFAR-10

The implementation is organized as separate client, server, model, data, training, posterior, metric, and plotting modules. See `docs/ALGORITHM_NOTES.md` for the exact mathematical contract and the places where the requested experiment design extends the two source papers.

## Environment

The package is targeted at the requested environment:

```text
Python       3.10.20
Linux        x86_64
PyTorch      2.7.0+cu128
CUDA         12.8
GPU          NVIDIA GeForce RTX 5090
```

Flower is pinned to `1.30.0` because this package intentionally remains compatible with Python 3.10. `bayesian-torch==0.5.0` supplies the reparameterized Bayesian Conv/Linear layers. Your working CUDA PyTorch build is intentionally **not** installed or replaced by `requirements.txt`.

## Installation

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
bash scripts/check_environment.sh
python -m pytest -q
# Or run the complete install/model validation:
bash scripts/validate_install.sh
```

If your environment already has `pytest`, the first command only needs the Flower, Ray, Bayesian-Torch, YAML, NumPy, and Matplotlib dependencies.

## Five-round smoke tests first

Run one mode in the foreground:

```bash
python -m bayesfl.main --config scripts/configs/smoke_bbb_mnist.yaml
python -m bayesfl.main --config scripts/configs/smoke_bbb_cifar10.yaml
```

Run all six smoke configurations sequentially:

```bash
bash scripts/run_smoke_all.sh
```

The smoke configs use 4/4 clients, 1 local epoch, and 5 rounds. They are execution tests, not accuracy benchmarks.

## Standard runs

The standard scripts use `nohup`, save the shell log in `logs/`, and return the PID immediately:

```bash
bash scripts/run_fedavg_mnist.sh
bash scripts/run_fedavg_cifar10.sh
bash scripts/run_bbb_mnist.sh
bash scripts/run_bbb_cifar10.sh
bash scripts/run_fola_mnist.sh
bash scripts/run_fola_cifar10.sh
```

Standard configs use at most 100 rounds. Explicit 300-round CIFAR configurations are also included to preserve the original full-run requirement:

```bash
bash scripts/run_fedavg_cifar10_full300.sh
bash scripts/run_bbb_cifar10_full300.sh
bash scripts/run_fola_cifar10_full300.sh
```

You can override round count without editing YAML:

```bash
python -m bayesfl.main --config scripts/configs/bbb_cifar10.yaml --rounds 5
```

You can also select a default config by CLI arguments:

```bash
python -m bayesfl.main --dataset mnist --method fola
```

## CIFAR-10 specification

The standard CIFAR configs use:

```text
clients                  100
participation            100/100
partition                sparse_dirichlet
Dirichlet alpha          0.1
active classes/client    4
Poisson mean             100
target total samples     10046
augmentation             false
normalization            (x - 0.5) / 0.5
model                    ResNet-56, GroupNorm-8
optimizer                SGD
base LR                  0.05
momentum                 0.9
weight decay             0
batch size               128
local epochs             10
cosine horizon           400 rounds
LR minimum               0.0001
```

With seed 0, the partition generator intentionally matches the requested recorded summary statistics: total 10,046 samples, mean 100.46, minimum 79, maximum 127, and exactly four represented classes per client under normal non-exhausted operation. Because the original historical client-index manifest was not supplied, this package cannot claim the same sample-by-sample partition as that prior run. It creates its own deterministic manifest, persists the exact generated indices and SHA-256 under `outputs/partitions/`, and reuses them thereafter.

The cosine schedule is one-based at the user interface and uses zero-based `r=round-1` internally:

```text
round 1     0.05000
round 50    0.04817
round 100   0.04280
round 150   0.03471
round 200   0.02515
round 250   0.01557
round 300   0.00744
round 400   0.00010
```

## MNIST specification

MNIST uses the 784-500-300-10 MLP. The partition uses all 60,000 training samples with class-wise Dirichlet heterogeneity (`alpha=0.3`) and a lognormal client-size bias (`sigma=0.5`), followed by a minimum-size backfill. This produces non-IID and unbalanced clients while preserving every training sample.

Standard MNIST starting hyperparameters:

```text
FedAvg LR    0.01
BBB LR       0.001
FOLA LR      0.01
batch        64
local epochs 5
rounds       100
```

## BBB configuration

The requested variational state is configured as:

```text
posterior_mu_init       0.0
posterior_rho_init      -3.0
sigma                   softplus(rho)
prior                   two-Gaussian zero-mean scale mixture
pi                      0.5
sigma1                  1.0
sigma2                  exp(-6)
kl_weight               null -> 1 / Bayesian dimension
kl scheme               equal minibatch
warmup                   disabled (20-round value is inert)
size scaling            enabled
MC train                2
MC eval                 5
variance floor ratio    0.5
```

`bayesian-torch` initializes `mu` and `rho` from small normal distributions centered on the requested initialization values; this package keeps that library behavior.

The CIFAR-10 BBB model asserts a Bayesian random-variable dimension of exactly `851,514`. The variational optimizer therefore holds `1,703,028` stochastic scalars (`mu` and `rho`) plus deterministic GroupNorm parameters.

The base LR is used for posterior means and deterministic parameters. `rho_lr_multiplier=0.1` is a configurable stability factor for posterior-scale parameters; set it to `1.0` for the same LR on all variational parameters.

## FOLA configuration

FOLA uses deterministic network weights as posterior means and maintains one diagonal precision tensor for every model parameter. Local training applies the global Gaussian as an anisotropic prior and accumulates task-gradient squares online. Server aggregation is precision-weighted Gaussian-product aggregation.

The default `prior_lambda` is `1.0` because the source paper treats lambda as a tunable balance factor rather than giving one universal value. It is exposed in every YAML config.

## Metrics and outputs

Every run creates:

```text
logs/<run>.log
outputs/<run>/
  resolved_config.yaml
  source_config.yaml
  environment.json
  partition_metadata.json
  metrics/
    global_metrics.csv
    client_metrics.csv
    round_train_metrics.csv
  posterior/
    posterior_summary.csv
  reliability/
    round_XXXX.npz
  checkpoints/
    global_round_XXXX.npz
  plots/
```

Central evaluation records:

- accuracy
- predictive NLL
- Brier score
- ECE and MCE
- mean confidence
- predictive entropy
- expected entropy
- mutual information

BBB posterior summaries include mean magnitude, posterior sigma, and signal-to-noise ratio. FOLA summaries include posterior mean magnitude, sigma, and precision statistics. Per-client rows also contain compact posterior summaries, update norms, local losses, and variance-floor activity. Full global posterior checkpoints are written every `checkpoint_every` rounds instead of saving every client's full posterior on every round.

## Plot generation

All Matplotlib code is isolated in `src/bayesfl/utils.py`:

```bash
bash scripts/generate_plots.sh outputs/<run_directory>
```

This generates global accuracy/loss/ECE/Brier/MI curves, client training curves, posterior summaries, and a reliability diagram from the latest saved round.

## Ray/GPU resources

Flower's Simulation Runtime uses Ray. The default CIFAR config assigns `0.25` GPU per virtual client, allowing Ray to schedule up to about four GPU client jobs concurrently on one GPU resource; MNIST uses `0.125`. These are scheduling fractions, not hard VRAM limits. If the RTX 5090 is close to memory saturation, change:

```yaml
runtime:
  client_num_gpus: 0.5   # about two concurrent GPU clients
```

or use `1.0` for one GPU client at a time. The client releases model references and calls `torch.cuda.empty_cache()` after each Flower fit call.

## Important research note

BBB itself does not define federated posterior aggregation. The default BBB server rule in this package is explicitly a project extension using the FOLA paper's tempered Gaussian product. It is not presented as an equation from the BBB paper. The FOLA implementation, by contrast, directly follows the paper's prior-loss and Gaussian-product framework, with a practical minibatch-gradient diagonal Fisher approximation.
