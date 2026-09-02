# bayesfl baseline v2.0

Unified Flower/Ray baseline for future federated Bayesian-learning work.

This branch intentionally removes all wireless/AirComp simulation and the old
two-phase precision/natural-mean protocol.  Each FL round contains exactly one
client fit call for both methods:

- `fedavg`: deterministic FedAvg
- `bayesavg`: one-phase Bayesian-Torch variational FL

`proposed` is accepted as a compatibility alias for `bayesavg`, but new logs
use the canonical name `bayesavg`.

## Dataset/model selection

```bash
python main.py --dataset mnist --method fedavg
python main.py --dataset mnist --method bayesavg

python main.py --dataset cifar10 --method fedavg
python main.py --dataset cifar10 --method bayesavg
```

Canonical model mapping:

- MNIST -> `PaperCNN`, 62,346 parameters
- CIFAR-10 -> `ResNet56-GN`, 855,770 parameters

CIFAR Bayesian-Torch scope:

- Bayesian Conv/Linear coordinates: 851,514
- deterministic GroupNorm affine coordinates: 4,256
- BayesAvg upload state: `mu + rho_BT + GroupNorm` = 1,707,284 scalars/client/round

## Bayesian variational configuration

```text
prior                     N(0, 1)
posterior_mu_init          0.0
posterior_rho_init         -3.0
kl_weight                  null -> 1 / Bayesian dimension
kl_weight_schedule         false
kl_warmup_rounds           20 (inert while schedule=false)
lambda_scale_by_size       true
mc_train                   2
mc_eval                    5
variance_floor_ratio       0.5
```

For CIFAR-10:

```text
d = 851,514
1/d = 1.1743788123272196e-6
```

The local loss is

```text
mean MC cross entropy
+ lambda_client * full coordinate-sum KL(q || N(0,I))
```

where

```text
lambda_client = (1/d) * |D_k| / realised_mean_client_size
```

when `lambda_scale_by_size=true`.

The global posterior initializes each client's local posterior.  The KL prior
remains the fixed standard normal every round.

## BayesAvg aggregation

The server sample-size-weighted averages Bayesian-Torch variational state
coordinates directly:

- posterior `mu`
- Bayesian-Torch unconstrained `rho`
- deterministic GroupNorm weight/bias

This is a transparent baseline aggregation rule, not the old natural-parameter
AirComp algorithm.

## Client participation

Default CIFAR setting is full participation:

```text
100 / 100 clients every round
```

Override from the CLI:

```bash
python main.py \
  --dataset cifar10 \
  --method bayesavg \
  --client-fraction 0.1
```

With 100 total clients, this selects 10 clients/round.

## CIFAR-10 baseline

```text
clients                    100
partition                  sparse_dirichlet
alpha                      0.1
average client size        100
support classes/client     4
augmentation               false

optimizer                  SGD
LR                         0.05
momentum                   0.9
weight decay               0
batch                      128
local epochs               10
round max                  300
cosine horizon             400
minimum LR                 0.0001
```

The fixed-horizon schedule is

```text
lr(r) = lr_min + 0.5*(lr-lr_min)*(1+cos(pi*min(r,H-1)/(H-1)))
r = server_round - 1
H = 400
```

Expected checkpoints:

```text
round   1 : 0.05000000
round  50 : 0.04816603
round 100 : 0.04279619
round 150 : 0.03471127
round 200 : 0.02514822
round 250 : 0.01557015
round 300 : 0.00744245
round 400 : 0.00010000
```

For seed 0, the sparse-Dirichlet partition is regression-tested to record:

```text
total_samples_used                10046
mean_size                         100.46
min_size                          79
max_size                          127
mean_classes_per_client           4.0
empty_client_backfills            0
num_empty_clients_after_backfill  0
class_draws_exhausted             0
```

Each client selects four random distinct class labels; alpha=0.1 controls the
within-client mixture on those labels. Every selected class receives at least
one example. Dataset indices are unique across clients.

## MNIST baseline

The local/data/model profile preserves the old v1.6.1 settings that remain
meaningful after removing wireless communication:

```text
model                      PaperCNN, d=62,346
clients                    40
labels/client              1
Poisson mean samples       10
optimizer                  SGD
LR                         0.1
momentum                   0
batch                      10
local epochs               3
LR schedule                constant
```

The default common round count is 240 because the old method-specific wireless
channel-use budget no longer exists. `--rounds` can override it.

## Validation

```bash
unset AIRCOMP_DATASET
python -m pytest -q
```

CIFAR dry run:

```bash
python main.py \
  --dataset cifar10 \
  --methods fedavg,bayesavg \
  --seed 0 \
  --dry-run
```

MNIST dry run:

```bash
python main.py \
  --dataset mnist \
  --methods fedavg,bayesavg \
  --seed 0 \
  --dry-run
```

The source has no Pyro, AirComp, wireless, sparse-posterior, or two-phase
Bayesian dependency.
