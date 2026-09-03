# CIFAR-10 paper environment + ResNet-56 + preserved project settings

This profile intentionally combines the CIFAR-10 **data/federation/augmentation/FOLA semantics**
from *A Bayesian Federated Learning Framework with Online Laplace Approximation* with two
explicit project overrides requested by the user:

1. keep the existing **ResNet-56 + GroupNorm-8** model for FedAvg, FOLA, and BBB;
2. keep the existing local-training, LR-schedule, and BBB variational/prior settings.

It is therefore **not an exact Table-II reproduction**. In particular, the paper used its
BasicCNN and selected batch/local-epoch/LR settings from a grid, while this profile fixes the
settings listed below.

## Paper-aligned CIFAR environment

- CIFAR-10 full training set (no old ~10k cap)
- 20 clients, 20/20 participation
- class-wise Dirichlet allocation with alpha=0.01
- no four-class-per-client cap
- RandomCrop(32, padding=4, fill=128)
- RandomHorizontalFlip
- CIFAR-10 AutoAugment
- Cutout(1,16)
- Normalize mean/std=(0.5,0.5,0.5)
- FOLA `paper_reference` mode: online task-gradient-square omega, prior iteration,
  omega-weighted Gaussian-product server aggregation, mean/MAP global evaluation

Torchvision's CIFAR-10 AutoAugment policy is used instead of vendoring the paper's helper.

## Explicitly retained model

All CIFAR methods use:

- `resnet56_gn8`
- ResNet-56
- GroupNorm with 8 groups
- stochastic Conv/Linear dimension for BBB: **851,514**

The optional `paper_basiccnn` implementation remains available in the source, but none of the
main/smoke/sweep CIFAR configs select it.

## Explicitly preserved local training

Shared by FedAvg / FOLA / BBB:

- SGD
- base lr=0.05
- momentum=0.9
- weight_decay=0
- batch_size=128
- E=10
- rounds=300
- cosine schedule
- lr_min=0.0001
- horizon H=400

Requested schedule values:

- R1   = 0.05000000
- R50  = 0.04816603
- R100 = 0.04279619
- R150 = 0.03471127
- R200 = 0.02514822
- R250 = 0.01557015
- R300 = 0.00744245
- R400 = 0.00010000 (horizon only)

## Explicitly preserved BBB settings

- prior = N(0,1)
- posterior_mu_init=0
- posterior_rho_init=-3
- kl_weight=null
- because ResNet-56 has d=851,514, null KL resolves naturally to `1/851514`
- kl_weight_schedule=false; kl_warmup_rounds=20 is inert
- lambda_scale_by_size=true
- mc_train=2
- mc_eval=5
- variance_floor_ratio=0.5
- Gaussian-product BBB aggregation

No `kl_reference_dimension` override is needed anymore: the requested denominator and the
actual ResNet-56 stochastic dimension are the same.

## FOLA lambda

The paper treats lambda as a tuned hyperparameter. The exact winning lambda for the user's
fixed E=10 / B=128 / lr=0.05 schedule is not specified, so the patch supplies a 20-round
sweep over:

`0, 1, 10, 100, 1000, 10000`

Run:

```bash
bash scripts/run_paper_fola_lambda_sweep.sh
```

The selector creates `scripts/configs/fola_cifar10_selected.yaml` with the best R20 mean
accuracy and promotes it to 300 rounds. Then run:

```bash
bash scripts/run_fola_cifar10.sh
```

## Main configs

- `scripts/configs/fedavg_cifar10.yaml`
- `scripts/configs/bbb_cifar10.yaml`
- `scripts/configs/fola_cifar10.yaml`
- selected FOLA after sweep: `scripts/configs/fola_cifar10_selected.yaml`

Legacy debug configs from earlier 100-client diagnostics should not be used for this profile.
