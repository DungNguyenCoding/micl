# Paper-like grouped Flower OTA-FL for CIFAR-10 Fig. 1

This version is designed to reproduce the proposed `Alg. 2` curves in Fig. 1 of
**Over-the-Air Aggregation-Based Federated Learning in Cache-Enabled Wireless Edge Networks**.

It keeps Flower in the experiment loop, but groups the 300 physical devices into fewer
Flower virtual clients to reduce Ray overhead on a single workstation:

- physical devices: `K = 300`
- Flower virtual clients: default `24`
- each Flower virtual client sequentially simulates about 12 or 13 physical devices
- the 24 Flower clients run in parallel through Flower/Ray

## What changed from the previous grouped version

The previous grouped version was optimized for runtime and general CIFAR-10 training.
This paper-like version changes the defaults and code paths most likely to affect Fig. 1 reproduction:

1. **Paper-like horizon**: default `num_rounds=1000`, because `D=307498`, `F=1024`, and `N=ceil(D/F)=301`, so 1000 rounds gives about 301,000 symbol transmissions.
2. **Evaluate every round**: default `eval_every=1`, matching the dense Fig. 1 curve.
3. **Local training is weaker**: default `local_epochs=1`, `batch_size=32`, `lr=0.02`. The paper does not disclose these exactly; this setting makes the BS-dataset communication effect more visible.
4. **Plain 7-layer CNN**: six 3x3 convolution layers plus one linear layer, exactly `D=307,498`, with no BatchNorm or Dropout.
5. **Split/runtime seeds separated**: `split_seed` fixes data/distance splits, `runtime_seed` controls local shuffling, wireless channels, and noise.
6. **Nested stratified BS cache**: the `|M0|=20`, `160`, and `1600` prefixes are all approximately class-balanced, reducing split noise.

## Run proposed Fig. 1 curves

```bash
python main.py \
  num_rounds=1000 \
  m0_values='[1600,160,20]' \
  num_devices=300 \
  num_flower_clients=24 \
  client_cpus=1 \
  client_gpus=0 \
  coverage_m=550 \
  local_epochs=1 \
  batch_size=32 \
  lr=0.02 \
  eval_every=1 \
  split_seed=42 \
  runtime_seed=42 \
  output_dir=outputs/fig1_paperlike_cov550
```

The ICC paper is internally ambiguous about BS coverage distance: the text says 550 m, while Table I says 400 m. Run this second command for the table setting:

```bash
python main.py \
  num_rounds=1000 \
  m0_values='[1600,160,20]' \
  num_devices=300 \
  num_flower_clients=24 \
  client_cpus=1 \
  client_gpus=0 \
  coverage_m=400 \
  local_epochs=1 \
  batch_size=32 \
  lr=0.02 \
  eval_every=1 \
  split_seed=42 \
  runtime_seed=42 \
  output_dir=outputs/fig1_paperlike_cov400
```

## Important limitation

This package reproduces the proposed `Alg. 2` curves. It still does **not** implement the TCI benchmark curve or centralized-learning upper bound. If you need a pixel-level clone of the complete paper figure, those baselines should be added as separate runs.
