#!/usr/bin/env bash
set -euo pipefail

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
