#!/usr/bin/env bash
set -euo pipefail

# Short sanity test for the AirComp Bayesian FL paper simulator.
# This is not meant to reproduce the paper figures. It checks that the full
# algorithmic pipeline can run end-to-end quickly.

python aircomp_bayesfl.py \
  --experiment default \
  --methods fedavg,fedprox,proposed \
  --realizations 1 \
  --max_channel_uses 300000 \
  --eval_every 1 \
  --num_devices 8 \
  --coverage_radius_m 200 \
  --mean_client_examples 5 \
  --local_classes 1 \
  --power_dbm 23 \
  --noise_dbm -74 \
  --num_subchannels 1024 \
  --pathloss_alpha 4 \
  --gamma_db 10 \
  --lr 0.1 \
  --batch_size 10 \
  --local_epochs 1 \
  --vi_lambda 0.00002 \
  --mc_samples 2 \
  --rho_init 100 \
  --posterior_sample_scale 1.0 \
  --device auto \
  --num_workers 0 \
  --seed 42 \
  --output_dir outputs/aircomp_bayesfl_mnist_paper_smoke_seed42
