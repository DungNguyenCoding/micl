#!/usr/bin/env bash
set -euo pipefail

# Reproduce the default scenario used for Fig. 2 and Fig. 6 in AirComBayesFL:
# scarce, one-label, non-IID MNIST with K=40 devices and mean local sample count 10.
# For closer paper-grade curves, set REALIZATIONS=10 and MAX_CHANNEL_USES=30000000.

REALIZATIONS="${REALIZATIONS:-10}"
MAX_CHANNEL_USES="${MAX_CHANNEL_USES:-30000000}"
METHODS="${METHODS:-fedavg,fedprox,scaffold,proposed}"
DEVICE="${DEVICE:-auto}"

python aircomp_bayesfl.py \
  --experiment default \
  --methods "${METHODS}" \
  --realizations "${REALIZATIONS}" \
  --max_channel_uses "${MAX_CHANNEL_USES}" \
  --eval_every 1 \
  --num_devices 40 \
  --coverage_radius_m 200 \
  --mean_client_examples 10 \
  --local_classes 1 \
  --power_dbm 23 \
  --noise_dbm -74 \
  --num_subchannels 1024 \
  --pathloss_alpha 4 \
  --gamma_db 10 \
  --lr 0.1 \
  --batch_size 10 \
  --local_epochs 3 \
  --vi_lambda 0.00002 \
  --mc_samples 5 \
  --rho_init 100 \
  --posterior_sample_scale 1.0 \
  --fedprox_mu 0.001 \
  --device "${DEVICE}" \
  --num_workers 0 \
  --seed 42 \
  --output_dir outputs/aircomp_bayesfl_mnist_paper_default_seed42
