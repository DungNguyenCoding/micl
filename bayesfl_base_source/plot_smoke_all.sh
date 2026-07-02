#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs plots/smoke_mix plots/smoke_fedavg plots/smoke_ola plots/smoke_vi

echo "===== Mixed plots: all methods ====="
python utils.py mix \
  --runs \
    fedavg=outputs/smoke_fedavg_mnist \
    ola=outputs/smoke_ola_mnist \
    vi=outputs/smoke_vi_mnist \
  --metrics \
    global_accuracy \
    global_loss \
    global_ece \
    train_loss \
    posterior_sigma_mean \
    posterior_snr_raw_p50 \
  --output_dir plots/smoke_mix

echo "===== Single metric plots ====="
python utils.py metric \
  --history outputs/smoke_fedavg_mnist/metrics.csv \
  --metric global_accuracy \
  --output_dir plots/smoke_fedavg

python utils.py metric \
  --history outputs/smoke_ola_mnist/metrics.csv \
  --metric global_accuracy \
  --output_dir plots/smoke_ola

python utils.py metric \
  --history outputs/smoke_vi_mnist/metrics.csv \
  --metric global_accuracy \
  --output_dir plots/smoke_vi

echo "===== Selected-client plots ====="
python utils.py selected \
  --selection outputs/smoke_fedavg_mnist/selection_summary.csv \
  --output_dir plots/smoke_fedavg

python utils.py selected \
  --selection outputs/smoke_ola_mnist/selection_summary.csv \
  --output_dir plots/smoke_ola

python utils.py selected \
  --selection outputs/smoke_vi_mnist/selection_summary.csv \
  --output_dir plots/smoke_vi

echo "===== Device radar plot ====="
python utils.py radar \
  --device_summary outputs/smoke_ola_mnist/device_summary.csv \
  --output_dir plots/smoke_ola

echo "===== Calibration plots ====="
python utils.py calibration \
  --calibration outputs/smoke_fedavg_mnist/calibration_bins.csv \
  --eval_scope global_test \
  --output_dir plots/smoke_fedavg

python utils.py calibration \
  --calibration outputs/smoke_ola_mnist/calibration_bins.csv \
  --eval_scope global_test \
  --output_dir plots/smoke_ola

python utils.py calibration \
  --calibration outputs/smoke_vi_mnist/calibration_bins.csv \
  --eval_scope global_test \
  --output_dir plots/smoke_vi

echo "===== Bayesian SNR plots ====="
python utils.py snr \
  --snr outputs/smoke_ola_mnist/snr_histograms.csv \
  --layer all \
  --value_space db \
  --output_dir plots/smoke_ola

python utils.py snr \
  --snr outputs/smoke_vi_mnist/snr_histograms.csv \
  --layer all \
  --value_space db \
  --output_dir plots/smoke_vi

echo "===== Plot tests finished ====="
find plots -type f -name "*.png" | sort
