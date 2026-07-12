#!/usr/bin/env bash
set -euo pipefail

# Allow this script to be launched from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"


FA="outputs/final_compare_mnist_noniid_unbalanced/fedavg_seed42"
OLA="outputs/final_compare_mnist_noniid_unbalanced/ola_seed42"
OUT="plots/final_compare_mnist_noniid_unbalanced/fa_vs_ola"

mkdir -p "${OUT}"

echo "===== FA vs OLA: performance plots ====="
python utils.py mix \
  --runs \
    fedavg="${FA}" \
    ola="${OLA}" \
  --metrics \
    global_accuracy \
    global_mean_accuracy \
    global_mc_accuracy \
    global_loss \
    global_mean_loss \
    global_mc_loss \
    global_nll \
    global_ece \
    global_mean_ece \
    global_mc_ece \
    global_brier \
    global_mean_confidence \
    global_mean_entropy \
    local_accuracy_weighted \
    local_loss_weighted \
    train_loss \
  --output_dir "${OUT}/performance"

echo "===== FA vs OLA: FL dynamics plots ====="
python utils.py mix \
  --runs \
    fedavg="${FA}" \
    ola="${OLA}" \
  --metrics \
    selected_count \
    selected_examples \
    selected_label_entropy_mean \
    selected_kl_to_global_label_mean \
    client_update_l2_mean \
    client_update_l2_std \
    client_update_cosine_mean \
    aggregation_delta_l2 \
    aggregation_delta_linf \
    aggregation_weight_entropy \
  --output_dir "${OUT}/fl_dynamics"

echo "===== FA vs OLA: runtime plots ====="
python utils.py mix \
  --runs \
    fedavg="${FA}" \
    ola="${OLA}" \
  --metrics \
    round_time_sec \
    fit_time_sec \
    aggregate_time_sec \
    eval_time_sec \
  --output_dir "${OUT}/runtime"

echo "===== OLA Bayesian posterior plots with FedAvg reference where available ====="
python utils.py mix \
  --runs \
    fedavg="${FA}" \
    ola="${OLA}" \
  --metrics \
    posterior_sigma_mean \
    posterior_sigma_p50 \
    posterior_sigma_p90 \
    posterior_precision_mean \
    posterior_precision_p50 \
    posterior_precision_p90 \
    posterior_snr_raw_mean \
    posterior_snr_raw_p50 \
    posterior_snr_raw_p90 \
    posterior_snr_db_mean \
    posterior_snr_db_p50 \
    posterior_snr_frac_lt_1 \
    posterior_snr_frac_gt_1 \
    ola_prior_loss_mean \
    ola_task_loss_mean \
    ola_fisher_mean \
    ola_precision_mean \
    ola_sigma_mean \
  --output_dir "${OUT}/bayesian_posterior"

echo "===== OLA-specific characteristics dashboard ====="
python utils.py characteristics \
  --run "${OLA}" \
  --method ola \
  --final_round 200 \
  --best_round 188 \
  --best_ece_round 129 \
  --output_dir "${OUT}/ola_characteristics"

echo "===== Selected-client plots ====="
python utils.py selected \
  --selection "${FA}/selection_summary.csv" \
  --output_dir "${OUT}/selection_fedavg"

python utils.py selected \
  --selection "${OLA}/selection_summary.csv" \
  --output_dir "${OUT}/selection_ola"

echo "===== Device radar plot, same for both methods ====="
python utils.py radar \
  --device_summary "${FA}/device_summary.csv" \
  --output_dir "${OUT}/device_distribution"

echo "===== Calibration plots: final round ====="
python utils.py calibration \
  --calibration "${FA}/calibration_bins.csv" \
  --round 200 \
  --eval_scope global_test \
  --output_dir "${OUT}/calibration_fedavg_final"

python utils.py calibration \
  --calibration "${OLA}/calibration_bins.csv" \
  --round 200 \
  --eval_scope global_test \
  --output_dir "${OUT}/calibration_ola_final"

echo "===== Calibration plots: best accuracy round ====="
# FedAvg best accuracy round = 183
# OLA best accuracy round = 188
python utils.py calibration \
  --calibration "${FA}/calibration_bins.csv" \
  --round 183 \
  --eval_scope global_test \
  --output_dir "${OUT}/calibration_fedavg_best_acc"

python utils.py calibration \
  --calibration "${OLA}/calibration_bins.csv" \
  --round 188 \
  --eval_scope global_test \
  --output_dir "${OUT}/calibration_ola_best_acc"

echo "===== Calibration plots: best ECE round ====="
# FedAvg best ECE round = 129
# OLA best ECE round = 129
python utils.py calibration \
  --calibration "${FA}/calibration_bins.csv" \
  --round 129 \
  --eval_scope global_test \
  --output_dir "${OUT}/calibration_fedavg_best_ece"

python utils.py calibration \
  --calibration "${OLA}/calibration_bins.csv" \
  --round 129 \
  --eval_scope global_test \
  --output_dir "${OUT}/calibration_ola_best_ece"

echo "===== OLA SNR density/CDF plots ====="
python utils.py snr \
  --snr "${OLA}/snr_histograms.csv" \
  --round 200 \
  --layer all \
  --value_space db \
  --output_dir "${OUT}/snr_ola_final"

python utils.py snr \
  --snr "${OLA}/snr_histograms.csv" \
  --round 188 \
  --layer all \
  --value_space db \
  --output_dir "${OUT}/snr_ola_best_acc"

echo "===== FA vs OLA plot generation finished ====="
find "${OUT}" -type f -name "*.png" | sort
