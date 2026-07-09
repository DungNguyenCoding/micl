#!/usr/bin/env bash
set -euo pipefail

FA="outputs/final_compare_mnist_noniid_unbalanced/fedavg_seed42"
VI="outputs/final_compare_mnist_noniid_unbalanced/vi_seed42"
OUT="plots/final_compare_mnist_noniid_unbalanced/fa_vs_vi"

mkdir -p "${OUT}"

echo "===== FA vs VI: performance plots ====="
python utils.py mix \
  --runs \
    fedavg="${FA}" \
    vi="${VI}" \
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

echo "===== FA vs VI: FL dynamics plots ====="
python utils.py mix \
  --runs \
    fedavg="${FA}" \
    vi="${VI}" \
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

echo "===== FA vs VI: runtime plots ====="
python utils.py mix \
  --runs \
    fedavg="${FA}" \
    vi="${VI}" \
  --metrics \
    round_time_sec \
    fit_time_sec \
    aggregate_time_sec \
    eval_time_sec \
  --output_dir "${OUT}/runtime"

echo "===== VI Bayesian posterior plots ====="
python utils.py mix \
  --runs \
    fedavg="${FA}" \
    vi="${VI}" \
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
    vi_elbo_loss_mean \
    vi_kl_loss_mean \
    vi_likelihood_loss_mean \
    vi_scale_mean \
    vi_scale_p50 \
    vi_scale_p90 \
  --output_dir "${OUT}/bayesian_posterior"

echo "===== Selected-client plots ====="
python utils.py selected \
  --selection "${FA}/selection_summary.csv" \
  --output_dir "${OUT}/selection_fedavg"

python utils.py selected \
  --selection "${VI}/selection_summary.csv" \
  --output_dir "${OUT}/selection_vi"

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
  --calibration "${VI}/calibration_bins.csv" \
  --round 200 \
  --eval_scope global_test \
  --output_dir "${OUT}/calibration_vi_final"

echo "===== Calibration plots: best accuracy round ====="
# FedAvg best accuracy round = 183
# VI best accuracy round = 106
python utils.py calibration \
  --calibration "${FA}/calibration_bins.csv" \
  --round 183 \
  --eval_scope global_test \
  --output_dir "${OUT}/calibration_fedavg_best_acc"

python utils.py calibration \
  --calibration "${VI}/calibration_bins.csv" \
  --round 106 \
  --eval_scope global_test \
  --output_dir "${OUT}/calibration_vi_best_acc"

echo "===== Calibration plots: best ECE round ====="
# FedAvg best ECE round = 129
# VI best ECE round = 42
python utils.py calibration \
  --calibration "${FA}/calibration_bins.csv" \
  --round 129 \
  --eval_scope global_test \
  --output_dir "${OUT}/calibration_fedavg_best_ece"

python utils.py calibration \
  --calibration "${VI}/calibration_bins.csv" \
  --round 42 \
  --eval_scope global_test \
  --output_dir "${OUT}/calibration_vi_best_ece"

echo "===== VI SNR density/CDF plots ====="
python utils.py snr \
  --snr "${VI}/snr_histograms.csv" \
  --round 200 \
  --layer all \
  --value_space db \
  --output_dir "${OUT}/snr_vi_final"

python utils.py snr \
  --snr "${VI}/snr_histograms.csv" \
  --round 106 \
  --layer all \
  --value_space db \
  --output_dir "${OUT}/snr_vi_best_acc"

python utils.py snr \
  --snr "${VI}/snr_histograms.csv" \
  --round 42 \
  --layer all \
  --value_space db \
  --output_dir "${OUT}/snr_vi_best_ece"

echo "===== FA vs VI plot generation finished ====="
find "${OUT}" -type f -name "*.png" | sort
