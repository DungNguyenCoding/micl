#!/usr/bin/env bash
set -euo pipefail

# Allow this script to be launched from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"


# Compare sparse communication ratios for one method: VI.
# Note: in training, --sparse_ratio is the KEEP/SEND fraction.
# The run names also include prune/drop fraction: prune075_keep025 means drop 75%, send 25%.

METHOD="vi"
SEED="${SEED:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/sparse_comm_mnist_noniid_unbalanced}"
PLOT_ROOT="${PLOT_ROOT:-plots/sparse_comm_mnist_noniid_unbalanced/vi_ratio_sweep}"
DENSE_BASELINE="${DENSE_BASELINE:-outputs/final_compare_mnist_noniid_unbalanced/vi_seed42}"
FINAL_ROUND="${FINAL_ROUND:-200}"
EVAL_SCOPE="${EVAL_SCOPE:-global_test}"

mkdir -p "${PLOT_ROOT}" logs

# Create/refresh dense baseline symlink if possible.
BASELINE_LINK="${OUTPUT_ROOT}/vi_prune000_dense_baseline_seed${SEED}"
if [[ -d "${DENSE_BASELINE}" ]]; then
  mkdir -p "${OUTPUT_ROOT}"
  ln -sfn "$(realpath "${DENSE_BASELINE}")" "${BASELINE_LINK}"
fi

RUN_ARGS=()
SPECS_FILE="${PLOT_ROOT}/run_specs.csv"
echo "label,prune_fraction,keep_ratio,path" > "${SPECS_FILE}"

add_run() {
  local label="$1"
  local prune_fraction="$2"
  local keep_ratio="$3"
  local path="$4"
  if [[ -f "${path}/metrics.csv" ]]; then
    RUN_ARGS+=("${label}=${path}")
    echo "${label},${prune_fraction},${keep_ratio},${path}" >> "${SPECS_FILE}"
    echo "[add] ${label}: ${path}"
  else
    echo "[skip-missing] ${label}: ${path}/metrics.csv not found"
  fi
}

add_run "drop000_keep100" "0.0"  "1.00" "${BASELINE_LINK}"
add_run "drop050_keep050" "0.5"  "0.50" "${OUTPUT_ROOT}/vi_update_snr_prune050_keep050_seed${SEED}"
add_run "drop075_keep025" "0.75" "0.25" "${OUTPUT_ROOT}/vi_update_snr_prune075_keep025_seed${SEED}"
add_run "drop090_keep010" "0.9"  "0.10" "${OUTPUT_ROOT}/vi_update_snr_prune090_keep010_seed${SEED}"
add_run "drop095_keep005" "0.95" "0.05" "${OUTPUT_ROOT}/vi_update_snr_prune095_keep005_seed${SEED}"
add_run "drop098_keep002" "0.98" "0.02" "${OUTPUT_ROOT}/vi_update_snr_prune098_keep002_seed${SEED}"

echo "===== VI sparse-ratio sweep plotting ====="
echo "PLOT_ROOT=${PLOT_ROOT}"
echo "SPECS_FILE=${SPECS_FILE}"
echo "num_runs=${#RUN_ARGS[@]}"

run_mix() {
  local name="$1"
  shift
  local metrics=("$@")
  if [[ "${#RUN_ARGS[@]}" -lt 2 ]]; then
    echo "[skip] Need at least 2 completed runs for utils.py mix; found ${#RUN_ARGS[@]}"
    return 0
  fi
  echo "===== ${name} ====="
  mkdir -p "${PLOT_ROOT}/${name}"
  # Plot each metric separately. This makes the script robust when some
  # metrics are unavailable in dense baseline runs or older sparse runs.
  for metric in "${metrics[@]}"; do
    python utils.py mix \
      --runs "${RUN_ARGS[@]}" \
      --metrics "${metric}" \
      --output_dir "${PLOT_ROOT}/${name}" \
      || echo "[skip-plot] ${name}/${metric}: no usable data or unsupported column"
  done
}

run_mix "performance" \
  global_accuracy global_mean_accuracy global_mc_accuracy \
  global_loss global_mean_loss global_mc_loss global_nll \
  global_ece global_mean_ece global_mc_ece global_brier \
  global_mean_confidence global_mean_entropy \
  local_accuracy_weighted local_loss_weighted train_loss

run_mix "communication" \
  communication_dense_params communication_sent_params_mean communication_sent_params_total \
  communication_compression_ratio communication_dense_bytes communication_sparse_bytes \
  communication_index_bytes communication_value_bytes communication_saving_ratio

run_mix "bayesian_posterior" \
  posterior_sigma_mean posterior_sigma_p50 posterior_sigma_p90 \
  posterior_precision_mean posterior_precision_p50 posterior_precision_p90 \
  posterior_snr_raw_mean posterior_snr_raw_p50 posterior_snr_raw_p90 \
  posterior_snr_db_mean posterior_snr_db_p50 \
  posterior_snr_frac_lt_1 posterior_snr_frac_gt_1 \
  vi_elbo_loss_mean vi_kl_loss_mean vi_likelihood_loss_mean \
  vi_scale_mean vi_scale_p50 vi_scale_p90

run_mix "sparse_diagnostics" \
  sparse_score_mean sparse_score_p50 sparse_score_p90 sparse_threshold_mean \
  sparse_sent_update_l2_mean sparse_dropped_update_l2_mean sparse_sent_update_fraction_l2_mean

# Per-ratio final calibration and SNR plots.
echo "===== final calibration and SNR per ratio ====="
tail -n +2 "${SPECS_FILE}" | while IFS=, read -r label prune_fraction keep_ratio path; do
  if [[ -f "${path}/calibration_bins.csv" ]]; then
    python utils.py calibration \
      --calibration "${path}/calibration_bins.csv" \
      --round "${FINAL_ROUND}" \
      --eval_scope "${EVAL_SCOPE}" \
      --output_dir "${PLOT_ROOT}/calibration_final/${label}" || true
  fi
  if [[ -f "${path}/snr_histograms.csv" ]]; then
    python utils.py snr \
      --snr "${path}/snr_histograms.csv" \
      --round "${FINAL_ROUND}" \
      --layer all \
      --value_space db \
      --output_dir "${PLOT_ROOT}/snr_final/${label}" || true
  fi
done

# Summary CSV and across-ratio plots.
python - "${SPECS_FILE}" "${PLOT_ROOT}" <<'PY'
import csv
import math
import os
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt

specs_file = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
summary_dir = out_dir / "summary"
summary_dir.mkdir(parents=True, exist_ok=True)
client_agg_dir = out_dir / "sparse_client_aggregates"
client_agg_dir.mkdir(parents=True, exist_ok=True)

def fnum(x):
    try:
        if x is None or x == "":
            return math.nan
        v = float(x)
        return v if math.isfinite(v) else math.nan
    except Exception:
        return math.nan

def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))

def valid(v):
    return isinstance(v, (int, float)) and math.isfinite(v)

def first_valid(row, keys):
    for k in keys:
        if k in row:
            v = fnum(row.get(k))
            if valid(v):
                return v
    return math.nan

def row_round(row):
    v = first_valid(row, ["round", "server_round"])
    return v if valid(v) else -1

def mean_valid(vals):
    vals = [v for v in vals if valid(v)]
    return sum(vals) / len(vals) if vals else math.nan

def sanitize(s):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(s))

with specs_file.open(newline="") as fh:
    specs = list(csv.DictReader(fh))

summary_rows = []
client_round_rows = []

for spec in specs:
    label = spec["label"]
    run_path = Path(spec["path"])
    prune_fraction = fnum(spec["prune_fraction"])
    keep_ratio = fnum(spec["keep_ratio"])
    rows = read_csv(run_path / "metrics.csv")
    if not rows:
        continue
    rows_sorted = sorted(rows, key=row_round)
    final = rows_sorted[-1]

    acc_keys = ["global_accuracy", "global_mean_accuracy", "global_mc_accuracy"]
    loss_keys = ["global_loss", "global_mean_loss", "global_mc_loss"]
    ece_keys = ["global_ece", "global_mean_ece", "global_mc_ece"]

    acc_pairs = [(row_round(r), first_valid(r, acc_keys)) for r in rows_sorted]
    acc_pairs = [(rd, v) for rd, v in acc_pairs if valid(v)]
    ece_pairs = [(row_round(r), first_valid(r, ece_keys)) for r in rows_sorted]
    ece_pairs = [(rd, v) for rd, v in ece_pairs if valid(v)]

    best_acc_round, best_acc = max(acc_pairs, key=lambda x: x[1]) if acc_pairs else (math.nan, math.nan)
    best_ece_round, best_ece = min(ece_pairs, key=lambda x: x[1]) if ece_pairs else (math.nan, math.nan)

    def final_metric(keys):
        return first_valid(final, keys)
    def mean_metric(key):
        return mean_valid([fnum(r.get(key)) for r in rows_sorted])

    out = {
        "label": label,
        "path": str(run_path),
        "prune_fraction": prune_fraction,
        "keep_ratio": keep_ratio,
        "final_round": row_round(final),
        "final_global_accuracy": final_metric(acc_keys),
        "best_global_accuracy": best_acc,
        "best_accuracy_round": best_acc_round,
        "final_global_loss": final_metric(loss_keys),
        "final_global_ece": final_metric(ece_keys),
        "best_global_ece": best_ece,
        "best_ece_round": best_ece_round,
        "final_global_nll": final_metric(["global_nll"]),
        "final_global_brier": final_metric(["global_brier"]),
        "final_local_accuracy_weighted": final_metric(["local_accuracy_weighted"]),
        "communication_saving_ratio_final": final_metric(["communication_saving_ratio"]),
        "communication_saving_ratio_mean": mean_metric("communication_saving_ratio"),
        "communication_compression_ratio_final": final_metric(["communication_compression_ratio"]),
        "communication_compression_ratio_mean": mean_metric("communication_compression_ratio"),
        "communication_sent_params_mean_final": final_metric(["communication_sent_params_mean"]),
        "communication_sent_params_mean_mean": mean_metric("communication_sent_params_mean"),
        "communication_sparse_bytes_mean": mean_metric("communication_sparse_bytes"),
        "communication_dense_bytes_mean": mean_metric("communication_dense_bytes"),
        "posterior_sigma_mean_final": final_metric(["posterior_sigma_mean"]),
        "posterior_snr_raw_p50_final": final_metric(["posterior_snr_raw_p50"]),
        "vi_elbo_loss_mean_final": final_metric(["vi_elbo_loss_mean"]),
        "vi_kl_loss_mean_final": final_metric(["vi_kl_loss_mean"]),
    }

    # Dense baseline does not have sparse communication columns. Fill natural baseline values.
    if abs(prune_fraction) < 1e-12:
        if not valid(out["communication_saving_ratio_final"]):
            out["communication_saving_ratio_final"] = 0.0
        if not valid(out["communication_saving_ratio_mean"]):
            out["communication_saving_ratio_mean"] = 0.0
        if not valid(out["communication_compression_ratio_final"]):
            out["communication_compression_ratio_final"] = 1.0
        if not valid(out["communication_compression_ratio_mean"]):
            out["communication_compression_ratio_mean"] = 1.0
    summary_rows.append(out)

    # Aggregate per-client sparse_comm_metrics.csv by round if available.
    sc_rows = read_csv(run_path / "sparse_comm_metrics.csv")
    if sc_rows:
        by_round = defaultdict(list)
        for r in sc_rows:
            rd = int(fnum(r.get("round"))) if valid(fnum(r.get("round"))) else -1
            by_round[rd].append(r)
        candidate_metrics = [
            "sparse_num_params_sent", "sparse_num_params_total", "sparse_compression_ratio",
            "sparse_threshold", "sparse_score_mean", "sparse_score_p50", "sparse_score_p90",
            "sparse_sent_update_l2", "sparse_dropped_update_l2", "sparse_sent_update_fraction_l2",
        ]
        for rd, rr in sorted(by_round.items()):
            agg = {"label": label, "prune_fraction": prune_fraction, "keep_ratio": keep_ratio, "round": rd}
            for m in candidate_metrics:
                vals = [fnum(x.get(m)) for x in rr]
                agg[m + "_mean"] = mean_valid(vals)
            client_round_rows.append(agg)

summary_rows.sort(key=lambda r: r["prune_fraction"])
summary_csv = summary_dir / "sparse_ratio_summary.csv"
if summary_rows:
    fieldnames = list(summary_rows[0].keys())
    with summary_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)
    print(summary_csv)

client_csv = client_agg_dir / "sparse_client_round_agg.csv"
if client_round_rows:
    fieldnames = sorted(set().union(*(r.keys() for r in client_round_rows)))
    with client_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in client_round_rows:
            writer.writerow(r)
    print(client_csv)

def plot_vs_prune(metric, ylabel, filename):
    xs, ys, labels = [], [], []
    for r in summary_rows:
        v = r.get(metric)
        if valid(v):
            xs.append(r["prune_fraction"])
            ys.append(v)
            labels.append(f"drop {r['prune_fraction']:.2f}\nkeep {r['keep_ratio']:.2f}")
    if len(xs) < 2:
        return
    plt.figure(figsize=(8, 4.8))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Drop / prune fraction")
    plt.ylabel(ylabel)
    plt.title(ylabel + " vs sparse drop fraction")
    plt.xticks(xs, labels, rotation=30, ha="right")
    plt.tight_layout()
    path = summary_dir / filename
    plt.savefig(path, dpi=180)
    plt.close()
    print(path)

for metric, ylabel, filename in [
    ("final_global_accuracy", "Final global accuracy", "final_global_accuracy_vs_drop_fraction.png"),
    ("best_global_accuracy", "Best global accuracy", "best_global_accuracy_vs_drop_fraction.png"),
    ("final_global_loss", "Final global loss", "final_global_loss_vs_drop_fraction.png"),
    ("final_global_ece", "Final global ECE", "final_global_ece_vs_drop_fraction.png"),
    ("best_global_ece", "Best global ECE", "best_global_ece_vs_drop_fraction.png"),
    ("final_local_accuracy_weighted", "Final local weighted accuracy", "final_local_accuracy_vs_drop_fraction.png"),
    ("communication_saving_ratio_mean", "Mean communication saving ratio", "mean_comm_saving_vs_drop_fraction.png"),
    ("communication_compression_ratio_mean", "Mean communication compression ratio", "mean_comm_compression_vs_drop_fraction.png"),
    ("communication_sent_params_mean_mean", "Mean sent parameters", "mean_sent_params_vs_drop_fraction.png"),
    ("posterior_sigma_mean_final", "Final posterior sigma mean", "posterior_sigma_mean_vs_drop_fraction.png"),
    ("posterior_snr_raw_p50_final", "Final posterior SNR raw p50", "posterior_snr_p50_vs_drop_fraction.png"),
    ("vi_elbo_loss_mean_final", "Final VI ELBO loss mean", "vi_elbo_loss_vs_drop_fraction.png"),
    ("vi_kl_loss_mean_final", "Final VI KL loss mean", "vi_kl_loss_vs_drop_fraction.png"),
]:
    plot_vs_prune(metric, ylabel, filename)

# Accuracy vs communication saving scatter/line.
xs, ys, labs = [], [], []
for r in summary_rows:
    x = r.get("communication_saving_ratio_mean")
    y = r.get("best_global_accuracy")
    if valid(x) and valid(y):
        xs.append(x); ys.append(y); labs.append(r["label"])
if len(xs) >= 2:
    plt.figure(figsize=(7, 4.8))
    plt.plot(xs, ys, marker="o")
    for x, y, lab in zip(xs, ys, labs):
        plt.annotate(lab.replace("_", "\n"), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    plt.xlabel("Mean communication saving ratio")
    plt.ylabel("Best global accuracy")
    plt.title("Accuracy vs communication saving")
    plt.tight_layout()
    path = summary_dir / "accuracy_vs_communication_saving.png"
    plt.savefig(path, dpi=180)
    plt.close()
    print(path)

# Per-client sparse metrics over rounds.
if client_round_rows:
    metrics = [k for k in client_round_rows[0].keys() if k.endswith("_mean")]
    for metric in metrics:
        series = defaultdict(list)
        for r in client_round_rows:
            v = r.get(metric)
            if valid(v):
                series[r["label"]].append((r["round"], v))
        if sum(1 for points in series.values() if points) < 1:
            continue
        plt.figure(figsize=(8, 4.8))
        plotted = 0
        for label, points in sorted(series.items()):
            points = sorted(points)
            if len(points) < 2:
                continue
            plt.plot([p[0] for p in points], [p[1] for p in points], label=label)
            plotted += 1
        if plotted == 0:
            plt.close(); continue
        plt.xlabel("Round")
        plt.ylabel(metric)
        plt.title(metric + " over rounds")
        plt.legend(fontsize=8)
        plt.tight_layout()
        path = client_agg_dir / f"{sanitize(metric)}_round.png"
        plt.savefig(path, dpi=180)
        plt.close()
        print(path)
PY

echo "===== VI sparse-ratio plotting finished ====="
find "${PLOT_ROOT}" -type f \( -name "*.png" -o -name "*.csv" \) | sort
