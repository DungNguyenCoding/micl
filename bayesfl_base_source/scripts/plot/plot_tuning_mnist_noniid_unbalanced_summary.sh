#!/usr/bin/env bash
set -euo pipefail

# Allow this script to be launched from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"


mkdir -p logs plots/tuning_compare

VI_ROOT="outputs/tune_vi_mnist_noniid_unbalanced"
OLA_ROOT="outputs/tune_ola_mnist_noniid_unbalanced"

VI_PLOTS="plots/tune_vi_mnist_noniid_unbalanced"
OLA_PLOTS="plots/tune_ola_mnist_noniid_unbalanced"
COMPARE_PLOTS="plots/tuning_compare"

METRICS=(
  global_accuracy
  global_loss
  global_ece
  global_nll
  local_accuracy_weighted
  train_loss
  posterior_sigma_mean
  posterior_snr_raw_p50
)

echo "===== Plot VI top-5 tuning runs ====="
if [[ -s "${VI_ROOT}/top5_runs.args" ]]; then
  mapfile -t VI_RUNS < "${VI_ROOT}/top5_runs.args"
  python utils.py mix \
    --runs "${VI_RUNS[@]}" \
    --metrics "${METRICS[@]}" \
    --output_dir "${VI_PLOTS}/top5"
else
  echo "[skip] Missing ${VI_ROOT}/top5_runs.args"
fi

echo "===== Plot OLA top-5 tuning runs ====="
if [[ -s "${OLA_ROOT}/top5_runs.args" ]]; then
  mapfile -t OLA_RUNS < "${OLA_ROOT}/top5_runs.args"
  python utils.py mix \
    --runs "${OLA_RUNS[@]}" \
    --metrics "${METRICS[@]}" \
    --output_dir "${OLA_PLOTS}/top5"
else
  echo "[skip] Missing ${OLA_ROOT}/top5_runs.args"
fi

echo "===== Plot best VI vs best OLA ====="
BEST_VI=""
BEST_OLA=""

if [[ -s "${VI_ROOT}/top5_runs.args" ]]; then
  BEST_VI="$(head -n 1 "${VI_ROOT}/top5_runs.args")"
fi

if [[ -s "${OLA_ROOT}/top5_runs.args" ]]; then
  BEST_OLA="$(head -n 1 "${OLA_ROOT}/top5_runs.args")"
fi

if [[ -n "${BEST_VI}" && -n "${BEST_OLA}" ]]; then
  python utils.py mix \
    --runs "${BEST_VI}" "${BEST_OLA}" \
    --metrics "${METRICS[@]}" \
    --output_dir "${COMPARE_PLOTS}/best_vi_vs_ola"
else
  echo "[skip] Cannot compare best VI vs OLA because one top5 file is missing"
fi

echo "===== Create ranking bar plots ====="
python - <<'PY'
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read_rows(path):
    path = Path(path)
    if not path.exists():
        print(f"[skip] Missing {path}")
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

def to_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

def plot_top10(csv_path, out_dir, title_prefix):
    rows = read_rows(csv_path)
    rows = [r for r in rows if r.get("status", "ok") == "ok"]
    if not rows:
        print(f"[skip] No rows in {csv_path}")
        return

    rows.sort(key=lambda r: int(r.get("rank", "999999")))
    top = rows[:10]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = [r.get("run_label", f"run_{i}") for i, r in enumerate(top)]

    plots = [
        ("final_global_accuracy", "Final global accuracy", "top10_final_global_accuracy.png", False),
        ("final_global_loss", "Final global loss", "top10_final_global_loss.png", True),
        ("final_global_ece", "Final global ECE", "top10_final_global_ece.png", True),
        ("final_posterior_sigma_mean", "Final posterior sigma mean", "top10_posterior_sigma_mean.png", True),
        ("final_posterior_snr_raw_p50", "Final posterior SNR p50", "top10_posterior_snr_p50.png", False),
    ]

    for key, ylabel, filename, lower_is_better in plots:
        values = [to_float(r.get(key, "")) for r in top]
        if all(v != v for v in values):
            print(f"[skip] {title_prefix}: no values for {key}")
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        y = list(range(len(labels)))
        ax.barh(y, values)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(ylabel)
        ax.set_title(f"{title_prefix}: {ylabel}")
        ax.grid(True, axis="x", alpha=0.3)
        fig.subplots_adjust(left=0.42, right=0.98, top=0.90, bottom=0.15)
        out = out_dir / filename
        fig.savefig(out, dpi=180)
        plt.close(fig)
        print(out)

plot_top10(
    "outputs/tune_vi_mnist_noniid_unbalanced/sweep_ranking.csv",
    "plots/tune_vi_mnist_noniid_unbalanced/ranking",
    "VI tuning",
)

plot_top10(
    "outputs/tune_ola_mnist_noniid_unbalanced/sweep_ranking.csv",
    "plots/tune_ola_mnist_noniid_unbalanced/ranking",
    "OLA tuning",
)
PY

echo "===== Tuning plot generation finished ====="
find plots/tune_vi_mnist_noniid_unbalanced plots/tune_ola_mnist_noniid_unbalanced plots/tuning_compare -type f -name "*.png" | sort
