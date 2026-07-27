#!/usr/bin/env bash
set -euo pipefail

# Generic plot wrapper for sparse-selection or sparse-metric ablation outputs.
# It searches OUTPUT_ROOT for run folders and calls utils.py sparse-ablation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SEED="${SEED:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/sparse_selection_ablation_mnist_extreme_a001_ub01_cf10_nowarmup}"
PLOT_DIR="${PLOT_DIR:-plots/$(basename "${OUTPUT_ROOT}")}"

mkdir -p "${PLOT_DIR}"

RUNS=()
while IFS= read -r -d '' metrics; do
  run_dir="$(dirname "${metrics}")"
  label="$(basename "${run_dir}")"
  RUNS+=("${label}=${run_dir}")
done < <(find "${OUTPUT_ROOT}" -mindepth 2 -maxdepth 2 -name metrics.csv -print0 | sort -z)

if [[ "${#RUNS[@]}" -eq 0 ]]; then
  echo "[error] No metrics.csv files found under ${OUTPUT_ROOT}"
  exit 1
fi

echo "===== Plot sparse ablation ====="
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "PLOT_DIR=${PLOT_DIR}"
echo "RUNS=${#RUNS[@]}"

python utils.py sparse-ablation \
  --runs "${RUNS[@]}" \
  --output_dir "${PLOT_DIR}"

echo "===== Sparse ablation plots generated ====="
find "${PLOT_DIR}" -type f | sort
