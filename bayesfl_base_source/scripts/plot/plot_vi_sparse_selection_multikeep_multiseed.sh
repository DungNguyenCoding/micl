#!/usr/bin/env bash
set -euo pipefail

# Plot VI sparse-selection multikeep/multiseed outputs. It generates:
#   - final/best accuracy vs keep ratio, mean±std over seeds
#   - ECE vs keep ratio
#   - communication/retention plots
#   - all accuracy/loss/ECE-vs-round curves for every run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DATASET="${DATASET:-mnist}"
MODEL="${MODEL:-resnet}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/vi_sparse_selection_${DATASET}_${MODEL}_extreme_a001_ub01_cf10_nowarmup_multiseed}"
PLOT_DIR="${PLOT_DIR:-plots/vi_sparse_selection_${DATASET}_${MODEL}_extreme_a001_ub01_cf10_nowarmup_multiseed}"

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

echo "===== Plot VI sparse-selection multikeep/multiseed ====="
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "PLOT_DIR=${PLOT_DIR}"
echo "RUNS=${#RUNS[@]}"

python utils.py sparse-ablation \
  --runs "${RUNS[@]}" \
  --output_dir "${PLOT_DIR}"

echo "===== VI sparse-selection plots generated ====="
find "${PLOT_DIR}" -type f | sort
