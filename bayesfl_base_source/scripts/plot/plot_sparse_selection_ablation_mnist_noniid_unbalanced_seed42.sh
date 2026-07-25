#!/usr/bin/env bash
set -euo pipefail

# Plot Bayesian-vs-random sparse-selection ablation outputs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SEED="${SEED:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/sparse_selection_ablation_mnist_noniid_unbalanced}"
PLOT_DIR="${PLOT_DIR:-plots/sparse_selection_ablation_mnist_noniid_unbalanced}"
KEEP_LABELS=(${KEEP_LABELS:-100 050 010 005 002})
SELECTIONS=(${SELECTIONS:-bayesian random})
METHODS=(${METHODS:-vi ola})

RUNS=()
for METHOD in "${METHODS[@]}"; do
  for SELECTION in "${SELECTIONS[@]}"; do
    for KEEP_LABEL in "${KEEP_LABELS[@]}"; do
      RUN_DIR="${OUTPUT_ROOT}/${METHOD}_sparse_${SELECTION}_keep${KEEP_LABEL}_seed${SEED}"
      if [[ -f "${RUN_DIR}/metrics.csv" ]]; then
        RUNS+=("${METHOD}_${SELECTION}_keep${KEEP_LABEL}=${RUN_DIR}")
      else
        echo "[skip-missing] ${RUN_DIR}"
      fi
    done
  done
done

if [[ "${#RUNS[@]}" -eq 0 ]]; then
  echo "[error] No sparse-selection ablation runs found under ${OUTPUT_ROOT}"
  exit 1
fi

mkdir -p "${PLOT_DIR}"
python utils.py sparse-ablation \
  --runs "${RUNS[@]}" \
  --output_dir "${PLOT_DIR}"

echo "===== Sparse-selection ablation plots generated ====="
find "${PLOT_DIR}" -type f | sort
