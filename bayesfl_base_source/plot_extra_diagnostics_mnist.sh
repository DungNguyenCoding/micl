#!/usr/bin/env bash
set -euo pipefail

ROOT_OUT="plots/research_diagnostics_mnist_noniid_unbalanced"
FINAL_ROOT="outputs/final_compare_mnist_noniid_unbalanced"
SPARSE_ROOT="outputs/sparse_comm_mnist_noniid_unbalanced"
mkdir -p "${ROOT_OUT}" logs

FA="${FINAL_ROOT}/fedavg_seed42"
VI="${FINAL_ROOT}/vi_seed42"
OLA="${FINAL_ROOT}/ola_seed42"

# One-run diagnostics explaining stability/over-training/posterior uncertainty.
for RUN in "${FA}" "${VI}" "${OLA}"; do
  LABEL=$(basename "${RUN}")
  if [[ -d "${RUN}" ]]; then
    echo "===== diagnostics: ${LABEL} ====="
    python utils.py diagnostics --run "${RUN}" --output_dir "${ROOT_OUT}/${LABEL}/diagnostics"
    python utils.py heterogeneity --run "${RUN}" --output_dir "${ROOT_OUT}/${LABEL}/heterogeneity" || true
    if [[ -f "${RUN}/snr_histograms.csv" ]]; then
      python utils.py snr-evolution --snr "${RUN}/snr_histograms.csv" --rounds 0 50 100 150 200 --output_dir "${ROOT_OUT}/${LABEL}/snr_evolution" || true
    fi
  fi
done

# Dense-method best/final degradation summary.
python utils.py compare-diagnostics \
  --runs \
    fedavg="${FA}" \
    vi="${VI}" \
    ola="${OLA}" \
  --output_dir "${ROOT_OUT}/dense_compare_best_final"

# Sparse VI/OLA selected-ratio diagnostics if those runs exist.
SPARSE_VI="${SPARSE_ROOT}/vi_update_snr_prune090_keep010_seed42"
SPARSE_OLA="${SPARSE_ROOT}/ola_precision_update_prune090_keep010_seed42"
if [[ -d "${SPARSE_VI}" && -d "${SPARSE_OLA}" ]]; then
  python utils.py compare-diagnostics \
    --runs \
      vi_dense="${VI}" \
      vi_sparse_keep010="${SPARSE_VI}" \
      ola_dense="${OLA}" \
      ola_sparse_keep010="${SPARSE_OLA}" \
    --output_dir "${ROOT_OUT}/dense_vs_sparse_best_final"
fi

find "${ROOT_OUT}" -type f | sort
