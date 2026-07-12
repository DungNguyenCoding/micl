#!/usr/bin/env bash
set -euo pipefail

# Generate research diagnostics for the current validated MNIST non-IID unbalanced stage.
# This covers dense final comparison, sparse-ratio experiments, stabilized VI, and best VI method.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ROOT_OUT="plots/research_diagnostics_mnist_noniid_unbalanced"
FINAL_ROOT="outputs/final_compare_mnist_noniid_unbalanced"
SPARSE_ROOT="outputs/sparse_comm_mnist_noniid_unbalanced"
mkdir -p "${ROOT_OUT}" logs

FA="${FINAL_ROOT}/fedavg_seed42"
VI="${FINAL_ROOT}/vi_seed42"
OLA="${FINAL_ROOT}/ola_seed42"
VI_SPARSE_KEEP010="${SPARSE_ROOT}/vi_update_snr_prune090_keep010_seed42"
OLA_SPARSE_KEEP010="${SPARSE_ROOT}/ola_precision_update_prune090_keep010_seed42"
VI_DECAY="outputs/vi_mnist_stabilized_decay_seed42"
VI_SPARSE_KEEP010_DECAY="outputs/vi_sparse_keep010_decay_seed42"

for ENTRY in \
  "fedavg_seed42:${FA}" \
  "vi_seed42:${VI}" \
  "ola_seed42:${OLA}" \
  "vi_sparse_keep010:${VI_SPARSE_KEEP010}" \
  "ola_sparse_keep010:${OLA_SPARSE_KEEP010}" \
  "vi_stabilized_decay:${VI_DECAY}" \
  "vi_sparse_keep010_decay:${VI_SPARSE_KEEP010_DECAY}"
do
  LABEL="${ENTRY%%:*}"
  RUN="${ENTRY#*:}"
  if [[ -d "${RUN}" ]]; then
    echo "===== diagnostics: ${LABEL} ====="
    python utils.py diagnostics --run "${RUN}" --output_dir "${ROOT_OUT}/${LABEL}/diagnostics" || true
    python utils.py heterogeneity --run "${RUN}" --output_dir "${ROOT_OUT}/${LABEL}/heterogeneity" || true
    if [[ -f "${RUN}/snr_histograms.csv" ]]; then
      python utils.py snr-evolution \
        --snr "${RUN}/snr_histograms.csv" \
        --rounds 0 50 100 150 200 \
        --output_dir "${ROOT_OUT}/${LABEL}/snr_evolution" || true
    fi
  else
    echo "[skip] missing run: ${LABEL} -> ${RUN}"
  fi
done

if [[ -d "${FA}" && -d "${VI}" && -d "${OLA}" ]]; then
  python utils.py compare-diagnostics \
    --runs fedavg="${FA}" vi="${VI}" ola="${OLA}" \
    --output_dir "${ROOT_OUT}/dense_compare_best_final"
fi

if [[ -d "${VI_SPARSE_KEEP010}" && -d "${OLA_SPARSE_KEEP010}" ]]; then
  python utils.py compare-diagnostics \
    --runs \
      vi_dense="${VI}" \
      vi_sparse_keep010="${VI_SPARSE_KEEP010}" \
      ola_dense="${OLA}" \
      ola_sparse_keep010="${OLA_SPARSE_KEEP010}" \
    --output_dir "${ROOT_OUT}/dense_vs_sparse_best_final"
fi

if [[ -d "${VI_DECAY}" && -d "${VI_SPARSE_KEEP010}" ]]; then
  python utils.py compare-diagnostics \
    --runs old_dense_vi="${VI}" sparse_keep010="${VI_SPARSE_KEEP010}" vi_decay="${VI_DECAY}" \
    --output_dir "${ROOT_OUT}/vi_stability_compare"
fi

if [[ -d "${VI_SPARSE_KEEP010_DECAY}" ]]; then
  python utils.py compare-diagnostics \
    --runs \
      old_dense_vi="${VI}" \
      sparse_keep010="${VI_SPARSE_KEEP010}" \
      vi_decay="${VI_DECAY}" \
      sparse_keep010_decay="${VI_SPARSE_KEEP010_DECAY}" \
    --output_dir "${ROOT_OUT}/vi_sparse_decay_compare"
fi

if [[ -d "${VI_SPARSE_KEEP010_DECAY}" ]]; then
  python utils.py compare-diagnostics \
    --runs \
      fedavg="${FA}" \
      ola="${OLA}" \
      dense_vi="${VI}" \
      sparse_vi_keep010="${VI_SPARSE_KEEP010}" \
      sparse_vi_keep010_decay="${VI_SPARSE_KEEP010_DECAY}" \
    --output_dir "${ROOT_OUT}/final_best_method_compare"
fi

find "${ROOT_OUT}" -type f | sort
