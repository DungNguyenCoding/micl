#!/usr/bin/env bash
set -euo pipefail

# Step 2: Sparse VI communication sweep
# IMPORTANT SEMANTICS:
#   The paper-style ratios below are PRUNE/DROP fractions.
#   The code argument --sparse_ratio is the KEEP/SEND fraction.
#   Therefore: --sparse_ratio = 1.0 - PRUNE_FRACTION
#
# Example:
#   PRUNE_FRACTION=0.75 means drop 75% and send top 25%.
#   The script passes --sparse_ratio 0.25.
#
# Baseline PRUNE_FRACTION=0.0 is dense VI, so by default this script reuses
# outputs/final_compare_mnist_noniid_unbalanced/vi_seed42 instead of rerunning.

ROOT_DIR="${ROOT_DIR:-$PWD}"
SEED="${SEED:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/sparse_comm_mnist_noniid_unbalanced}"
LOG_DIR="${LOG_DIR:-logs/sparse_comm_mnist_noniid_unbalanced}"
BASELINE_DENSE_RUN="${BASELINE_DENSE_RUN:-outputs/final_compare_mnist_noniid_unbalanced/vi_seed42}"

# Requested pruning/drop fractions. 0.0 is handled as dense baseline.
PRUNE_FRACTIONS=(${PRUNE_FRACTIONS:-0.0 0.5 0.75 0.9 0.95 0.98})

# Sparse-communication controls
SPARSE_WARMUP_ROUNDS="${SPARSE_WARMUP_ROUNDS:-20}"
SPARSE_MIN_KEEP="${SPARSE_MIN_KEEP:-100}"
SPARSE_METRIC="${SPARSE_METRIC:-update_snr}"

# Run controls
NUM_ROUNDS="${NUM_ROUNDS:-200}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
STOP_RAY_BETWEEN_RUNS="${STOP_RAY_BETWEEN_RUNS:-true}"

# Device controls
FORCE_DEVICE="${FORCE_DEVICE:-auto}"       # auto | cuda | cpu
CLIENT_GPUS="${CLIENT_GPUS:-0.5}"
CLIENT_CPUS="${CLIENT_CPUS:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TORCH_THREADS="${TORCH_THREADS:-1}"

cd "${ROOT_DIR}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

has_cuda() {
python - <<'PY'
try:
    import torch
    print('yes' if torch.cuda.is_available() else 'no')
except Exception:
    print('no')
PY
}

if [[ "${FORCE_DEVICE}" == "cuda" ]]; then
  DEVICE="cuda"
  CLIENT_GPUS_ARG="${CLIENT_GPUS}"
elif [[ "${FORCE_DEVICE}" == "cpu" ]]; then
  DEVICE="cpu"
  CLIENT_GPUS_ARG="0"
else
  if [[ "$(has_cuda)" == "yes" ]]; then
    DEVICE="cuda"
    CLIENT_GPUS_ARG="${CLIENT_GPUS}"
  else
    DEVICE="cpu"
    CLIENT_GPUS_ARG="0"
  fi
fi

echo "===== Sparse VI sweep ====="
echo "ROOT_DIR=${ROOT_DIR}"
echo "SEED=${SEED}"
echo "DEVICE=${DEVICE}"
echo "CLIENT_GPUS=${CLIENT_GPUS_ARG}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "LOG_DIR=${LOG_DIR}"
echo "PRUNE_FRACTIONS=${PRUNE_FRACTIONS[*]}"
echo "SPARSE_METRIC=${SPARSE_METRIC}"
echo "SPARSE_WARMUP_ROUNDS=${SPARSE_WARMUP_ROUNDS}"
echo "SPARSE_MIN_KEEP=${SPARSE_MIN_KEEP}"
echo "SKIP_EXISTING=${SKIP_EXISTING}"

to_label() {
python - "$1" <<'PY'
import sys
x = float(sys.argv[1])
print(f"{int(round(x * 100)):03d}")
PY
}

keep_from_prune() {
python - "$1" <<'PY'
import sys
p = float(sys.argv[1])
keep = 1.0 - p
if keep < 0:
    keep = 0.0
if keep > 1:
    keep = 1.0
print(f"{keep:.6f}")
PY
}

stop_ray() {
  if command -v ray >/dev/null 2>&1; then
    ray stop -f >/dev/null 2>&1 || true
  fi
}

# Dense baseline for prune fraction 0.0.
# This is equivalent to no sparse dropping. Reusing existing dense VI is preferred.
BASELINE_LINK="${OUTPUT_ROOT}/vi_prune000_dense_baseline_seed${SEED}"
if [[ -d "${BASELINE_DENSE_RUN}" ]]; then
  if [[ -L "${BASELINE_LINK}" || ! -e "${BASELINE_LINK}" ]]; then
    ln -sfn "$(realpath "${BASELINE_DENSE_RUN}")" "${BASELINE_LINK}"
    echo "[baseline] prune_fraction=0.0 uses dense VI run: ${BASELINE_LINK} -> ${BASELINE_DENSE_RUN}"
  else
    echo "[baseline] ${BASELINE_LINK} exists and is not a symlink; leaving unchanged."
  fi
else
  echo "[warning] Dense baseline not found: ${BASELINE_DENSE_RUN}"
  echo "          Run vi_mnist_noniid_unbalanced.sh first, or set BASELINE_DENSE_RUN=/path/to/vi_seed42"
fi

echo "baseline_dense_run=${BASELINE_DENSE_RUN}" > "${OUTPUT_ROOT}/vi_sweep_info_seed${SEED}.txt"
echo "baseline_link=${BASELINE_LINK}" >> "${OUTPUT_ROOT}/vi_sweep_info_seed${SEED}.txt"
echo "prune_fraction,keep_ratio,output_dir,log_file" > "${OUTPUT_ROOT}/vi_sparse_sweep_manifest_seed${SEED}.csv"

for PRUNE_FRACTION in "${PRUNE_FRACTIONS[@]}"; do
  # 0.0 is the dense baseline, not a sparse run.
  if python - "$PRUNE_FRACTION" <<'PY'
import sys
raise SystemExit(0 if abs(float(sys.argv[1])) < 1e-12 else 1)
PY
  then
    echo "[skip-run] prune_fraction=0.0 is dense baseline; not rerunning sparse VI."
    echo "0.0,1.000000,${BASELINE_LINK}," >> "${OUTPUT_ROOT}/vi_sparse_sweep_manifest_seed${SEED}.csv"
    continue
  fi

  KEEP_RATIO="$(keep_from_prune "${PRUNE_FRACTION}")"
  PRUNE_LABEL="$(to_label "${PRUNE_FRACTION}")"
  KEEP_LABEL="$(to_label "${KEEP_RATIO}")"

  RUN_NAME="vi_update_snr_prune${PRUNE_LABEL}_keep${KEEP_LABEL}_seed${SEED}"
  RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
  LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

  echo "${PRUNE_FRACTION},${KEEP_RATIO},${RUN_DIR},${LOG_FILE}" >> "${OUTPUT_ROOT}/vi_sparse_sweep_manifest_seed${SEED}.csv"

  if [[ "${SKIP_EXISTING}" == "true" && -f "${RUN_DIR}/run_summary.csv" ]]; then
    echo "[skip-existing] ${RUN_DIR}"
    continue
  fi

  echo "===== Running sparse VI: prune_fraction=${PRUNE_FRACTION}, keep/send ratio=${KEEP_RATIO} ====="
  echo "Output: ${RUN_DIR}"
  echo "Log:    ${LOG_FILE}"

  if [[ "${STOP_RAY_BETWEEN_RUNS}" == "true" ]]; then
    stop_ray
  fi

  python main.py \
    --method vi \
    --dataset mnist \
    --model mlp \
    --iid false \
    --balanced false \
    --noniid_alpha 0.1 \
    --unbalanced_alpha 0.5 \
    --num_devices 300 \
    --num_virtual_clients 24 \
    --client_fraction 0.05 \
    --num_rounds "${NUM_ROUNDS}" \
    --mlp_hidden 128 \
    --val_ratio 0.1 \
    --eval_every 1 \
    --heavy_eval_every 5 \
    --local_eval_every 5 \
    --local_eval_fraction 0.2 \
    --save_posterior_every 10 \
    --eval_mc_samples 5 \
    --posterior_sample_scale 0.001 \
    --metrics_level bayes \
    --device "${DEVICE}" \
    --client_gpus "${CLIENT_GPUS_ARG}" \
    --client_cpus "${CLIENT_CPUS}" \
    --num_workers "${NUM_WORKERS}" \
    --torch_threads "${TORCH_THREADS}" \
    --seed "${SEED}" \
    --vi_lr 0.005 \
    --vi_prior_scale 0.05 \
    --vi_init_scale 0.05 \
    --vi_min_scale 1e-5 \
    --vi_particles 1 \
    --bayes_aggregation product \
    --batch_size 256 \
    --local_epochs 5 \
    --sparse_comm true \
    --sparse_metric "${SPARSE_METRIC}" \
    --sparse_ratio "${KEEP_RATIO}" \
    --sparse_warmup_rounds "${SPARSE_WARMUP_ROUNDS}" \
    --sparse_min_keep "${SPARSE_MIN_KEEP}" \
    --output_dir "${RUN_DIR}" \
    > "${LOG_FILE}" 2>&1

  echo "[done] ${RUN_NAME}"

  if [[ "${STOP_RAY_BETWEEN_RUNS}" == "true" ]]; then
    stop_ray
  fi

done

echo "===== Sparse VI sweep finished ====="
echo "Manifest: ${OUTPUT_ROOT}/vi_sparse_sweep_manifest_seed${SEED}.csv"
echo "Outputs:"
find "${OUTPUT_ROOT}" -maxdepth 1 \( -type d -o -type l \) -name "vi_*seed${SEED}*" -print | sort
