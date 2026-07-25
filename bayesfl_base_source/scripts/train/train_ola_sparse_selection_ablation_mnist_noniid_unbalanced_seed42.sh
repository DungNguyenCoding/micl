#!/usr/bin/env bash
set -euo pipefail

# Sparse-selection ablation for OLA/FOLA Bayesian FL.
# Compares Bayesian precision-update top-k selection against random top-k
# selection under identical keep ratios and training settings.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SEED="${SEED:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/sparse_selection_ablation_mnist_noniid_unbalanced}"
LOG_DIR="${LOG_DIR:-logs/sparse_selection_ablation_mnist_noniid_unbalanced}"
KEEP_RATIOS=(${KEEP_RATIOS:-1.0 0.5 0.1 0.05 0.02})
SELECTIONS=(${SELECTIONS:-bayesian random})

NUM_ROUNDS="${NUM_ROUNDS:-200}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
STOP_RAY_BETWEEN_RUNS="${STOP_RAY_BETWEEN_RUNS:-true}"
SPARSE_WARMUP_ROUNDS="${SPARSE_WARMUP_ROUNDS:-20}"
SPARSE_MIN_KEEP="${SPARSE_MIN_KEEP:-100}"
SPARSE_METRIC="${SPARSE_METRIC:-precision_update}"

FORCE_DEVICE="${FORCE_DEVICE:-auto}"
CLIENT_GPUS="${CLIENT_GPUS:-0.25}"
CLIENT_CPUS="${CLIENT_CPUS:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TORCH_THREADS="${TORCH_THREADS:-1}"

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
  DEVICE="cuda"; CLIENT_GPUS_ARG="${CLIENT_GPUS}"
elif [[ "${FORCE_DEVICE}" == "cpu" ]]; then
  DEVICE="cpu"; CLIENT_GPUS_ARG="0"
else
  if [[ "$(has_cuda)" == "yes" ]]; then
    DEVICE="cuda"; CLIENT_GPUS_ARG="${CLIENT_GPUS}"
  else
    DEVICE="cpu"; CLIENT_GPUS_ARG="0"
  fi
fi

label_keep() {
python - "$1" <<'PY'
import sys
x=float(sys.argv[1])
print(f"{int(round(x*100)):03d}")
PY
}

stop_ray() {
  if command -v ray >/dev/null 2>&1; then
    ray stop -f >/dev/null 2>&1 || true
  fi
}

MANIFEST="${OUTPUT_ROOT}/ola_sparse_selection_ablation_manifest_seed${SEED}.csv"
echo "method,selection,keep_ratio,output_dir,log_file" > "${MANIFEST}"

echo "===== OLA sparse-selection ablation ====="
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "LOG_DIR=${LOG_DIR}"
echo "KEEP_RATIOS=${KEEP_RATIOS[*]}"
echo "SELECTIONS=${SELECTIONS[*]}"
echo "SPARSE_METRIC=${SPARSE_METRIC}"
echo "DEVICE=${DEVICE} CLIENT_GPUS=${CLIENT_GPUS_ARG}"

for SELECTION in "${SELECTIONS[@]}"; do
  for KEEP_RATIO in "${KEEP_RATIOS[@]}"; do
    KEEP_LABEL="$(label_keep "${KEEP_RATIO}")"
    RUN_NAME="ola_sparse_${SELECTION}_keep${KEEP_LABEL}_seed${SEED}"
    RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
    LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
    echo "ola,${SELECTION},${KEEP_RATIO},${RUN_DIR},${LOG_FILE}" >> "${MANIFEST}"

    if [[ "${SKIP_EXISTING}" == "true" && -f "${RUN_DIR}/run_summary.csv" ]]; then
      echo "[skip-existing] ${RUN_DIR}"
      continue
    fi

    if [[ "${STOP_RAY_BETWEEN_RUNS}" == "true" ]]; then
      stop_ray
    fi

    echo "===== Running OLA ${SELECTION} keep=${KEEP_RATIO} ====="
    echo "Output: ${RUN_DIR}"
    echo "Log:    ${LOG_FILE}"

    python main.py \
      --method ola \
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
      --optimizer sgd \
      --lr 0.005 \
      --batch_size 32 \
      --local_epochs 2 \
      --ola_prior_lambda 0.05 \
      --precision_init 0.001 \
      --precision_floor 1e-8 \
      --fisher_clip 10.0 \
      --sparse_comm true \
      --sparse_metric "${SPARSE_METRIC}" \
      --sparse_selection "${SELECTION}" \
      --sparse_ratio "${KEEP_RATIO}" \
      --sparse_warmup_rounds "${SPARSE_WARMUP_ROUNDS}" \
      --sparse_min_keep "${SPARSE_MIN_KEEP}" \
      --save_best_checkpoints true \
      --output_dir "${RUN_DIR}" \
      > "${LOG_FILE}" 2>&1

    echo "[done] ${RUN_NAME}"
    if [[ "${STOP_RAY_BETWEEN_RUNS}" == "true" ]]; then
      stop_ray
    fi
  done
done

echo "===== OLA sparse-selection ablation finished ====="
echo "Manifest: ${MANIFEST}"
