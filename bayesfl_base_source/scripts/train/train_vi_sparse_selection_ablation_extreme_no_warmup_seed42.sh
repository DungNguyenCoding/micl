#!/usr/bin/env bash
set -euo pipefail

# Extreme non-IID/unbalanced VI sparse-selection ablation with no sparse warmup.
# Goal: compare Bayesian update-SNR top-k selection vs random top-k selection
# under the same keep ratio and communication budget.
#
# Default stress setting:
#   noniid_alpha=0.01, unbalanced_alpha=0.1
#   300 physical clients, 24 Flower virtual clients
#   client_fraction=10/300 ~= 0.0333333333333
#   sparse_warmup_rounds=0
#
# You may override any variable before calling this script, e.g.:
#   DATASET=cifar10 MLP_HIDDEN=512,256 KEEP_RATIOS="0.1 0.05" bash ...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SEED="${SEED:-42}"
DATASET="${DATASET:-mnist}"
MODEL="${MODEL:-mlp}"
MLP_HIDDEN="${MLP_HIDDEN:-128}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/sparse_selection_ablation_${DATASET}_extreme_a001_ub01_cf10_nowarmup}"
LOG_DIR="${LOG_DIR:-logs/sparse_selection_ablation_${DATASET}_extreme_a001_ub01_cf10_nowarmup}"
KEEP_RATIOS=(${KEEP_RATIOS:-1.0 0.5 0.1 0.05 0.02})
SELECTIONS=(${SELECTIONS:-bayesian random})

NUM_ROUNDS="${NUM_ROUNDS:-200}"
NONIID_ALPHA="${NONIID_ALPHA:-0.01}"
UNBALANCED_ALPHA="${UNBALANCED_ALPHA:-0.1}"
MIN_CLIENT_EXAMPLES="${MIN_CLIENT_EXAMPLES:-5}"
NUM_DEVICES="${NUM_DEVICES:-300}"
NUM_VIRTUAL_CLIENTS="${NUM_VIRTUAL_CLIENTS:-24}"
CLIENT_FRACTION="${CLIENT_FRACTION:-0.0333333333333}"  # 10 / 300 clients per round
SPARSE_WARMUP_ROUNDS="${SPARSE_WARMUP_ROUNDS:-0}"
SPARSE_MIN_KEEP="${SPARSE_MIN_KEEP:-100}"
SPARSE_METRIC="${SPARSE_METRIC:-update_snr}"
VI_USE_DECAY="${VI_USE_DECAY:-true}"

VAL_RATIO="${VAL_RATIO:-0.1}"
EVAL_EVERY="${EVAL_EVERY:-1}"
HEAVY_EVAL_EVERY="${HEAVY_EVAL_EVERY:-5}"
LOCAL_EVAL_EVERY="${LOCAL_EVAL_EVERY:-5}"
LOCAL_EVAL_FRACTION="${LOCAL_EVAL_FRACTION:-0.2}"
SAVE_POSTERIOR_EVERY="${SAVE_POSTERIOR_EVERY:-10}"
EVAL_MC_SAMPLES="${EVAL_MC_SAMPLES:-5}"
POSTERIOR_SAMPLE_SCALE="${POSTERIOR_SAMPLE_SCALE:-0.001}"

BATCH_SIZE="${BATCH_SIZE:-256}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
VI_LR="${VI_LR:-0.005}"
VI_PRIOR_SCALE="${VI_PRIOR_SCALE:-0.05}"
VI_INIT_SCALE="${VI_INIT_SCALE:-0.05}"
VI_MIN_SCALE="${VI_MIN_SCALE:-1e-5}"
VI_LR_DECAY_MILESTONES="${VI_LR_DECAY_MILESTONES:-80,120,160}"
VI_LR_DECAY_GAMMA="${VI_LR_DECAY_GAMMA:-0.5}"
VI_MAX_SCALE="${VI_MAX_SCALE:-0.5}"

SKIP_EXISTING="${SKIP_EXISTING:-true}"
STOP_RAY_BETWEEN_RUNS="${STOP_RAY_BETWEEN_RUNS:-true}"
FORCE_DEVICE="${FORCE_DEVICE:-auto}"
CLIENT_GPUS="${CLIENT_GPUS:-0.5}"
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

MANIFEST="${OUTPUT_ROOT}/vi_sparse_selection_extreme_manifest_seed${SEED}.csv"
echo "method,dataset,selection,sparse_metric,keep_ratio,output_dir,log_file" > "${MANIFEST}"

echo "===== VI sparse-selection extreme no-warmup ablation ====="
echo "DATASET=${DATASET} MODEL=${MODEL} MLP_HIDDEN=${MLP_HIDDEN}"
echo "NONIID_ALPHA=${NONIID_ALPHA} UNBALANCED_ALPHA=${UNBALANCED_ALPHA} MIN_CLIENT_EXAMPLES=${MIN_CLIENT_EXAMPLES}"
echo "NUM_DEVICES=${NUM_DEVICES} NUM_VIRTUAL_CLIENTS=${NUM_VIRTUAL_CLIENTS} CLIENT_FRACTION=${CLIENT_FRACTION}"
echo "KEEP_RATIOS=${KEEP_RATIOS[*]} SELECTIONS=${SELECTIONS[*]} SPARSE_WARMUP_ROUNDS=${SPARSE_WARMUP_ROUNDS}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "DEVICE=${DEVICE} CLIENT_GPUS=${CLIENT_GPUS_ARG}"

for SELECTION in "${SELECTIONS[@]}"; do
  for KEEP_RATIO in "${KEEP_RATIOS[@]}"; do
    KEEP_LABEL="$(label_keep "${KEEP_RATIO}")"
    RUN_NAME="vi_sparse_${SELECTION}_keep${KEEP_LABEL}_seed${SEED}"
    RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
    LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
    echo "vi,${DATASET},${SELECTION},${SPARSE_METRIC},${KEEP_RATIO},${RUN_DIR},${LOG_FILE}" >> "${MANIFEST}"

    if [[ "${SKIP_EXISTING}" == "true" && -f "${RUN_DIR}/run_summary.csv" ]]; then
      echo "[skip-existing] ${RUN_DIR}"
      continue
    fi

    if [[ "${STOP_RAY_BETWEEN_RUNS}" == "true" ]]; then
      stop_ray
    fi

    DECAY_ARGS=()
    if [[ "${VI_USE_DECAY}" == "true" ]]; then
      DECAY_ARGS=(--vi_lr_decay_milestones "${VI_LR_DECAY_MILESTONES}" --vi_lr_decay_gamma "${VI_LR_DECAY_GAMMA}" --vi_max_scale "${VI_MAX_SCALE}")
    fi

    echo "===== Running VI ${SELECTION} keep=${KEEP_RATIO} dataset=${DATASET} ====="
    python main.py \
      --method vi \
      --dataset "${DATASET}" \
      --model "${MODEL}" \
      --iid false \
      --balanced false \
      --noniid_alpha "${NONIID_ALPHA}" \
      --unbalanced_alpha "${UNBALANCED_ALPHA}" \
      --min_client_examples "${MIN_CLIENT_EXAMPLES}" \
      --num_devices "${NUM_DEVICES}" \
      --num_virtual_clients "${NUM_VIRTUAL_CLIENTS}" \
      --client_fraction "${CLIENT_FRACTION}" \
      --num_rounds "${NUM_ROUNDS}" \
      --mlp_hidden "${MLP_HIDDEN}" \
      --val_ratio "${VAL_RATIO}" \
      --eval_every "${EVAL_EVERY}" \
      --heavy_eval_every "${HEAVY_EVAL_EVERY}" \
      --local_eval_every "${LOCAL_EVAL_EVERY}" \
      --local_eval_fraction "${LOCAL_EVAL_FRACTION}" \
      --save_posterior_every "${SAVE_POSTERIOR_EVERY}" \
      --eval_mc_samples "${EVAL_MC_SAMPLES}" \
      --posterior_sample_scale "${POSTERIOR_SAMPLE_SCALE}" \
      --metrics_level bayes \
      --device "${DEVICE}" \
      --client_gpus "${CLIENT_GPUS_ARG}" \
      --client_cpus "${CLIENT_CPUS}" \
      --num_workers "${NUM_WORKERS}" \
      --torch_threads "${TORCH_THREADS}" \
      --seed "${SEED}" \
      --vi_lr "${VI_LR}" \
      "${DECAY_ARGS[@]}" \
      --vi_prior_scale "${VI_PRIOR_SCALE}" \
      --vi_init_scale "${VI_INIT_SCALE}" \
      --vi_min_scale "${VI_MIN_SCALE}" \
      --vi_particles 1 \
      --bayes_aggregation product \
      --batch_size "${BATCH_SIZE}" \
      --local_epochs "${LOCAL_EPOCHS}" \
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

echo "===== VI sparse-selection extreme no-warmup ablation finished ====="
echo "Manifest: ${MANIFEST}"
