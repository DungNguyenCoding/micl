#!/usr/bin/env bash
set -euo pipefail

# Extreme non-IID/unbalanced OLA sparse metric ablation with no sparse warmup.
# Goal: diagnose which OLA Bayesian score is useful against random top-k.
#
# Bayesian score candidates:
#   precision_update: |delta_mu| * accumulated precision
#   update_snr:       |delta_mu| / sigma
#   fisher_update:    |delta_mu| * current Fisher diagonal
#   kl:               coordinate-wise KL(q_local || q_global)
#
# Random baseline is run once per keep ratio using RANDOM_METRIC for the
# importance-score logging. Random selection itself is independent of metric.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SEED="${SEED:-42}"
DATASET="${DATASET:-mnist}"
MODEL="${MODEL:-mlp}"
MLP_HIDDEN="${MLP_HIDDEN:-128}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ola_sparse_metric_ablation_${DATASET}_extreme_a001_ub01_cf10_nowarmup}"
LOG_DIR="${LOG_DIR:-logs/ola_sparse_metric_ablation_${DATASET}_extreme_a001_ub01_cf10_nowarmup}"
KEEP_RATIOS=(${KEEP_RATIOS:-1.0 0.5 0.1 0.05 0.02})
BAYESIAN_METRICS=(${BAYESIAN_METRICS:-precision_update update_snr fisher_update kl})
RUN_RANDOM="${RUN_RANDOM:-true}"
RANDOM_METRIC="${RANDOM_METRIC:-precision_update}"

NUM_ROUNDS="${NUM_ROUNDS:-200}"
NONIID_ALPHA="${NONIID_ALPHA:-0.01}"
UNBALANCED_ALPHA="${UNBALANCED_ALPHA:-0.1}"
MIN_CLIENT_EXAMPLES="${MIN_CLIENT_EXAMPLES:-5}"
NUM_DEVICES="${NUM_DEVICES:-300}"
NUM_VIRTUAL_CLIENTS="${NUM_VIRTUAL_CLIENTS:-24}"
CLIENT_FRACTION="${CLIENT_FRACTION:-0.0333333333333}"  # 10 / 300 clients per round
SPARSE_WARMUP_ROUNDS="${SPARSE_WARMUP_ROUNDS:-0}"
SPARSE_MIN_KEEP="${SPARSE_MIN_KEEP:-100}"

VAL_RATIO="${VAL_RATIO:-0.1}"
EVAL_EVERY="${EVAL_EVERY:-1}"
HEAVY_EVAL_EVERY="${HEAVY_EVAL_EVERY:-5}"
LOCAL_EVAL_EVERY="${LOCAL_EVAL_EVERY:-5}"
LOCAL_EVAL_FRACTION="${LOCAL_EVAL_FRACTION:-0.2}"
SAVE_POSTERIOR_EVERY="${SAVE_POSTERIOR_EVERY:-10}"
EVAL_MC_SAMPLES="${EVAL_MC_SAMPLES:-5}"
POSTERIOR_SAMPLE_SCALE="${POSTERIOR_SAMPLE_SCALE:-0.001}"

OPTIMIZER="${OPTIMIZER:-sgd}"
LR="${LR:-0.005}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-2}"
OLA_PRIOR_LAMBDA="${OLA_PRIOR_LAMBDA:-0.05}"
PRECISION_INIT="${PRECISION_INIT:-0.001}"
PRECISION_FLOOR="${PRECISION_FLOOR:-1e-8}"
FISHER_CLIP="${FISHER_CLIP:-10.0}"

SKIP_EXISTING="${SKIP_EXISTING:-true}"
STOP_RAY_BETWEEN_RUNS="${STOP_RAY_BETWEEN_RUNS:-true}"
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

safe_metric_label() {
  echo "$1" | tr ',' '_' | tr '-' '_'
}

stop_ray() {
  if command -v ray >/dev/null 2>&1; then
    ray stop -f >/dev/null 2>&1 || true
  fi
}

run_one() {
  local SELECTION="$1"
  local METRIC="$2"
  local KEEP_RATIO="$3"
  local KEEP_LABEL="$4"
  local METRIC_LABEL
  METRIC_LABEL="$(safe_metric_label "${METRIC}")"
  local RUN_NAME
  if [[ "${SELECTION}" == "bayesian" ]]; then
    RUN_NAME="ola_sparse_bayesian_${METRIC_LABEL}_keep${KEEP_LABEL}_seed${SEED}"
  else
    RUN_NAME="ola_sparse_random_keep${KEEP_LABEL}_seed${SEED}"
  fi
  local RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
  local LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
  echo "ola,${DATASET},${SELECTION},${METRIC},${KEEP_RATIO},${RUN_DIR},${LOG_FILE}" >> "${MANIFEST}"

  if [[ "${SKIP_EXISTING}" == "true" && -f "${RUN_DIR}/run_summary.csv" ]]; then
    echo "[skip-existing] ${RUN_DIR}"
    return 0
  fi

  if [[ "${STOP_RAY_BETWEEN_RUNS}" == "true" ]]; then
    stop_ray
  fi

  echo "===== Running OLA selection=${SELECTION} metric=${METRIC} keep=${KEEP_RATIO} dataset=${DATASET} ====="
  python main.py \
    --method ola \
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
    --optimizer "${OPTIMIZER}" \
    --lr "${LR}" \
    --batch_size "${BATCH_SIZE}" \
    --local_epochs "${LOCAL_EPOCHS}" \
    --ola_prior_lambda "${OLA_PRIOR_LAMBDA}" \
    --precision_init "${PRECISION_INIT}" \
    --precision_floor "${PRECISION_FLOOR}" \
    --fisher_clip "${FISHER_CLIP}" \
    --sparse_comm true \
    --sparse_metric "${METRIC}" \
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
}

MANIFEST="${OUTPUT_ROOT}/ola_sparse_metric_ablation_manifest_seed${SEED}.csv"
echo "method,dataset,selection,sparse_metric,keep_ratio,output_dir,log_file" > "${MANIFEST}"

echo "===== OLA sparse-metric extreme no-warmup ablation ====="
echo "DATASET=${DATASET} MODEL=${MODEL} MLP_HIDDEN=${MLP_HIDDEN}"
echo "NONIID_ALPHA=${NONIID_ALPHA} UNBALANCED_ALPHA=${UNBALANCED_ALPHA} MIN_CLIENT_EXAMPLES=${MIN_CLIENT_EXAMPLES}"
echo "NUM_DEVICES=${NUM_DEVICES} NUM_VIRTUAL_CLIENTS=${NUM_VIRTUAL_CLIENTS} CLIENT_FRACTION=${CLIENT_FRACTION}"
echo "KEEP_RATIOS=${KEEP_RATIOS[*]} BAYESIAN_METRICS=${BAYESIAN_METRICS[*]} RUN_RANDOM=${RUN_RANDOM} SPARSE_WARMUP_ROUNDS=${SPARSE_WARMUP_ROUNDS}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "DEVICE=${DEVICE} CLIENT_GPUS=${CLIENT_GPUS_ARG}"

for KEEP_RATIO in "${KEEP_RATIOS[@]}"; do
  KEEP_LABEL="$(label_keep "${KEEP_RATIO}")"
  for METRIC in "${BAYESIAN_METRICS[@]}"; do
    run_one bayesian "${METRIC}" "${KEEP_RATIO}" "${KEEP_LABEL}"
  done
  if [[ "${RUN_RANDOM}" == "true" ]]; then
    run_one random "${RANDOM_METRIC}" "${KEEP_RATIO}" "${KEEP_LABEL}"
  fi
done

echo "===== OLA sparse-metric extreme no-warmup ablation finished ====="
echo "Manifest: ${MANIFEST}"
