#!/usr/bin/env bash
set -euo pipefail

# VI-only sparse-selection ablation for the current research direction.
# Runs Bayesian update-SNR top-k vs random top-k for multiple keep ratios and
# multiple random seeds.  Defaults use a lightweight ResNet-style model.
#
# Example overrides:
#   DATASET=cifar10 MODEL=resnet RESNET_WIDTH=16 SEEDS="42" NUM_ROUNDS=20 bash ...
#   MODEL=mlp MLP_HIDDEN=128 SEEDS="42 43 44" bash ...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DATASET="${DATASET:-mnist}"
MODEL="${MODEL:-resnet}"
MLP_HIDDEN="${MLP_HIDDEN:-128}"
RESNET_WIDTH="${RESNET_WIDTH:-16}"
RESNET_BLOCKS="${RESNET_BLOCKS:-1,1,1}"
SEEDS=(${SEEDS:-42 43 44})
KEEP_RATIOS=(${KEEP_RATIOS:-1.0 0.75 0.5 0.25 0.1 0.05 0.02})
SELECTIONS=(${SELECTIONS:-bayesian random})
SPARSE_METRIC="${SPARSE_METRIC:-update_snr}"

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

# ResNet VI is heavier than MLP VI, so these defaults are slightly more conservative.
BATCH_SIZE="${BATCH_SIZE:-128}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-3}"
VI_LR="${VI_LR:-0.003}"
VI_PRIOR_SCALE="${VI_PRIOR_SCALE:-0.05}"
VI_INIT_SCALE="${VI_INIT_SCALE:-0.05}"
VI_MIN_SCALE="${VI_MIN_SCALE:-1e-5}"
VI_LR_DECAY_MILESTONES="${VI_LR_DECAY_MILESTONES:-80,120,160}"
VI_LR_DECAY_GAMMA="${VI_LR_DECAY_GAMMA:-0.5}"
VI_MAX_SCALE="${VI_MAX_SCALE:-0.5}"
VI_PARTICLES="${VI_PARTICLES:-1}"
VI_USE_DECAY="${VI_USE_DECAY:-true}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/vi_sparse_selection_${DATASET}_${MODEL}_extreme_a001_ub01_cf10_nowarmup_multiseed}"
LOG_DIR="${LOG_DIR:-logs/vi_sparse_selection_${DATASET}_${MODEL}_extreme_a001_ub01_cf10_nowarmup_multiseed}"
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

MANIFEST="${OUTPUT_ROOT}/vi_sparse_selection_multiseed_manifest.csv"
echo "method,dataset,model,selection,sparse_metric,keep_ratio,seed,output_dir,log_file" > "${MANIFEST}"

echo "===== VI sparse-selection multikeep multiseed ====="
echo "DATASET=${DATASET} MODEL=${MODEL} MLP_HIDDEN=${MLP_HIDDEN} RESNET_WIDTH=${RESNET_WIDTH} RESNET_BLOCKS=${RESNET_BLOCKS}"
echo "SEEDS=${SEEDS[*]} KEEP_RATIOS=${KEEP_RATIOS[*]} SELECTIONS=${SELECTIONS[*]}"
echo "NONIID_ALPHA=${NONIID_ALPHA} UNBALANCED_ALPHA=${UNBALANCED_ALPHA} CLIENT_FRACTION=${CLIENT_FRACTION}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "DEVICE=${DEVICE} CLIENT_GPUS=${CLIENT_GPUS_ARG}"

for SEED in "${SEEDS[@]}"; do
  for SELECTION in "${SELECTIONS[@]}"; do
    for KEEP_RATIO in "${KEEP_RATIOS[@]}"; do
      KEEP_LABEL="$(label_keep "${KEEP_RATIO}")"
      RUN_NAME="vi_sparse_${SELECTION}_${MODEL}_keep${KEEP_LABEL}_seed${SEED}"
      RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
      LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
      echo "vi,${DATASET},${MODEL},${SELECTION},${SPARSE_METRIC},${KEEP_RATIO},${SEED},${RUN_DIR},${LOG_FILE}" >> "${MANIFEST}"

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

      MODEL_ARGS=(--mlp_hidden "${MLP_HIDDEN}" --resnet_width "${RESNET_WIDTH}" --resnet_blocks "${RESNET_BLOCKS}")

      echo "===== Running VI ${SELECTION} keep=${KEEP_RATIO} seed=${SEED} dataset=${DATASET} model=${MODEL} ====="
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
        "${MODEL_ARGS[@]}" \
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
        --vi_particles "${VI_PARTICLES}" \
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
done

echo "===== VI sparse-selection multikeep multiseed finished ====="
echo "Manifest: ${MANIFEST}"
