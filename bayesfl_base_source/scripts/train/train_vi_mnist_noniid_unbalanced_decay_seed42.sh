#!/usr/bin/env bash
set -euo pipefail

# Stabilized dense VI baseline: MNIST non-IID unbalanced, seed 42.
# Adds LR decay + posterior-scale clamp to reduce late-round VI drift.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs outputs
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

FORCE_DEVICE="${FORCE_DEVICE:-auto}"       # auto | cpu | cuda
CLIENT_GPUS="${CLIENT_GPUS:-0.5}"
SEED="${SEED:-42}"
OUT_DIR="${OUT_DIR:-outputs/vi_mnist_stabilized_decay_seed${SEED}}"

if [[ "${FORCE_DEVICE}" == "cpu" ]]; then
  DEVICE_ARGS=(--device cpu --client_gpus 0)
elif [[ "${FORCE_DEVICE}" == "cuda" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  DEVICE_ARGS=(--device cuda --client_gpus "${CLIENT_GPUS}")
else
  if python - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
  then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    DEVICE_ARGS=(--device cuda --client_gpus "${CLIENT_GPUS}")
  else
    DEVICE_ARGS=(--device cpu --client_gpus 0)
  fi
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
  --num_rounds 200 \
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
  "${DEVICE_ARGS[@]}" \
  --client_cpus 1 \
  --num_workers 0 \
  --torch_threads 1 \
  --seed "${SEED}" \
  --vi_lr 0.005 \
  --vi_lr_decay_milestones 80,120,160 \
  --vi_lr_decay_gamma 0.5 \
  --vi_max_scale 0.5 \
  --vi_prior_scale 0.05 \
  --vi_init_scale 0.05 \
  --vi_min_scale 1e-5 \
  --vi_particles 1 \
  --bayes_aggregation product \
  --batch_size 256 \
  --local_epochs 5 \
  --save_best_checkpoints true \
  --output_dir "${OUT_DIR}"
