#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs outputs/sparse_comm_mnist_noniid_unbalanced

FORCE_DEVICE="${FORCE_DEVICE:-auto}"
CLIENT_GPUS="${CLIENT_GPUS:-0.5}"
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

OUT="outputs/sparse_comm_mnist_noniid_unbalanced/vi_update_snr_prune000_keep100_control_seed42"
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
  --client_cpus 1 \
  --num_workers 0 \
  --torch_threads 1 \
  --seed 42 \
  --vi_lr 0.005 \
  --vi_prior_scale 0.05 \
  --vi_init_scale 0.05 \
  --vi_min_scale 1e-5 \
  --vi_particles 1 \
  --bayes_aggregation product \
  --batch_size 256 \
  --local_epochs 5 \
  --sparse_comm true \
  --sparse_metric update_snr \
  --sparse_ratio 1.0 \
  --sparse_warmup_rounds 20 \
  --sparse_min_keep 100 \
  --save_best_checkpoints true \
  "${DEVICE_ARGS[@]}" \
  --output_dir "${OUT}"
