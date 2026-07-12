#!/usr/bin/env bash
set -euo pipefail

# Allow this script to be launched from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"


mkdir -p logs outputs/final_compare_mnist_noniid_unbalanced

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ------------------------------------------------------------
# Final comparison: VI Bayesian FL on MNIST non-IID unbalanced
# Same experiment setting as FedAvg/OLA final comparison.
#
# Best VI fine-tune result used:
#   vi_lr=0.005, batch_size=256, local_epochs=5, vi_prior_scale=0.05
#
# Optional overrides:
#   SEED=43 bash vi_mnist_noniid_unbalanced.sh
#   FORCE_DEVICE=cpu bash vi_mnist_noniid_unbalanced.sh
#   FORCE_DEVICE=cuda CUDA_VISIBLE_DEVICES=0 CLIENT_GPUS=0.5 bash vi_mnist_noniid_unbalanced.sh
# ------------------------------------------------------------

FORCE_DEVICE="${FORCE_DEVICE:-auto}"
CLIENT_GPUS="${CLIENT_GPUS:-0.5}"
SEED="${SEED:-42}"

detect_cuda() {
python - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
}

if [[ "${FORCE_DEVICE}" == "cpu" ]]; then
  DEVICE_ARGS=(--device cpu --client_gpus 0)
  echo "[device] FORCE_DEVICE=cpu. Using CPU."
elif [[ "${FORCE_DEVICE}" == "cuda" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  DEVICE_ARGS=(--device cuda --client_gpus "${CLIENT_GPUS}")
  echo "[device] FORCE_DEVICE=cuda. Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, client_gpus=${CLIENT_GPUS}."
else
  if detect_cuda; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    DEVICE_ARGS=(--device cuda --client_gpus "${CLIENT_GPUS}")
    echo "[device] CUDA available. Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, client_gpus=${CLIENT_GPUS}."
  else
    DEVICE_ARGS=(--device cpu --client_gpus 0)
    echo "[device] CUDA not available. Using CPU."
  fi
fi

echo "[env] Python executable: $(which python)"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device_count:", torch.cuda.device_count())
    print("cuda_device_0:", torch.cuda.get_device_name(0))
PY

OUT_DIR="outputs/final_compare_mnist_noniid_unbalanced/vi_seed${SEED}"

echo
echo "============================================================"
echo " Final comparison run: VI Bayesian FL"
echo " Output: ${OUT_DIR}"
echo " Seed: ${SEED}"
echo "============================================================"

if command -v ray >/dev/null 2>&1; then
  ray stop -f >/dev/null 2>&1 || true
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
  --metrics_level bayes \
  --client_cpus 1 \
  --num_workers 0 \
  --torch_threads 1 \
  --seed "${SEED}" \
  --vi_lr 0.005 \
  --batch_size 256 \
  --local_epochs 5 \
  --vi_prior_scale 0.05 \
  --vi_init_scale 0.05 \
  --vi_min_scale 1e-5 \
  --vi_particles 1 \
  --bayes_aggregation product \
  "${DEVICE_ARGS[@]}" \
  --output_dir "${OUT_DIR}"

if command -v ray >/dev/null 2>&1; then
  ray stop -f >/dev/null 2>&1 || true
fi

echo
echo "============================================================"
echo " VI final comparison finished"
echo " Output: ${OUT_DIR}"
echo "============================================================"

tail -n 5 "${OUT_DIR}/metrics.csv" || true
