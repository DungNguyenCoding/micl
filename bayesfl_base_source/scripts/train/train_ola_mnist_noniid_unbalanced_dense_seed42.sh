#!/usr/bin/env bash
set -euo pipefail

# Allow this script to be launched from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"


# OLA/FOLA final comparison run: MNIST / non-IID / unbalanced
# Same experimental environment as FedAvg/VI comparison, but with OLA tuned hyperparameters.
# Run:
#   nohup bash ola_mnist_noniid_unbalanced.sh > logs/ola_mnist_noniid_unbalanced.log 2>&1 &
# Optional multi-seed:
#   SEEDS="42 43 44" nohup bash ola_mnist_noniid_unbalanced.sh > logs/ola_mnist_noniid_unbalanced.log 2>&1 &

mkdir -p logs outputs/final_compare_mnist_noniid_unbalanced

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# -----------------------------
# Auto device selection
# -----------------------------
FORCE_DEVICE="${FORCE_DEVICE:-auto}"       # auto | cpu | cuda
CLIENT_GPUS="${CLIENT_GPUS:-0.25}"

cuda_available() {
python - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
}

if [[ "${FORCE_DEVICE}" == "cpu" ]]; then
  DEVICE_ARGS=(--device cpu --client_gpus 0)
  echo "[device] FORCE_DEVICE=cpu -> CPU"
elif [[ "${FORCE_DEVICE}" == "cuda" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  DEVICE_ARGS=(--device cuda --client_gpus "${CLIENT_GPUS}")
  echo "[device] FORCE_DEVICE=cuda -> CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, client_gpus=${CLIENT_GPUS}"
else
  if cuda_available; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    DEVICE_ARGS=(--device cuda --client_gpus "${CLIENT_GPUS}")
    echo "[device] CUDA available -> CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, client_gpus=${CLIENT_GPUS}"
  else
    DEVICE_ARGS=(--device cpu --client_gpus 0)
    echo "[device] CUDA unavailable -> CPU"
  fi
fi

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device_count:", torch.cuda.device_count())
    print("cuda_device_0:", torch.cuda.get_device_name(0))
PY

# -----------------------------
# Shared comparison setting
# -----------------------------
SEEDS="${SEEDS:-42}"
NUM_ROUNDS="${NUM_ROUNDS:-200}"
BASE_OUT="${BASE_OUT:-outputs/final_compare_mnist_noniid_unbalanced}"

COMMON_ARGS=(
  --dataset mnist
  --model mlp
  --iid false
  --balanced false
  --noniid_alpha 0.1
  --unbalanced_alpha 0.5
  --num_devices 300
  --num_virtual_clients 24
  --client_fraction 0.05
  --num_rounds "${NUM_ROUNDS}"
  --mlp_hidden 128
  --val_ratio 0.1
  --eval_every 1
  --heavy_eval_every 10
  --local_eval_every 10
  --local_eval_fraction 0.2
  --save_posterior_every 20
  --eval_mc_samples 5
  --metrics_level bayes
  --client_cpus 1
  --num_workers 0
  --torch_threads 1
)

# -----------------------------
# OLA tuned hyperparameters
# Based on fixed OLA sweep best region.
# Main comparison metric should be global_mean_accuracy/global_accuracy.
# -----------------------------
OLA_ARGS=(
  --method ola
  --optimizer sgd
  --lr 0.005
  --batch_size 32
  --local_epochs 2
  --ola_prior_lambda 0.05
  --precision_init 0.001
  --precision_floor 1e-8
  --fisher_clip 10.0
  --posterior_sample_scale 0.001
)

stop_ray() {
  if command -v ray >/dev/null 2>&1; then
    ray stop -f >/dev/null 2>&1 || true
  fi
}

for SEED in ${SEEDS}; do
  OUT_DIR="${BASE_OUT}/ola_seed${SEED}"
  echo
  echo "============================================================"
  echo "OLA/FOLA final comparison | seed=${SEED} | rounds=${NUM_ROUNDS}"
  echo "Output: ${OUT_DIR}"
  echo "============================================================"

  rm -rf "${OUT_DIR}"
  stop_ray

  python main.py \
    "${OLA_ARGS[@]}" \
    "${COMMON_ARGS[@]}" \
    "${DEVICE_ARGS[@]}" \
    --seed "${SEED}" \
    --output_dir "${OUT_DIR}"

  echo "[done] ${OUT_DIR}"
  tail -n 3 "${OUT_DIR}/metrics.csv" || true
done

stop_ray

echo
echo "OLA/FOLA final comparison finished."
echo "Outputs: ${BASE_OUT}/ola_seed*"
