#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs outputs plots

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ------------------------------------------------------------
# Auto device selection
# ------------------------------------------------------------
# Default behavior:
#   - use CUDA if torch.cuda.is_available() == True
#   - otherwise use CPU
#
# Optional overrides:
#   FORCE_DEVICE=cpu  nohup bash run_smoke_all.sh > logs/smoke_all.log 2>&1 &
#   FORCE_DEVICE=cuda CLIENT_GPUS=0.5 nohup bash run_smoke_all.sh > logs/smoke_all.log 2>&1 &
#   CUDA_VISIBLE_DEVICES=1 CLIENT_GPUS=0.25 nohup bash run_smoke_all.sh > logs/smoke_all.log 2>&1 &
# ------------------------------------------------------------

FORCE_DEVICE="${FORCE_DEVICE:-auto}"
CLIENT_GPUS="${CLIENT_GPUS:-0.25}"

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

echo "[env] Python executable:"
which python

echo "[env] Torch/CUDA info:"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device_count:", torch.cuda.device_count())
    print("cuda_device_0:", torch.cuda.get_device_name(0))
PY

# ------------------------------------------------------------
# Common smoke-test arguments
# ------------------------------------------------------------
# Small settings are intentional:
#   - MNIST
#   - small MLP
#   - 30 physical devices
#   - 6 Flower virtual clients
#   - 3 communication rounds
#   - full observability enabled
# ------------------------------------------------------------

COMMON_ARGS=(
  --dataset mnist
  --model mlp
  --iid false
  --balanced false
  --noniid_alpha 0.3
  --unbalanced_alpha 0.5

  --num_devices 30
  --num_virtual_clients 6
  --client_fraction 0.2

  --num_rounds 3
  --local_epochs 1
  --batch_size 32
  --mlp_hidden 64
  --val_ratio 0.1

  --eval_every 1
  --heavy_eval_every 1
  --local_eval_every 1
  --local_eval_fraction 1.0
  --save_posterior_every 1
  --eval_mc_samples 3
  --metrics_level full

  --client_cpus 1
  --num_workers 0
  --torch_threads 1
  --seed 42
)

stop_ray() {
  if command -v ray >/dev/null 2>&1; then
    ray stop -f >/dev/null 2>&1 || true
  fi
}

check_output() {
  local out_dir="$1"

  echo "[check] Checking ${out_dir}"

  test -f "${out_dir}/metrics.csv"
  test -f "${out_dir}/run_summary.csv"
  test -f "${out_dir}/client_data_summary.csv"
  test -f "${out_dir}/device_summary.csv"
  test -f "${out_dir}/selection_summary.csv"
  test -f "${out_dir}/selected_clients.csv"
  test -f "${out_dir}/client_train_metrics.csv"
  test -f "${out_dir}/client_eval_metrics.csv"
  test -f "${out_dir}/calibration_bins.csv"
  test -f "${out_dir}/posterior_summary.csv"
  test -f "${out_dir}/snr_histograms.csv"
  test -f "${out_dir}/aggregation_diagnostics.csv"
  test -f "${out_dir}/communication_metrics.csv"
  test -f "${out_dir}/final_model.pt"

  echo "[check] OK: ${out_dir}"
  ls -lh "${out_dir}" | head -40
}

run_fedavg() {
  local out_dir="outputs/smoke_fedavg_mnist"

  echo
  echo "============================================================"
  echo " Smoke test 1/3: FedAvg"
  echo " Output: ${out_dir}"
  echo "============================================================"

  rm -rf "${out_dir}"
  stop_ray

  python main.py \
    --method fedavg \
    --lr 0.05 \
    "${COMMON_ARGS[@]}" \
    "${DEVICE_ARGS[@]}" \
    --output_dir "${out_dir}"

  check_output "${out_dir}"
}

run_ola() {
  local out_dir="outputs/smoke_ola_mnist"

  echo
  echo "============================================================"
  echo " Smoke test 2/3: OLA/FOLA Bayesian FL"
  echo " Output: ${out_dir}"
  echo "============================================================"

  rm -rf "${out_dir}"
  stop_ray

  python main.py \
    --method ola \
    --lr 0.02 \
    --ola_prior_lambda 1.0 \
    --precision_init 1.0 \
    "${COMMON_ARGS[@]}" \
    "${DEVICE_ARGS[@]}" \
    --output_dir "${out_dir}"

  check_output "${out_dir}"
}

run_vi() {
  local out_dir="outputs/smoke_vi_mnist"

  echo
  echo "============================================================"
  echo " Smoke test 3/3: VI Bayesian FL"
  echo " Output: ${out_dir}"
  echo "============================================================"

  rm -rf "${out_dir}"
  stop_ray

  python main.py \
    --method vi \
    --vi_lr 0.001 \
    --vi_prior_scale 0.05 \
    --vi_init_scale 0.05 \
    --vi_particles 1 \
    --bayes_aggregation product \
    "${COMMON_ARGS[@]}" \
    "${DEVICE_ARGS[@]}" \
    --output_dir "${out_dir}"

  check_output "${out_dir}"
}

# ------------------------------------------------------------
# Run all smoke tests
# ------------------------------------------------------------

SECONDS=0

run_fedavg
run_ola
run_vi

stop_ray

echo
echo "============================================================"
echo " All smoke tests finished successfully"
echo " Total elapsed seconds: ${SECONDS}"
echo "============================================================"

echo
echo "Generated output folders:"
du -sh outputs/smoke_fedavg_mnist outputs/smoke_ola_mnist outputs/smoke_vi_mnist

echo
echo "Next: generate plots with:"
echo "  nohup bash plot_smoke_all.sh > logs/plot_smoke_all.log 2>&1 &"
