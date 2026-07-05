#!/usr/bin/env bash
# Hyperparameter sweep for OLA/FOLA Bayesian FL on MNIST non-IID unbalanced.
# Search variables: lr, batch_size, local_epochs.
# Run with:
#   nohup bash tune_mnist_ola.sh > logs/tune_mnist_ola.nohup.log 2>&1 &

set -uo pipefail

mkdir -p logs outputs plots

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ------------------------------------------------------------
# Experiment scale
# ------------------------------------------------------------
NUM_DEVICES="${NUM_DEVICES:-300}"
NUM_VIRTUAL_CLIENTS="${NUM_VIRTUAL_CLIENTS:-24}"
# 0.05 = 15/300 clients per round. This is harder than 0.1 = 30/300.
CLIENT_FRACTION="${CLIENT_FRACTION:-0.05}"
NUM_ROUNDS="${NUM_ROUNDS:-30}"
SEED="${SEED:-42}"

NONIID_ALPHA="${NONIID_ALPHA:-0.1}"
UNBALANCED_ALPHA="${UNBALANCED_ALPHA:-0.5}"
MLP_HIDDEN="${MLP_HIDDEN:-128}"

# Observability: enough for comparison, not too heavy.
METRICS_LEVEL="${METRICS_LEVEL:-bayes}"
EVAL_EVERY="${EVAL_EVERY:-1}"
HEAVY_EVAL_EVERY="${HEAVY_EVAL_EVERY:-5}"
LOCAL_EVAL_EVERY="${LOCAL_EVAL_EVERY:-5}"
LOCAL_EVAL_FRACTION="${LOCAL_EVAL_FRACTION:-0.2}"
SAVE_POSTERIOR_EVERY="${SAVE_POSTERIOR_EVERY:-10}"
EVAL_MC_SAMPLES="${EVAL_MC_SAMPLES:-5}"

# OLA constants. We tune LRS, batch size, and local epochs below.
OLA_PRIOR_LAMBDA="${OLA_PRIOR_LAMBDA:-1.0}"
PRECISION_INIT="${PRECISION_INIT:-1.0}"
PRECISION_FLOOR="${PRECISION_FLOOR:-1e-6}"
FISHER_CLIP="${FISHER_CLIP:-100.0}"
OPTIMIZER="${OPTIMIZER:-sgd}"
MOMENTUM="${MOMENTUM:-0.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"

ROOT_OUT="${ROOT_OUT:-outputs/tune_ola_mnist_noniid_unbalanced}"
ROOT_PLOTS="${ROOT_PLOTS:-plots/tune_ola_mnist_noniid_unbalanced}"
mkdir -p "${ROOT_OUT}" "${ROOT_PLOTS}" logs

# ------------------------------------------------------------
# Auto device selection
# ------------------------------------------------------------
# Overrides:
#   FORCE_DEVICE=cpu
#   FORCE_DEVICE=cuda CUDA_VISIBLE_DEVICES=0 CLIENT_GPUS=0.25
FORCE_DEVICE="${FORCE_DEVICE:-auto}"
CLIENT_GPUS="${CLIENT_GPUS:-0.25}"  # OLA usually lighter than VI.

has_cuda() {
python - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
}

if [[ "${FORCE_DEVICE}" == "cpu" ]]; then
  DEVICE_ARGS=(--device cpu --client_gpus 0)
  echo "[device] CPU forced"
elif [[ "${FORCE_DEVICE}" == "cuda" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  DEVICE_ARGS=(--device cuda --client_gpus "${CLIENT_GPUS}")
  echo "[device] CUDA forced: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, client_gpus=${CLIENT_GPUS}"
else
  if has_cuda; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    DEVICE_ARGS=(--device cuda --client_gpus "${CLIENT_GPUS}")
    echo "[device] CUDA available: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, client_gpus=${CLIENT_GPUS}"
  else
    DEVICE_ARGS=(--device cpu --client_gpus 0)
    echo "[device] CUDA not available, using CPU"
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

# ------------------------------------------------------------
# Hyperparameter grid
# ------------------------------------------------------------
# Default quick grid = 12 runs.
# Full grid example:
#   GRID_MODE=full NUM_ROUNDS=60 nohup bash tune_mnist_ola.sh > logs/tune_mnist_ola.log 2>&1 &
GRID_MODE="${GRID_MODE:-quick}"

if [[ "${GRID_MODE}" == "full" ]]; then
  LRS=(0.1 0.05 0.02 0.01 0.005)
  BATCH_SIZES=(16 32 64 128)
  LOCAL_EPOCHS=(1 2 5)
else
  LRS=(0.05 0.02 0.01)
  BATCH_SIZES=(32 64)
  LOCAL_EPOCHS=(1 2)
fi

COMMON_ARGS=(
  --method ola
  --dataset mnist
  --model mlp
  --iid false
  --balanced false
  --noniid_alpha "${NONIID_ALPHA}"
  --unbalanced_alpha "${UNBALANCED_ALPHA}"
  --num_devices "${NUM_DEVICES}"
  --num_virtual_clients "${NUM_VIRTUAL_CLIENTS}"
  --client_fraction "${CLIENT_FRACTION}"
  --num_rounds "${NUM_ROUNDS}"
  --mlp_hidden "${MLP_HIDDEN}"
  --val_ratio 0.1
  --eval_every "${EVAL_EVERY}"
  --heavy_eval_every "${HEAVY_EVAL_EVERY}"
  --local_eval_every "${LOCAL_EVAL_EVERY}"
  --local_eval_fraction "${LOCAL_EVAL_FRACTION}"
  --save_posterior_every "${SAVE_POSTERIOR_EVERY}"
  --eval_mc_samples "${EVAL_MC_SAMPLES}"
  --metrics_level "${METRICS_LEVEL}"
  --client_cpus 1
  --num_workers 0
  --torch_threads 1
  --seed "${SEED}"
  --ola_prior_lambda "${OLA_PRIOR_LAMBDA}"
  --precision_init "${PRECISION_INIT}"
  --precision_floor "${PRECISION_FLOOR}"
  --fisher_clip "${FISHER_CLIP}"
  --optimizer "${OPTIMIZER}"
  --momentum "${MOMENTUM}"
  --weight_decay "${WEIGHT_DECAY}"
)

safe_tag() {
  echo "$1" | sed 's/-/m/g; s/\./p/g; s/+//g'
}

stop_ray() {
  if command -v ray >/dev/null 2>&1; then
    ray stop -f >/dev/null 2>&1 || true
  fi
}

SWEEP_CSV="${ROOT_OUT}/sweep_results.csv"
RANK_CSV="${ROOT_OUT}/sweep_ranking.csv"
TOP5_ARGS="${ROOT_OUT}/top5_runs.args"

cat > "${SWEEP_CSV}" <<'CSV'
run_label,status,method,lr,batch_size,local_epochs,client_fraction,num_rounds,final_round,final_global_accuracy,final_global_loss,final_global_ece,final_global_nll,final_local_accuracy_weighted,final_posterior_sigma_mean,final_posterior_snr_raw_p50,output_dir
CSV

append_result() {
  local run_label="$1"
  local status="$2"
  local lr="$3"
  local batch_size="$4"
  local local_epochs="$5"
  local out_dir="$6"

  RUN_LABEL="${run_label}" RUN_STATUS="${status}" LR="${lr}" BATCH_SIZE="${batch_size}" LOCAL_EPOCHS_RUN="${local_epochs}" CLIENT_FRACTION_RUN="${CLIENT_FRACTION}" NUM_ROUNDS_RUN="${NUM_ROUNDS}" OUT_DIR="${out_dir}" SWEEP_CSV="${SWEEP_CSV}" python - <<'PY'
import csv, os
out_dir = os.environ["OUT_DIR"]
metrics_path = os.path.join(out_dir, "metrics.csv")
last = {}
if os.path.exists(metrics_path):
    with open(metrics_path, newline="") as f:
        rows = list(csv.DictReader(f))
        if rows:
            last = rows[-1]

def g(name):
    return last.get(name, "")

row = {
    "run_label": os.environ["RUN_LABEL"],
    "status": os.environ["RUN_STATUS"],
    "method": "ola",
    "lr": os.environ["LR"],
    "batch_size": os.environ["BATCH_SIZE"],
    "local_epochs": os.environ["LOCAL_EPOCHS_RUN"],
    "client_fraction": os.environ["CLIENT_FRACTION_RUN"],
    "num_rounds": os.environ["NUM_ROUNDS_RUN"],
    "final_round": g("round"),
    "final_global_accuracy": g("global_accuracy"),
    "final_global_loss": g("global_loss"),
    "final_global_ece": g("global_ece"),
    "final_global_nll": g("global_nll"),
    "final_local_accuracy_weighted": g("local_accuracy_weighted"),
    "final_posterior_sigma_mean": g("posterior_sigma_mean"),
    "final_posterior_snr_raw_p50": g("posterior_snr_raw_p50"),
    "output_dir": out_dir,
}
with open(os.environ["SWEEP_CSV"], "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    writer.writerow(row)
PY
}

run_one() {
  local lr="$1"
  local batch_size="$2"
  local local_epochs="$3"

  local lr_tag
  lr_tag="$(safe_tag "${lr}")"
  local label="ola_lr${lr_tag}_b${batch_size}_e${local_epochs}"
  local out_dir="${ROOT_OUT}/${label}"
  local log_file="logs/${label}.log"

  echo
  echo "============================================================"
  echo "[run] ${label}"
  echo "[out] ${out_dir}"
  echo "[log] ${log_file}"
  echo "============================================================"

  rm -rf "${out_dir}"
  stop_ray

  python main.py \
    "${COMMON_ARGS[@]}" \
    "${DEVICE_ARGS[@]}" \
    --lr "${lr}" \
    --batch_size "${batch_size}" \
    --local_epochs "${local_epochs}" \
    --output_dir "${out_dir}" \
    > "${log_file}" 2>&1

  local status=$?
  if [[ ${status} -eq 0 ]]; then
    echo "[ok] ${label}"
    append_result "${label}" "ok" "${lr}" "${batch_size}" "${local_epochs}" "${out_dir}"
  else
    echo "[fail] ${label}; see ${log_file}"
    append_result "${label}" "fail" "${lr}" "${batch_size}" "${local_epochs}" "${out_dir}"
  fi
}

SECONDS=0
for lr in "${LRS[@]}"; do
  for batch_size in "${BATCH_SIZES[@]}"; do
    for local_epochs in "${LOCAL_EPOCHS[@]}"; do
      run_one "${lr}" "${batch_size}" "${local_epochs}"
    done
  done
done
stop_ray

SWEEP_CSV="${SWEEP_CSV}" RANK_CSV="${RANK_CSV}" TOP5_ARGS="${TOP5_ARGS}" python - <<'PY'
import csv, os, math
sweep = os.environ["SWEEP_CSV"]
rank = os.environ["RANK_CSV"]
top5 = os.environ["TOP5_ARGS"]

def tof(x, default=float("nan")):
    try:
        return float(x)
    except Exception:
        return default

with open(sweep, newline="") as f:
    rows = list(csv.DictReader(f))
rows_ok = [r for r in rows if r.get("status") == "ok" and not math.isnan(tof(r.get("final_global_accuracy")))]
rows_ok.sort(key=lambda r: (-tof(r.get("final_global_accuracy")), tof(r.get("final_global_ece"), 999), tof(r.get("final_global_loss"), 999)))
fields = ["rank"] + (list(rows[0].keys()) if rows else [])
with open(rank, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for i, r in enumerate(rows_ok, 1):
        rr = dict(r)
        rr["rank"] = i
        w.writerow(rr)
with open(top5, "w") as f:
    for r in rows_ok[:5]:
        f.write(f"{r['run_label']}={r['output_dir']}\n")
print("[ranking]", rank)
if rows_ok:
    print("[best]", rows_ok[0])
else:
    print("[best] no successful runs")
PY

if [[ "${MAKE_PLOTS:-true}" == "true" && -s "${TOP5_ARGS}" ]]; then
  mapfile -t RUN_ARGS < "${TOP5_ARGS}"
  python utils.py mix \
    --runs "${RUN_ARGS[@]}" \
    --metrics global_accuracy global_loss global_ece posterior_sigma_mean posterior_snr_raw_p50 \
    --output_dir "${ROOT_PLOTS}/top5"
fi

echo
echo "============================================================"
echo "OLA sweep finished. elapsed_seconds=${SECONDS}"
echo "Results:  ${SWEEP_CSV}"
echo "Ranking:  ${RANK_CSV}"
echo "Top5 args:${TOP5_ARGS}"
echo "Plots:    ${ROOT_PLOTS}/top5"
echo "============================================================"
