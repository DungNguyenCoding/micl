#!/usr/bin/env bash
set -euo pipefail

# Allow this script to be launched from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"


# Fine-tune VI Bayesian FL on MNIST non-IID/unbalanced.
# Goal: continue from the previous VI sweep where the best setting was at the
# edge of the grid: vi_lr=0.002, batch_size=128, local_epochs=5.
#
# Default full grid:
#   vi_lr:         0.002, 0.003, 0.005
#   batch_size:    128, 256
#   local_epochs:  5, 10
#   vi_prior_scale:0.02, 0.05, 0.10
# Total = 36 runs.
#
# Example:
#   GRID_MODE=full CLIENT_FRACTION=0.05 NUM_ROUNDS=60 FORCE_DEVICE=auto \
#   nohup bash tune_mnist_vi_finetune.sh > logs/tune_mnist_vi_finetune_cf005.log 2>&1 &

mkdir -p logs outputs plots

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# -----------------------------------------------------------------------------
# User-overridable settings
# -----------------------------------------------------------------------------
GRID_MODE="${GRID_MODE:-full}"              # quick | full | extended
CLIENT_FRACTION="${CLIENT_FRACTION:-0.05}" # 0.05 = 15/300 clients per round
NUM_ROUNDS="${NUM_ROUNDS:-60}"
NUM_DEVICES="${NUM_DEVICES:-300}"
NUM_VIRTUAL_CLIENTS="${NUM_VIRTUAL_CLIENTS:-24}"
SEED="${SEED:-42}"
DATASET="${DATASET:-mnist}"
MODEL="${MODEL:-mlp}"
MLP_HIDDEN="${MLP_HIDDEN:-128}"
NONIID_ALPHA="${NONIID_ALPHA:-0.1}"
UNBALANCED_ALPHA="${UNBALANCED_ALPHA:-0.5}"
POSTERIOR_SAMPLE_SCALE="${POSTERIOR_SAMPLE_SCALE:-0.001}"
EVAL_MC_SAMPLES="${EVAL_MC_SAMPLES:-5}"
METRICS_LEVEL="${METRICS_LEVEL:-bayes}"

# Auto CUDA selection. Override with FORCE_DEVICE=cpu/cuda/auto.
FORCE_DEVICE="${FORCE_DEVICE:-auto}"
CLIENT_GPUS="${CLIENT_GPUS:-0.5}"  # VI is heavier than OLA/FedAvg.
CLIENT_CPUS="${CLIENT_CPUS:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TORCH_THREADS="${TORCH_THREADS:-1}"

# If true, call `ray stop -f` between runs. Keep false if any other Flower/Ray
# experiment is running on the same machine.
STOP_RAY_BETWEEN_RUNS="${STOP_RAY_BETWEEN_RUNS:-false}"

ROOT_DIR="${ROOT_DIR:-outputs/tune_vi_mnist_noniid_unbalanced_finetune}"
RESULTS_CSV="${ROOT_DIR}/sweep_results.csv"
RANKING_CSV="${ROOT_DIR}/sweep_ranking.csv"
TOP5_ARGS="${ROOT_DIR}/top5_runs.args"

mkdir -p "${ROOT_DIR}"

# -----------------------------------------------------------------------------
# Auto device logic
# -----------------------------------------------------------------------------
detect_cuda() {
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
  if detect_cuda; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    DEVICE_ARGS=(--device cuda --client_gpus "${CLIENT_GPUS}")
    echo "[device] CUDA available -> CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, client_gpus=${CLIENT_GPUS}"
  else
    DEVICE_ARGS=(--device cpu --client_gpus 0)
    echo "[device] CUDA not available -> CPU"
  fi
fi

python - <<'PY'
import torch
print("[env] torch:", torch.__version__)
print("[env] cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[env] cuda_device_count:", torch.cuda.device_count())
    print("[env] cuda_device_0:", torch.cuda.get_device_name(0))
PY

# -----------------------------------------------------------------------------
# Grid definition
# -----------------------------------------------------------------------------
if [[ "${GRID_MODE}" == "quick" ]]; then
  VI_LRS=(0.002 0.003)
  BATCH_SIZES=(128)
  LOCAL_EPOCHS_LIST=(5)
  VI_PRIOR_SCALES=(0.05)
elif [[ "${GRID_MODE}" == "extended" ]]; then
  VI_LRS=(0.002 0.003 0.005 0.007)
  BATCH_SIZES=(128 256)
  LOCAL_EPOCHS_LIST=(5 10)
  VI_PRIOR_SCALES=(0.01 0.02 0.05 0.10 0.20)
else
  VI_LRS=(0.002 0.003 0.005)
  BATCH_SIZES=(128 256)
  LOCAL_EPOCHS_LIST=(5 10)
  VI_PRIOR_SCALES=(0.02 0.05 0.10)
fi

# Fixed VI settings for this fine-tuning step.
VI_INIT_SCALE="${VI_INIT_SCALE:-0.05}"
VI_MIN_SCALE="${VI_MIN_SCALE:-1e-5}"
VI_PARTICLES="${VI_PARTICLES:-1}"
BAYES_AGGREGATION="${BAYES_AGGREGATION:-product}"

# Common experiment setting. Keep these fixed for fair comparison.
COMMON_ARGS=(
  --dataset "${DATASET}"
  --model "${MODEL}"
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
  --eval_every 1
  --heavy_eval_every 5
  --local_eval_every 5
  --local_eval_fraction 0.2
  --save_posterior_every 10
  --eval_mc_samples "${EVAL_MC_SAMPLES}"
  --posterior_sample_scale "${POSTERIOR_SAMPLE_SCALE}"
  --metrics_level "${METRICS_LEVEL}"

  --client_cpus "${CLIENT_CPUS}"
  --num_workers "${NUM_WORKERS}"
  --torch_threads "${TORCH_THREADS}"
  --seed "${SEED}"
)

stop_ray_if_requested() {
  if [[ "${STOP_RAY_BETWEEN_RUNS}" == "true" ]] && command -v ray >/dev/null 2>&1; then
    ray stop -f >/dev/null 2>&1 || true
  fi
}

sanitize_float() {
  # Convert 0.002 -> 0p002 for folder names.
  echo "$1" | sed 's/\./p/g' | sed 's/-/m/g'
}

init_results_csv() {
  if [[ ! -f "${RESULTS_CSV}" ]]; then
    cat > "${RESULTS_CSV}" <<'CSV'
run_label,status,method,vi_lr,batch_size,local_epochs,vi_prior_scale,vi_init_scale,vi_particles,bayes_aggregation,client_fraction,num_rounds,final_round,final_global_accuracy,final_global_mean_accuracy,final_global_mc_accuracy,final_global_loss,final_global_ece,final_global_nll,final_local_accuracy_weighted,final_posterior_sigma_mean,final_posterior_snr_raw_p50,output_dir
CSV
  fi
}

extract_metric() {
  local csv_path="$1"
  local key="$2"
  python - "$csv_path" "$key" <<'PY'
import csv, sys
path, key = sys.argv[1], sys.argv[2]
try:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("")
    else:
        print(rows[-1].get(key, ""))
except Exception:
    print("")
PY
}

is_completed_run() {
  local out_dir="$1"
  local metrics="${out_dir}/metrics.csv"
  if [[ ! -f "${metrics}" ]]; then
    return 1
  fi
  local final_round
  final_round="$(extract_metric "${metrics}" round)"
  [[ "${final_round}" == "${NUM_ROUNDS}" || "${final_round}" == "${NUM_ROUNDS}.0" ]]
}

append_result_row() {
  local run_label="$1"
  local status="$2"
  local vi_lr="$3"
  local batch_size="$4"
  local local_epochs="$5"
  local prior_scale="$6"
  local out_dir="$7"

  local metrics="${out_dir}/metrics.csv"
  local final_round=""
  local final_global_accuracy=""
  local final_global_mean_accuracy=""
  local final_global_mc_accuracy=""
  local final_global_loss=""
  local final_global_ece=""
  local final_global_nll=""
  local final_local_accuracy_weighted=""
  local final_posterior_sigma_mean=""
  local final_posterior_snr_raw_p50=""

  if [[ -f "${metrics}" ]]; then
    final_round="$(extract_metric "${metrics}" round)"
    final_global_accuracy="$(extract_metric "${metrics}" global_accuracy)"
    final_global_mean_accuracy="$(extract_metric "${metrics}" global_mean_accuracy)"
    final_global_mc_accuracy="$(extract_metric "${metrics}" global_mc_accuracy)"
    final_global_loss="$(extract_metric "${metrics}" global_loss)"
    final_global_ece="$(extract_metric "${metrics}" global_ece)"
    final_global_nll="$(extract_metric "${metrics}" global_nll)"
    final_local_accuracy_weighted="$(extract_metric "${metrics}" local_accuracy_weighted)"
    final_posterior_sigma_mean="$(extract_metric "${metrics}" posterior_sigma_mean)"
    final_posterior_snr_raw_p50="$(extract_metric "${metrics}" posterior_snr_raw_p50)"
  fi

  python - "${RESULTS_CSV}" <<PY
import csv
row = {
    "run_label": "${run_label}",
    "status": "${status}",
    "method": "vi",
    "vi_lr": "${vi_lr}",
    "batch_size": "${batch_size}",
    "local_epochs": "${local_epochs}",
    "vi_prior_scale": "${prior_scale}",
    "vi_init_scale": "${VI_INIT_SCALE}",
    "vi_particles": "${VI_PARTICLES}",
    "bayes_aggregation": "${BAYES_AGGREGATION}",
    "client_fraction": "${CLIENT_FRACTION}",
    "num_rounds": "${NUM_ROUNDS}",
    "final_round": "${final_round}",
    "final_global_accuracy": "${final_global_accuracy}",
    "final_global_mean_accuracy": "${final_global_mean_accuracy}",
    "final_global_mc_accuracy": "${final_global_mc_accuracy}",
    "final_global_loss": "${final_global_loss}",
    "final_global_ece": "${final_global_ece}",
    "final_global_nll": "${final_global_nll}",
    "final_local_accuracy_weighted": "${final_local_accuracy_weighted}",
    "final_posterior_sigma_mean": "${final_posterior_sigma_mean}",
    "final_posterior_snr_raw_p50": "${final_posterior_snr_raw_p50}",
    "output_dir": "${out_dir}",
}
path = "${RESULTS_CSV}"
with open(path, "r", newline="") as f:
    fieldnames = next(csv.reader(f))
with open(path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writerow(row)
PY
}

make_ranking() {
  python - "${RESULTS_CSV}" "${RANKING_CSV}" "${TOP5_ARGS}" <<'PY'
import csv, math, sys
from pathlib import Path

results_path = Path(sys.argv[1])
ranking_path = Path(sys.argv[2])
top5_path = Path(sys.argv[3])

with results_path.open(newline="") as f:
    rows = list(csv.DictReader(f))

# Deduplicate by run_label, keeping the last row if resumed.
latest = {}
for r in rows:
    latest[r.get("run_label", "")] = r
rows = list(latest.values())

def fnum(r, key, default=float("nan")):
    try:
        v = r.get(key, "")
        if v == "":
            return default
        return float(v)
    except Exception:
        return default

ok_rows = [r for r in rows if r.get("status") == "ok"]

# Prefer posterior-mean accuracy if present; otherwise use global_accuracy.
def score(r):
    mean_acc = fnum(r, "final_global_mean_accuracy")
    if not math.isnan(mean_acc):
        return mean_acc
    return fnum(r, "final_global_accuracy", -1.0)

ok_rows.sort(key=lambda r: (score(r), -fnum(r, "final_global_loss", 1e99)), reverse=True)

fieldnames = ["rank", "score_accuracy"] + list(rows[0].keys()) if rows else ["rank", "score_accuracy"]
with ranking_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for i, r in enumerate(ok_rows, start=1):
        out = dict(r)
        out["rank"] = i
        out["score_accuracy"] = score(r)
        writer.writerow(out)

with top5_path.open("w") as f:
    for r in ok_rows[:5]:
        label = r.get("run_label", "run")
        out_dir = r.get("output_dir", "")
        f.write(f"{label}={out_dir}\n")

print(f"[ranking] wrote {ranking_path}")
print(f"[ranking] wrote {top5_path}")
if ok_rows:
    print("[ranking] best:", ok_rows[0].get("run_label"), "acc=", score(ok_rows[0]), "dir=", ok_rows[0].get("output_dir"))
PY
}

run_one() {
  local vi_lr="$1"
  local batch_size="$2"
  local local_epochs="$3"
  local prior_scale="$4"

  local lr_s b_s e_s p_s run_label out_dir run_log
  lr_s="$(sanitize_float "${vi_lr}")"
  b_s="${batch_size}"
  e_s="${local_epochs}"
  p_s="$(sanitize_float "${prior_scale}")"
  run_label="vi_lr${lr_s}_b${b_s}_e${e_s}_prior${p_s}"
  out_dir="${ROOT_DIR}/${run_label}"
  run_log="logs/${run_label}.log"

  echo
  echo "============================================================"
  echo "[run] ${run_label}"
  echo "      vi_lr=${vi_lr}, batch_size=${batch_size}, local_epochs=${local_epochs}, vi_prior_scale=${prior_scale}"
  echo "      output=${out_dir}"
  echo "============================================================"

  if is_completed_run "${out_dir}"; then
    echo "[skip] Completed run exists: ${out_dir}"
    append_result_row "${run_label}" "ok" "${vi_lr}" "${batch_size}" "${local_epochs}" "${prior_scale}" "${out_dir}"
    make_ranking
    return 0
  fi

  rm -rf "${out_dir}"
  stop_ray_if_requested

  set +e
  python main.py \
    --method vi \
    --vi_lr "${vi_lr}" \
    --vi_prior_scale "${prior_scale}" \
    --vi_init_scale "${VI_INIT_SCALE}" \
    --vi_min_scale "${VI_MIN_SCALE}" \
    --vi_particles "${VI_PARTICLES}" \
    --bayes_aggregation "${BAYES_AGGREGATION}" \
    --batch_size "${batch_size}" \
    --local_epochs "${local_epochs}" \
    "${COMMON_ARGS[@]}" \
    "${DEVICE_ARGS[@]}" \
    --output_dir "${out_dir}" \
    > "${run_log}" 2>&1
  local exit_code=$?
  set -e

  if [[ ${exit_code} -eq 0 ]]; then
    append_result_row "${run_label}" "ok" "${vi_lr}" "${batch_size}" "${local_epochs}" "${prior_scale}" "${out_dir}"
  else
    echo "[error] ${run_label} failed with exit code ${exit_code}. See ${run_log}"
    append_result_row "${run_label}" "failed" "${vi_lr}" "${batch_size}" "${local_epochs}" "${prior_scale}" "${out_dir}"
  fi

  make_ranking
}

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
init_results_csv

TOTAL=$(( ${#VI_LRS[@]} * ${#BATCH_SIZES[@]} * ${#LOCAL_EPOCHS_LIST[@]} * ${#VI_PRIOR_SCALES[@]} ))
echo "[grid] GRID_MODE=${GRID_MODE}"
echo "[grid] total_runs=${TOTAL}"
echo "[grid] VI_LRS=${VI_LRS[*]}"
echo "[grid] BATCH_SIZES=${BATCH_SIZES[*]}"
echo "[grid] LOCAL_EPOCHS=${LOCAL_EPOCHS_LIST[*]}"
echo "[grid] VI_PRIOR_SCALES=${VI_PRIOR_SCALES[*]}"
echo "[grid] ROOT_DIR=${ROOT_DIR}"
echo "[grid] CLIENT_FRACTION=${CLIENT_FRACTION}, NUM_ROUNDS=${NUM_ROUNDS}"

SECONDS=0
for vi_lr in "${VI_LRS[@]}"; do
  for batch_size in "${BATCH_SIZES[@]}"; do
    for local_epochs in "${LOCAL_EPOCHS_LIST[@]}"; do
      for prior_scale in "${VI_PRIOR_SCALES[@]}"; do
        run_one "${vi_lr}" "${batch_size}" "${local_epochs}" "${prior_scale}"
      done
    done
  done
done

make_ranking

if command -v ray >/dev/null 2>&1 && [[ "${STOP_RAY_BETWEEN_RUNS}" == "true" ]]; then
  ray stop -f >/dev/null 2>&1 || true
fi

echo
echo "============================================================"
echo "[done] VI fine-tuning finished"
echo "[done] elapsed_seconds=${SECONDS}"
echo "[done] results=${RESULTS_CSV}"
echo "[done] ranking=${RANKING_CSV}"
echo "[done] top5=${TOP5_ARGS}"
echo "============================================================"

head -20 "${RANKING_CSV}" || true
