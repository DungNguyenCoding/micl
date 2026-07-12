#!/usr/bin/env bash

# Allow this script to be launched from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Improved OLA/FOLA hyperparameter sweep for MNIST non-IID unbalanced.
#
# This version is designed after fixing OLA evaluation to separate:
#   - global_mean_* : deterministic posterior mean evaluation, theta = mu
#   - global_mc_*   : MC posterior-predictive evaluation with posterior_sample_scale
#
# Main OLA tuning variables:
#   lr, batch_size, local_epochs, ola_prior_lambda, precision_init
#
# Posterior sampling scale is fixed by default because it mainly affects MC
# uncertainty evaluation, not posterior-mean training quality.
#
# Recommended launch:
#   GRID_MODE=full CLIENT_FRACTION=0.05 NUM_ROUNDS=60 FORCE_DEVICE=auto \
#   nohup bash tune_mnist_ola.sh > logs/tune_mnist_ola_fixed_full_cf005.nohup.log 2>&1 &
#
# If running together with VI on a 2-GPU system:
#   CUDA_VISIBLE_DEVICES=1 GRID_MODE=full CLIENT_FRACTION=0.05 NUM_ROUNDS=60 \
#   nohup bash tune_mnist_ola.sh > logs/tune_mnist_ola_fixed_full_cf005_gpu1.nohup.log 2>&1 &

set -uo pipefail

mkdir -p logs outputs plots

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# ------------------------------------------------------------
# Experiment scale
# ------------------------------------------------------------
NUM_DEVICES="${NUM_DEVICES:-300}"
NUM_VIRTUAL_CLIENTS="${NUM_VIRTUAL_CLIENTS:-24}"
CLIENT_FRACTION="${CLIENT_FRACTION:-0.05}"   # 0.05 = 15 / 300 clients per round
NUM_ROUNDS="${NUM_ROUNDS:-60}"
SEED="${SEED:-42}"

NONIID_ALPHA="${NONIID_ALPHA:-0.1}"
UNBALANCED_ALPHA="${UNBALANCED_ALPHA:-0.5}"
MLP_HIDDEN="${MLP_HIDDEN:-128}"
VAL_RATIO="${VAL_RATIO:-0.1}"

# Observability settings.
METRICS_LEVEL="${METRICS_LEVEL:-bayes}"
EVAL_EVERY="${EVAL_EVERY:-1}"
HEAVY_EVAL_EVERY="${HEAVY_EVAL_EVERY:-5}"
LOCAL_EVAL_EVERY="${LOCAL_EVAL_EVERY:-5}"
LOCAL_EVAL_FRACTION="${LOCAL_EVAL_FRACTION:-0.2}"
SAVE_POSTERIOR_EVERY="${SAVE_POSTERIOR_EVERY:-10}"
EVAL_MC_SAMPLES="${EVAL_MC_SAMPLES:-5}"

# OLA constants/tunables.
# posterior_sample_scale mainly affects MC posterior-predictive evaluation.
POSTERIOR_SAMPLE_SCALE="${POSTERIOR_SAMPLE_SCALE:-0.001}"
PRECISION_FLOOR="${PRECISION_FLOOR:-1e-8}"
FISHER_CLIP_DEFAULT="${FISHER_CLIP_DEFAULT:-10.0}"
OPTIMIZER="${OPTIMIZER:-sgd}"
MOMENTUM="${MOMENTUM:-0.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"

# Use a new output root so old invalid OLA sweeps are not overwritten/mixed.
ROOT_OUT="${ROOT_OUT:-outputs/tune_ola_mnist_noniid_unbalanced_fixed}"
ROOT_PLOTS="${ROOT_PLOTS:-plots/tune_ola_mnist_noniid_unbalanced_fixed}"
mkdir -p "${ROOT_OUT}" "${ROOT_PLOTS}" logs

# Do not stop Ray by default because you may run VI and OLA sweeps concurrently.
# If running only one sweep and you want to clean stale Ray sessions between runs:
#   STOP_RAY_BETWEEN=true bash tune_mnist_ola.sh
STOP_RAY_BETWEEN="${STOP_RAY_BETWEEN:-false}"
STOP_RAY_AT_END="${STOP_RAY_AT_END:-false}"

# Skip runs that already have metrics.csv reaching NUM_ROUNDS.
SKIP_EXISTING="${SKIP_EXISTING:-true}"
MAKE_PLOTS="${MAKE_PLOTS:-true}"
TOP_K="${TOP_K:-5}"

# ------------------------------------------------------------
# Auto device selection
# ------------------------------------------------------------
# Overrides:
#   FORCE_DEVICE=cpu
#   FORCE_DEVICE=cuda CUDA_VISIBLE_DEVICES=0 CLIENT_GPUS=0.25
#   FORCE_DEVICE=auto  # default; use CUDA if torch.cuda.is_available()
FORCE_DEVICE="${FORCE_DEVICE:-auto}"
CLIENT_GPUS="${CLIENT_GPUS:-0.25}"

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
# Override any list with comma-separated values, for example:
#   LRS_CSV="0.005,0.002" BATCH_SIZES_CSV="32,64" bash tune_mnist_ola.sh
#
# quick: small sanity grid
# full: recommended OLA grid after mean/MC evaluation fix
# research: larger grid, slower
GRID_MODE="${GRID_MODE:-quick}"

csv_to_array() {
  local csv="$1"
  local -n arr_ref="$2"
  IFS=',' read -r -a arr_ref <<< "${csv}"
}

if [[ "${GRID_MODE}" == "research" ]]; then
  LRS_DEFAULT=(0.005 0.002 0.001 0.0005)
  BATCH_SIZES_DEFAULT=(16 32 64)
  LOCAL_EPOCHS_DEFAULT=(1 2)
  OLA_PRIOR_LAMBDAS_DEFAULT=(0.001 0.005 0.01 0.05 0.1)
  PRECISION_INITS_DEFAULT=(0.0001 0.001 0.01 0.1)
  FISHER_CLIPS_DEFAULT=(1.0 10.0)
elif [[ "${GRID_MODE}" == "full" ]]; then
  # 3 * 2 * 2 * 3 * 2 * 1 = 72 runs by default.
  LRS_DEFAULT=(0.005 0.002 0.001)
  BATCH_SIZES_DEFAULT=(32 64)
  LOCAL_EPOCHS_DEFAULT=(1 2)
  OLA_PRIOR_LAMBDAS_DEFAULT=(0.001 0.01 0.05)
  PRECISION_INITS_DEFAULT=(0.001 0.01)
  FISHER_CLIPS_DEFAULT=(${FISHER_CLIP_DEFAULT})
else
  # 2 * 1 * 1 * 2 * 2 * 1 = 8 runs by default.
  LRS_DEFAULT=(0.005 0.002)
  BATCH_SIZES_DEFAULT=(32)
  LOCAL_EPOCHS_DEFAULT=(1)
  OLA_PRIOR_LAMBDAS_DEFAULT=(0.001 0.01)
  PRECISION_INITS_DEFAULT=(0.001 0.01)
  FISHER_CLIPS_DEFAULT=(${FISHER_CLIP_DEFAULT})
fi

if [[ -n "${LRS_CSV:-}" ]]; then csv_to_array "${LRS_CSV}" LRS; else LRS=("${LRS_DEFAULT[@]}"); fi
if [[ -n "${BATCH_SIZES_CSV:-}" ]]; then csv_to_array "${BATCH_SIZES_CSV}" BATCH_SIZES; else BATCH_SIZES=("${BATCH_SIZES_DEFAULT[@]}"); fi
if [[ -n "${LOCAL_EPOCHS_CSV:-}" ]]; then csv_to_array "${LOCAL_EPOCHS_CSV}" LOCAL_EPOCHS; else LOCAL_EPOCHS=("${LOCAL_EPOCHS_DEFAULT[@]}"); fi
if [[ -n "${OLA_PRIOR_LAMBDAS_CSV:-}" ]]; then csv_to_array "${OLA_PRIOR_LAMBDAS_CSV}" OLA_PRIOR_LAMBDAS; else OLA_PRIOR_LAMBDAS=("${OLA_PRIOR_LAMBDAS_DEFAULT[@]}"); fi
if [[ -n "${PRECISION_INITS_CSV:-}" ]]; then csv_to_array "${PRECISION_INITS_CSV}" PRECISION_INITS; else PRECISION_INITS=("${PRECISION_INITS_DEFAULT[@]}"); fi
if [[ -n "${FISHER_CLIPS_CSV:-}" ]]; then csv_to_array "${FISHER_CLIPS_CSV}" FISHER_CLIPS; else FISHER_CLIPS=("${FISHER_CLIPS_DEFAULT[@]}"); fi

TOTAL_RUNS=$(( ${#LRS[@]} * ${#BATCH_SIZES[@]} * ${#LOCAL_EPOCHS[@]} * ${#OLA_PRIOR_LAMBDAS[@]} * ${#PRECISION_INITS[@]} * ${#FISHER_CLIPS[@]} ))

echo "[grid] GRID_MODE=${GRID_MODE}; total_runs=${TOTAL_RUNS}"
echo "[grid] LRS=${LRS[*]}"
echo "[grid] BATCH_SIZES=${BATCH_SIZES[*]}"
echo "[grid] LOCAL_EPOCHS=${LOCAL_EPOCHS[*]}"
echo "[grid] OLA_PRIOR_LAMBDAS=${OLA_PRIOR_LAMBDAS[*]}"
echo "[grid] PRECISION_INITS=${PRECISION_INITS[*]}"
echo "[grid] FISHER_CLIPS=${FISHER_CLIPS[*]}"
echo "[grid] POSTERIOR_SAMPLE_SCALE=${POSTERIOR_SAMPLE_SCALE}"

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
  --val_ratio "${VAL_RATIO}"
  --eval_every "${EVAL_EVERY}"
  --heavy_eval_every "${HEAVY_EVAL_EVERY}"
  --local_eval_every "${LOCAL_EVAL_EVERY}"
  --local_eval_fraction "${LOCAL_EVAL_FRACTION}"
  --save_posterior_every "${SAVE_POSTERIOR_EVERY}"
  --eval_mc_samples "${EVAL_MC_SAMPLES}"
  --posterior_sample_scale "${POSTERIOR_SAMPLE_SCALE}"
  --metrics_level "${METRICS_LEVEL}"
  --client_cpus 1
  --num_workers 0
  --torch_threads 1
  --seed "${SEED}"
  --precision_floor "${PRECISION_FLOOR}"
  --optimizer "${OPTIMIZER}"
  --momentum "${MOMENTUM}"
  --weight_decay "${WEIGHT_DECAY}"
)

safe_tag() {
  echo "$1" | sed 's/-/m/g; s/\./p/g; s/+//g; s/e-/em/g; s/E-/em/g'
}

stop_ray() {
  if [[ "${STOP_RAY_BETWEEN}" == "true" ]] && command -v ray >/dev/null 2>&1; then
    ray stop -f >/dev/null 2>&1 || true
  fi
}

is_completed() {
  local out_dir="$1"
  local metrics="${out_dir}/metrics.csv"
  [[ -f "${metrics}" ]] || return 1
  python - "${metrics}" "${NUM_ROUNDS}" <<'PY'
import csv, sys
metrics, rounds = sys.argv[1], int(float(sys.argv[2]))
try:
    with open(metrics, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(1)
    last_round = int(float(rows[-1].get("round", -1)))
    raise SystemExit(0 if last_round >= rounds else 1)
except Exception:
    raise SystemExit(1)
PY
}

SWEEP_CSV="${ROOT_OUT}/sweep_results.csv"
RANK_CSV="${ROOT_OUT}/sweep_ranking.csv"
TOPK_ARGS="${ROOT_OUT}/top${TOP_K}_runs.args"

cat > "${SWEEP_CSV}" <<'CSV'
run_label,status,method,lr,batch_size,local_epochs,ola_prior_lambda,precision_init,precision_floor,fisher_clip,posterior_sample_scale,client_fraction,num_rounds,final_round,final_global_accuracy,final_global_mean_accuracy,final_global_mc_accuracy,final_global_loss,final_global_mean_loss,final_global_mc_loss,final_global_ece,final_global_mean_ece,final_global_mc_ece,final_global_nll,final_local_accuracy_weighted,final_posterior_sigma_mean,final_posterior_snr_raw_p50,output_dir
CSV

append_result() {
  local run_label="$1"
  local status="$2"
  local lr="$3"
  local batch_size="$4"
  local local_epochs="$5"
  local ola_prior_lambda="$6"
  local precision_init="$7"
  local fisher_clip="$8"
  local out_dir="$9"

  RUN_LABEL="${run_label}" \
  RUN_STATUS="${status}" \
  LR="${lr}" \
  BATCH_SIZE="${batch_size}" \
  LOCAL_EPOCHS_RUN="${local_epochs}" \
  OLA_PRIOR_LAMBDA_RUN="${ola_prior_lambda}" \
  PRECISION_INIT_RUN="${precision_init}" \
  PRECISION_FLOOR_RUN="${PRECISION_FLOOR}" \
  FISHER_CLIP_RUN="${fisher_clip}" \
  POSTERIOR_SAMPLE_SCALE_RUN="${POSTERIOR_SAMPLE_SCALE}" \
  CLIENT_FRACTION_RUN="${CLIENT_FRACTION}" \
  NUM_ROUNDS_RUN="${NUM_ROUNDS}" \
  OUT_DIR="${out_dir}" \
  SWEEP_CSV="${SWEEP_CSV}" python - <<'PY'
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

def g_first(*names):
    for n in names:
        v = g(n)
        if v not in ("", None):
            return v
    return ""

row = {
    "run_label": os.environ["RUN_LABEL"],
    "status": os.environ["RUN_STATUS"],
    "method": "ola",
    "lr": os.environ["LR"],
    "batch_size": os.environ["BATCH_SIZE"],
    "local_epochs": os.environ["LOCAL_EPOCHS_RUN"],
    "ola_prior_lambda": os.environ["OLA_PRIOR_LAMBDA_RUN"],
    "precision_init": os.environ["PRECISION_INIT_RUN"],
    "precision_floor": os.environ["PRECISION_FLOOR_RUN"],
    "fisher_clip": os.environ["FISHER_CLIP_RUN"],
    "posterior_sample_scale": os.environ["POSTERIOR_SAMPLE_SCALE_RUN"],
    "client_fraction": os.environ["CLIENT_FRACTION_RUN"],
    "num_rounds": os.environ["NUM_ROUNDS_RUN"],
    "final_round": g("round"),
    "final_global_accuracy": g("global_accuracy"),
    "final_global_mean_accuracy": g_first("global_mean_accuracy", "global_accuracy"),
    "final_global_mc_accuracy": g("global_mc_accuracy"),
    "final_global_loss": g("global_loss"),
    "final_global_mean_loss": g_first("global_mean_loss", "global_loss"),
    "final_global_mc_loss": g("global_mc_loss"),
    "final_global_ece": g("global_ece"),
    "final_global_mean_ece": g_first("global_mean_ece", "global_ece"),
    "final_global_mc_ece": g("global_mc_ece"),
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
  local ola_prior_lambda="$4"
  local precision_init="$5"
  local fisher_clip="$6"

  local lr_tag lam_tag pinit_tag fclip_tag
  lr_tag="$(safe_tag "${lr}")"
  lam_tag="$(safe_tag "${ola_prior_lambda}")"
  pinit_tag="$(safe_tag "${precision_init}")"
  fclip_tag="$(safe_tag "${fisher_clip}")"

  local label="ola_lr${lr_tag}_b${batch_size}_e${local_epochs}_lam${lam_tag}_pinit${pinit_tag}_fclip${fclip_tag}"
  local out_dir="${ROOT_OUT}/${label}"
  local log_file="logs/${label}.log"

  CURRENT_RUN=$((CURRENT_RUN + 1))
  echo
  echo "============================================================"
  echo "[run ${CURRENT_RUN}/${TOTAL_RUNS}] ${label}"
  echo "[out] ${out_dir}"
  echo "[log] ${log_file}"
  echo "============================================================"

  if [[ "${SKIP_EXISTING}" == "true" ]] && is_completed "${out_dir}"; then
    echo "[skip-existing] ${label}"
    append_result "${label}" "ok" "${lr}" "${batch_size}" "${local_epochs}" "${ola_prior_lambda}" "${precision_init}" "${fisher_clip}" "${out_dir}"
    return 0
  fi

  rm -rf "${out_dir}"
  stop_ray

  python main.py \
    "${COMMON_ARGS[@]}" \
    "${DEVICE_ARGS[@]}" \
    --lr "${lr}" \
    --batch_size "${batch_size}" \
    --local_epochs "${local_epochs}" \
    --ola_prior_lambda "${ola_prior_lambda}" \
    --precision_init "${precision_init}" \
    --fisher_clip "${fisher_clip}" \
    --output_dir "${out_dir}" \
    > "${log_file}" 2>&1

  local status=$?
  if [[ ${status} -eq 0 ]]; then
    echo "[ok] ${label}"
    append_result "${label}" "ok" "${lr}" "${batch_size}" "${local_epochs}" "${ola_prior_lambda}" "${precision_init}" "${fisher_clip}" "${out_dir}"
  else
    echo "[fail] ${label}; see ${log_file}"
    append_result "${label}" "fail" "${lr}" "${batch_size}" "${local_epochs}" "${ola_prior_lambda}" "${precision_init}" "${fisher_clip}" "${out_dir}"
  fi
}

SECONDS=0
CURRENT_RUN=0

for lr in "${LRS[@]}"; do
  for batch_size in "${BATCH_SIZES[@]}"; do
    for local_epochs in "${LOCAL_EPOCHS[@]}"; do
      for ola_prior_lambda in "${OLA_PRIOR_LAMBDAS[@]}"; do
        for precision_init in "${PRECISION_INITS[@]}"; do
          for fisher_clip in "${FISHER_CLIPS[@]}"; do
            run_one "${lr}" "${batch_size}" "${local_epochs}" "${ola_prior_lambda}" "${precision_init}" "${fisher_clip}"
          done
        done
      done
    done
  done
done

if [[ "${STOP_RAY_AT_END}" == "true" ]] && command -v ray >/dev/null 2>&1; then
  ray stop -f >/dev/null 2>&1 || true
fi

SWEEP_CSV="${SWEEP_CSV}" RANK_CSV="${RANK_CSV}" TOPK_ARGS="${TOPK_ARGS}" TOP_K="${TOP_K}" python - <<'PY'
import csv, os, math
sweep = os.environ["SWEEP_CSV"]
rank = os.environ["RANK_CSV"]
topk_path = os.environ["TOPK_ARGS"]
top_k = int(os.environ.get("TOP_K", "5"))

def tof(x, default=float("nan")):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default

with open(sweep, newline="") as f:
    rows = list(csv.DictReader(f))

rows_ok = []
for r in rows:
    if r.get("status") != "ok":
        continue
    score = tof(r.get("final_global_mean_accuracy") or r.get("final_global_accuracy"))
    if math.isnan(score):
        continue
    rows_ok.append(r)

# Main ranking: posterior-mean accuracy descending, then mean ECE ascending, then mean loss ascending.
rows_ok.sort(
    key=lambda r: (
        -tof(r.get("final_global_mean_accuracy") or r.get("final_global_accuracy"), -1),
        tof(r.get("final_global_mean_ece") or r.get("final_global_ece"), 999),
        tof(r.get("final_global_mean_loss") or r.get("final_global_loss"), 999),
    )
)

fields = ["rank"] + (list(rows[0].keys()) if rows else [])
with open(rank, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for i, r in enumerate(rows_ok, 1):
        rr = dict(r)
        rr["rank"] = i
        w.writerow(rr)

with open(topk_path, "w") as f:
    for r in rows_ok[:top_k]:
        f.write(f"{r['run_label']}={r['output_dir']}\n")

print("[ranking]", rank)
if rows_ok:
    print("[best]", rows_ok[0])
else:
    print("[best] no successful runs")
PY

if [[ "${MAKE_PLOTS}" == "true" && -s "${TOPK_ARGS}" ]]; then
  mapfile -t RUN_ARGS < "${TOPK_ARGS}"
  python utils.py mix \
    --runs "${RUN_ARGS[@]}" \
    --metrics \
      global_accuracy \
      global_mean_accuracy \
      global_mc_accuracy \
      global_loss \
      global_mean_loss \
      global_mc_loss \
      global_ece \
      global_mean_ece \
      global_mc_ece \
      posterior_sigma_mean \
      posterior_snr_raw_p50 \
    --output_dir "${ROOT_PLOTS}/top${TOP_K}"
fi

echo
echo "============================================================"
echo "Improved OLA sweep finished. elapsed_seconds=${SECONDS}"
echo "Results:    ${SWEEP_CSV}"
echo "Ranking:    ${RANK_CSV}"
echo "Top args:   ${TOPK_ARGS}"
echo "Plots:      ${ROOT_PLOTS}/top${TOP_K}"
echo "============================================================"
