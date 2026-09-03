#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/paper_env_keep_optim_fola_lambda_sweep_${STAMP}.log"

nohup bash -c '
set -euo pipefail
for L in 0 1 10 100 1000 10000; do
  CFG="scripts/configs/paper_sweep/fola_cifar10_E10_lambda${L}_r20.yaml"
  echo "================================================================"
  echo "START $CFG"
  echo "================================================================"
  python -m bayesfl.main --config "$CFG"
done
python scripts/select_paper_fola_lambda.py
' >"$LOG" 2>&1 &

PID=$!
echo "Started sweep PID=$PID"
echo "Log: $LOG"
echo "After completion: bash scripts/run_fola_cifar10.sh"
