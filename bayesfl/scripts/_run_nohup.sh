#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <config.yaml> [extra bayesfl args...]" >&2
  exit 2
fi

CONFIG="$1"
shift
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
STAMP="$(date +%Y%m%d_%H%M%S)"
NAME="$(basename "$CONFIG" .yaml)"
LOG="logs/${NAME}_${STAMP}.log"

nohup python -m bayesfl.main --config "$CONFIG" "$@" >"$LOG" 2>&1 &
PID=$!
echo "Started PID=$PID"
echo "Log: $LOG"
