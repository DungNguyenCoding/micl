#!/usr/bin/env bash
# Config paths or WRESUME|absolute_run_dir|total_round_cap|total_time_cap_or_none.
# Separate from the historical byte-budget resume grammar.
set -euo pipefail
GPU="${1:?Usage: _run_wireless_queue.sh GPU_ID QUEUE_FILE}"
QUEUE="${2:?Usage: _run_wireless_queue.sh GPU_ID QUEUE_FILE}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY=/home/micl/anaconda3/envs/dungndh/bin/python
export PATH="$(dirname "$PY"):$PATH"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p logs
[[ -f "$QUEUE" ]] || { echo "Missing queue: $QUEUE"; exit 2; }
# Prevent accidentally launching the same queue twice (Linux util-linux flock).
exec 9>"${QUEUE}.lock"
flock -n 9 || { echo "Queue is already running: $QUEUE"; exit 2; }
echo "WIRELESS QUEUE START | GPU=$GPU | QUEUE=$QUEUE"
count=0
while IFS= read -r raw || [[ -n "$raw" ]]; do
    line="${raw%$'\r'}"
    [[ -z "${line//[[:space:]]/}" || "$line" =~ ^[[:space:]]*# ]] && continue
    CMD=("$PY" -m bayesfl.main)
    if [[ "$line" == WRESUME\|* ]]; then
        IFS='|' read -r kind run cap time_cap extra <<< "$line"
        [[ -d "$run" && "$cap" =~ ^[0-9]+$ && -n "$time_cap" && -z "${extra:-}" ]] || { echo "Malformed WRESUME: $line"; exit 2; }
        CMD+=(--resume "$run" --rounds "$cap")
        if [[ "$time_cap" != "none" ]]; then CMD+=(--max-modeled-time-seconds "$time_cap"); fi
        name="$(basename "$run")"
    else
        [[ -f "$line" ]] || { echo "Missing config: $line"; exit 2; }
        "$PY" - "$line" <<'PY'
import sys
from bayesfl.config import load_config
cfg=load_config(sys.argv[1])
if not cfg.wireless_enabled:raise SystemExit('Refusing legacy config in wireless queue')
PY
        CMD+=(--config "$line")
        name="$(basename "$line" .yaml)"
    fi
    stamp="$(date +%Y%m%d_%H%M%S)"
    log="logs/${name}_wireless_gpu${GPU}_${stamp}.log"
    echo "START | $name | $log"
    if "${CMD[@]}" >"$log" 2>&1; then
        echo "SUCCESS | $name"
        tail -n 5 "$log"
    else
        status=$?
        echo "FAILED | $name | status=$status | $log"
        tail -n 60 "$log"
        exit "$status"
    fi
    count=$((count+1))
done < "$QUEUE"
echo "WIRELESS QUEUE FINISHED | GPU=$GPU | completed jobs=$count"
