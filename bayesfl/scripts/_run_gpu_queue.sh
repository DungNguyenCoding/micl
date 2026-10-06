#!/usr/bin/env bash
set -uo pipefail

GPU="${1:?Usage: _run_gpu_queue.sh GPU_ID QUEUE_FILE}"
QUEUE="${2:?Usage: _run_gpu_queue.sh GPU_ID QUEUE_FILE}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY=/home/micl/anaconda3/envs/dungndh/bin/python

export PATH="/home/micl/anaconda3/envs/dungndh/bin:$PATH"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU"

mkdir -p logs

echo "============================================================"
echo "GPU QUEUE STARTED"
echo "GPU   : $GPU"
echo "QUEUE : $QUEUE"
echo "TIME  : $(date)"
echo "============================================================"

while IFS= read -r raw || [[ -n "$raw" ]]; do

    line="${raw%$'\r'}"

    # Skip blank lines and comments.
    [[ -z "${line//[[:space:]]/}" ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue

    stamp="$(date +%Y%m%d_%H%M%S)"

    if [[ "$line" == RESUME\|* ]]; then

        IFS='|' read -r kind run_dir round_cap budget expected_round <<< "$line"

        if [[ "$kind" != "RESUME" || -z "$run_dir" || -z "$round_cap" || -z "$budget" ]]; then
            echo "ERROR: malformed resume queue record:"
            echo "$line"
            exit 2
        fi

        run_dir="$(realpath "$run_dir")"
        run_name="$(basename "$run_dir")"

        CMD=(
            "$PY"
            -m bayesfl.main
            --resume "$run_dir"
            --rounds "$round_cap"
            --max-communication-bytes "$budget"
        )

        CONFIG_LABEL="resume=$run_dir | cap=R$round_cap | budget=$budget | expected=R${expected_round:-?}"

    else

        config="$line"

        if [[ ! -f "$config" ]]; then
            echo "ERROR: config does not exist: $config"
            exit 2
        fi

        run_name="$(
            "$PY" - "$config" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

print(cfg.get("run_name", "unnamed_run"))
PY
        )"

        CMD=(
            "$PY"
            -m bayesfl.main
            --config "$config"
        )

        CONFIG_LABEL="$config"
    fi

    LOG="logs/${run_name}_resumequeue_${stamp}.log"

    echo
    echo "============================================================"
    echo "START"
    echo "GPU      : $GPU"
    echo "RUN      : $run_name"
    echo "CONFIG   : $CONFIG_LABEL"
    echo "TIME     : $(date)"
    echo "============================================================"

    "${CMD[@]}" > "$LOG" 2>&1 &
    pid=$!

    echo "Started PID=$pid"
    echo "LOG      : $LOG"
    echo "Waiting for completion..."

    while kill -0 "$pid" 2>/dev/null; do
        sleep 30

        last="$(
            grep -E \
                'Round [0-9]+ centralized eval|Traceback|ERROR|RuntimeError|ValueError' \
                "$LOG" 2>/dev/null \
                | tail -n 1 \
                || true
        )"

        if [[ -n "$last" ]]; then
            echo "$(date +%H:%M:%S) | $last"
        fi
    done

    wait "$pid"
    rc=$?

    if [[ "$rc" -ne 0 ]]; then
        echo
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "FAILED  : $run_name"
        echo "EXIT    : $rc"
        echo "LOG     : $LOG"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        tail -n 80 "$LOG"
        exit "$rc"
    fi

    echo
    echo "------------------------------------------------------------"
    echo "SUCCESS : $run_name"
    echo "------------------------------------------------------------"

    grep -E 'Round [0-9]+ centralized eval' "$LOG" \
        | tail -n 3 \
        || true

done < "$QUEUE"

echo
echo "============================================================"
echo "GPU QUEUE FINISHED"
echo "GPU   : $GPU"
echo "QUEUE : $QUEUE"
echo "TIME  : $(date)"
echo "============================================================"
