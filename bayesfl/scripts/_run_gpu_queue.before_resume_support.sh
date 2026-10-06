#!/usr/bin/env bash

set -u

GPU="$1"
QUEUE="$2"

if [ ! -f "$QUEUE" ]; then
    echo "Queue does not exist: $QUEUE"
    exit 1
fi

echo "============================================================"
echo "GPU QUEUE STARTED"
echo "GPU   : $GPU"
echo "QUEUE : $QUEUE"
echo "TIME  : $(date)"
echo "============================================================"

while IFS= read -r CFG
do
    # Ignore blank lines/comments
    [ -z "$CFG" ] && continue
    [[ "$CFG" =~ ^# ]] && continue

    if [ ! -f "$CFG" ]; then
        echo
        echo "ERROR: missing config: $CFG"
        continue
    fi

    RUN_NAME="$(
        python - "$CFG" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

print(cfg["run_name"])
PY
    )"

    echo
    echo "============================================================"
    echo "START"
    echo "GPU      : $GPU"
    echo "RUN      : $RUN_NAME"
    echo "CONFIG   : $CFG"
    echo "TIME     : $(date)"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES="$GPU" \
        bash scripts/_run_nohup.sh "$CFG"

    # Give _run_nohup.sh time to create the log/process.
    sleep 5

    LOG="$(
        ls -t logs/${RUN_NAME}_*.log \
        2>/dev/null | head -1
    )"

    # Retry briefly if log creation is delayed.
    for _ in {1..6}
    do
        [ -n "$LOG" ] && break

        sleep 5

        LOG="$(
            ls -t logs/${RUN_NAME}_*.log \
            2>/dev/null | head -1
        )"
    done

    if [ -z "$LOG" ]; then
        echo "ERROR: no log created for $RUN_NAME"
        continue
    fi

    echo "LOG      : $LOG"
    echo "Waiting for completion..."

    while true
    do
        if grep -q \
            "Simulation finished successfully" \
            "$LOG"
        then
            STATUS="SUCCESS"
            break
        fi

        # Queue worker itself does not contain RUN_NAME
        # in its command line, so this checks the actual
        # simulation process.
        if ! pgrep -f "$RUN_NAME" >/dev/null
        then
            STATUS="FAILED_OR_STOPPED"
            break
        fi

        LAST="$(
            grep "centralized eval" "$LOG" \
            | tail -1
        )"

        if [ -n "$LAST" ]; then
            echo "$(date '+%H:%M:%S') | $LAST"
        fi

        sleep 30
    done

    echo
    echo "------------------------------------------------------------"
    echo "$STATUS : $RUN_NAME"
    echo "------------------------------------------------------------"

    grep "centralized eval" "$LOG" | tail -3

    if [ "$STATUS" != "SUCCESS" ]; then
        echo
        echo "Possible errors:"

        grep -iE \
        'traceback|exception|runtimeerror|cuda out of memory|(^|[^[:alnum:]_])(nan|inf)([^[:alnum:]_]|$)' \
        "$LOG" | tail -30
    fi

    # Let Ray clean up before launching next simulation.
    sleep 10

done < "$QUEUE"

echo
echo "============================================================"
echo "GPU QUEUE FINISHED"
echo "GPU   : $GPU"
echo "QUEUE : $QUEUE"
echo "TIME  : $(date)"
echo "============================================================"
