#!/usr/bin/env bash
set -u

CONFIG="configs/cifar_final/cifar_sparse_compare_fixed.yaml"
ROUNDS=80

run_seed () {
    SEED="$1"
    GPU="$2"

    OUT="results/dense_cifar_keep100_seed${SEED}"
    LOG="logs/dense_cifar_keep100_seed${SEED}.log"

    echo "============================================================"
    echo "Seed ${SEED} | GPU ${GPU}"
    echo "Output: ${OUT}"
    echo "Log:    ${LOG}"
    echo "============================================================"

    # Do not overwrite an existing final result.
    if [ -e "${OUT}" ]; then
        echo "SKIP seed ${SEED}: ${OUT} already exists."
        return 0
    fi

    CUDA_VISIBLE_DEVICES="${GPU}" \
    python -u main_cifar10.py \
      --config "${CONFIG}" \
      --experiment fig2 \
      --methods proposed \
      --rounds "${ROUNDS}" \
      --replications 1 \
      --seed "${SEED}" \
      --path-loss-reference-m 1000 \
      --output "${OUT}" \
      > "${LOG}" \
      2>&1

    STATUS=$?

    if [ "${STATUS}" -eq 0 ]; then
        echo "DONE seed ${SEED}"
    else
        echo "FAILED seed ${SEED}, exit code=${STATUS}"
    fi

    return "${STATUS}"
}

# Safety: don't compete with the current sparse runs.
if pgrep -f "main_cifar10.py.*sparse_bayes_seed12025" >/dev/null \
   || pgrep -f "main_cifar10.py.*sparse_random_seed12025" >/dev/null
then
    echo "ERROR: seed12025 sparse jobs are still running."
    echo "Dense baselines were NOT started."
    exit 1
fi

echo "Starting dense Keep-100 baselines..."
echo
echo "Wave 1: seed12025 on GPU0 + seed12026 on GPU1"

run_seed 12025 0 &
PID_12025=$!

run_seed 12026 1 &
PID_12026=$!

echo "seed12025 worker PID=${PID_12025}"
echo "seed12026 worker PID=${PID_12026}"

wait "${PID_12025}"
STATUS_12025=$?

wait "${PID_12026}"
STATUS_12026=$?

echo
echo "Wave 1 completed:"
echo "  seed12025 status=${STATUS_12025}"
echo "  seed12026 status=${STATUS_12026}"

if [ "${STATUS_12025}" -ne 0 ] || [ "${STATUS_12026}" -ne 0 ]; then
    echo "At least one Wave-1 run failed."
    echo "Seed12027 will NOT start automatically."
    exit 1
fi

echo
echo "Wave 2: seed12027 on GPU0"

run_seed 12027 0
STATUS_12027=$?

echo
echo "============================================================"
echo "Dense Keep-100 campaign finished"
echo "seed12025=${STATUS_12025}"
echo "seed12026=${STATUS_12026}"
echo "seed12027=${STATUS_12027}"
echo "============================================================"

exit "${STATUS_12027}"
