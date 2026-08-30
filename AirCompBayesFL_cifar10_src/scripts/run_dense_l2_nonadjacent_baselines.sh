#!/usr/bin/env bash
set -euo pipefail

cd ~/DungNDH/micl/AirCompBayesFL_cifar10_src

SEED="${1:-12025}"
ROUNDS=100

PYRO_CONFIG="configs/cifar_priority/cifar_l2_nonadjacent_pyro.yaml"
BT_CONFIG="configs/cifar_priority/cifar_l2_nonadjacent_bayesian_torch.yaml"

FED_OUT="results/dense_l2_nonadjacent_fedavg_r100_seed${SEED}"
FED_LOG="logs/dense_l2_nonadjacent_fedavg_r100_seed${SEED}.log"

PYRO_OUT="results/dense_l2_nonadjacent_pyro_r100_seed${SEED}"
PYRO_LOG="logs/dense_l2_nonadjacent_pyro_r100_seed${SEED}.log"

BT_OUT="results/dense_l2_nonadjacent_bayesian_torch_r100_seed${SEED}"
BT_LOG="logs/dense_l2_nonadjacent_bayesian_torch_r100_seed${SEED}.log"

mkdir -p logs results

for path in \
    "$FED_OUT" \
    "$PYRO_OUT" \
    "$BT_OUT"
do
    if [ -e "$path" ]; then
        echo "ERROR: output already exists:"
        echo "$path"
        exit 1
    fi
done

if pgrep -f "python.*main_cifar10.py" >/dev/null; then
    echo "ERROR: another CIFAR simulation is active:"
    pgrep -af "python.*main_cifar10.py"
    exit 1
fi


echo "============================================================"
echo "L2 NON-ADJACENT BASELINE"
echo "seed=$SEED"
echo "rounds=$ROUNDS"
echo "============================================================"


# ------------------------------------------------------------
# GPU 0 — FedAvg
# ------------------------------------------------------------

CUDA_VISIBLE_DEVICES=0 \
nohup python -u main_cifar10.py \
    --config "$PYRO_CONFIG" \
    --experiment fig2 \
    --methods fedavg \
    --rounds "$ROUNDS" \
    --replications 1 \
    --seed "$SEED" \
    --path-loss-reference-m 1000 \
    --output "$FED_OUT" \
    > "$FED_LOG" 2>&1 &

FED_PID=$!

echo "FedAvg launched:"
echo "  PID=$FED_PID"
echo "  log=$FED_LOG"


# ------------------------------------------------------------
# GPU 1 — Proposed / Pyro
# ------------------------------------------------------------

CUDA_VISIBLE_DEVICES=1 \
nohup python -u main_cifar10.py \
    --config "$PYRO_CONFIG" \
    --experiment fig2 \
    --methods proposed \
    --rounds "$ROUNDS" \
    --replications 1 \
    --seed "$SEED" \
    --path-loss-reference-m 1000 \
    --output "$PYRO_OUT" \
    > "$PYRO_LOG" 2>&1 &

PYRO_PID=$!

echo "Pyro launched:"
echo "  PID=$PYRO_PID"
echo "  log=$PYRO_LOG"


# ------------------------------------------------------------
# Wait for FedAvg, then reuse GPU 0 for Bayesian-Torch
# ------------------------------------------------------------

echo
echo "Waiting for FedAvg before starting Bayesian-Torch..."

wait "$FED_PID"

echo
echo "FedAvg finished."
echo "Launching Bayesian-Torch on GPU 0."


CUDA_VISIBLE_DEVICES=0 \
nohup python -u main_cifar10.py \
    --config "$BT_CONFIG" \
    --experiment fig2 \
    --methods proposed \
    --rounds "$ROUNDS" \
    --replications 1 \
    --seed "$SEED" \
    --path-loss-reference-m 1000 \
    --output "$BT_OUT" \
    > "$BT_LOG" 2>&1 &

BT_PID=$!

echo "Bayesian-Torch launched:"
echo "  PID=$BT_PID"
echo "  log=$BT_LOG"


wait "$PYRO_PID"
echo "Pyro finished."

wait "$BT_PID"
echo "Bayesian-Torch finished."


echo
echo "============================================================"
echo "ALL L2 NON-ADJACENT BASELINES FINISHED"
echo "============================================================"

echo
echo "Partition hashes:"

sha256sum \
    "$FED_OUT"/partitions/*.json \
    "$PYRO_OUT"/partitions/*.json \
    "$BT_OUT"/partitions/*.json
