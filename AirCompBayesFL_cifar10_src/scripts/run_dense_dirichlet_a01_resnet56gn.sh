#!/usr/bin/env bash
set -euo pipefail

cd ~/DungNDH/micl/AirCompBayesFL_cifar10_src

SEED="${1:-12025}"
ROUNDS=100

PYRO_CONFIG="configs/cifar_priority/cifar_dirichlet_a01_resnet56gn_pyro.yaml"
BT_CONFIG="configs/cifar_priority/cifar_dirichlet_a01_resnet56gn_bayesian_torch.yaml"

FED_OUT="results/dense_dirichlet_a01_resnet56gn_fedavg_r100_seed${SEED}"
PYRO_OUT="results/dense_dirichlet_a01_resnet56gn_pyro_r100_seed${SEED}"
BT_OUT="results/dense_dirichlet_a01_resnet56gn_bayesian_torch_r100_seed${SEED}"

FED_LOG="logs/dense_dirichlet_a01_resnet56gn_fedavg_r100_seed${SEED}.log"
PYRO_LOG="logs/dense_dirichlet_a01_resnet56gn_pyro_r100_seed${SEED}.log"
BT_LOG="logs/dense_dirichlet_a01_resnet56gn_bayesian_torch_r100_seed${SEED}.log"

mkdir -p logs results


echo "============================================================"
echo "DIRICHLET alpha=0.1 / RESNET-56-GN"
echo "============================================================"
echo "seed             : ${SEED}"
echo "logical rounds   : ${ROUNDS}"
echo "FedAvg           : GPU 0"
echo "Pyro             : GPU 1"
echo "Bayesian-Torch   : GPU 0 after FedAvg"
echo "============================================================"


# ------------------------------------------------------------
# Safety checks
# ------------------------------------------------------------

if pgrep -f "python.*main_cifar10.py" >/dev/null; then
    echo
    echo "ERROR: another CIFAR simulation is active:"
    pgrep -af "python.*main_cifar10.py"
    exit 1
fi

for path in \
    "$FED_OUT" \
    "$PYRO_OUT" \
    "$BT_OUT"
do
    if [ -e "$path" ]; then
        echo
        echo "ERROR: output already exists:"
        echo "$path"
        exit 1
    fi
done


# ------------------------------------------------------------
# FedAvg — GPU 0
# ------------------------------------------------------------

echo
echo "Launching FedAvg..."

CUDA_VISIBLE_DEVICES=0 \
python -u main_cifar10.py \
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

echo "FedAvg PID: $FED_PID"


# ------------------------------------------------------------
# Proposed / Pyro — GPU 1
# ------------------------------------------------------------

echo
echo "Launching Proposed/Pyro..."

CUDA_VISIBLE_DEVICES=1 \
python -u main_cifar10.py \
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

echo "Pyro PID: $PYRO_PID"


# ------------------------------------------------------------
# Wait for FedAvg
# ------------------------------------------------------------

echo
echo "Waiting for FedAvg..."

wait "$FED_PID"

echo
echo "FedAvg finished successfully."


# ------------------------------------------------------------
# Proposed / Bayesian-Torch — reuse GPU 0
# ------------------------------------------------------------

echo
echo "Launching Proposed/Bayesian-Torch..."

CUDA_VISIBLE_DEVICES=0 \
python -u main_cifar10.py \
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

echo "Bayesian-Torch PID: $BT_PID"


# ------------------------------------------------------------
# Wait for remaining methods
# ------------------------------------------------------------

echo
echo "Waiting for Proposed/Pyro..."

wait "$PYRO_PID"

echo
echo "Pyro finished successfully."

echo
echo "Waiting for Proposed/Bayesian-Torch..."

wait "$BT_PID"

echo
echo "Bayesian-Torch finished successfully."


# ------------------------------------------------------------
# Verify identical partitions
# ------------------------------------------------------------

echo
echo "============================================================"
echo "PARTITION HASHES"
echo "============================================================"

sha256sum \
    "$FED_OUT"/partitions/*.json \
    "$PYRO_OUT"/partitions/*.json \
    "$BT_OUT"/partitions/*.json


echo
echo "============================================================"
echo "ALL THREE METHODS FINISHED"
echo "============================================================"
