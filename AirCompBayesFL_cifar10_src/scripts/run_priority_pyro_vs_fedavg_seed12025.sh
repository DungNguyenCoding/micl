#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/cifar_priority/cifar_baseline_e10_m09_cosine.yaml"
SEED=12025
ROUNDS=80

mkdir -p logs results

if pgrep -f "python.*main_cifar10.py" >/dev/null; then
    echo "ERROR: another main_cifar10.py simulation is already running."
    echo "No priority baseline was started."
    exit 1
fi

for out in \
    "results/dense_priority_pyro_proposed_seed${SEED}" \
    "results/dense_priority_fedavg_seed${SEED}"
do
    if [ -e "$out" ]; then
        echo "ERROR: final output already exists: $out"
        echo "Refusing to overwrite."
        exit 1
    fi
done

echo "Launching Proposed/Pyro on GPU 0..."
CUDA_VISIBLE_DEVICES=0 \
python -u main_cifar10.py \
  --config "$CONFIG" \
  --experiment fig2 \
  --methods proposed \
  --rounds "$ROUNDS" \
  --replications 1 \
  --seed "$SEED" \
  --path-loss-reference-m 1000 \
  --output "results/dense_priority_pyro_proposed_seed${SEED}" \
  > "logs/dense_priority_pyro_proposed_seed${SEED}.log" 2>&1 &
PID_PROPOSED=$!

echo "Launching FedAvg on GPU 1..."
CUDA_VISIBLE_DEVICES=1 \
python -u main_cifar10.py \
  --config "$CONFIG" \
  --experiment fig2 \
  --methods fedavg \
  --rounds "$ROUNDS" \
  --replications 1 \
  --seed "$SEED" \
  --path-loss-reference-m 1000 \
  --output "results/dense_priority_fedavg_seed${SEED}" \
  > "logs/dense_priority_fedavg_seed${SEED}.log" 2>&1 &
PID_FEDAVG=$!

echo "Proposed/Pyro PID=$PID_PROPOSED"
echo "FedAvg PID=$PID_FEDAVG"

wait "$PID_PROPOSED"
STATUS_PROPOSED=$?
wait "$PID_FEDAVG"
STATUS_FEDAVG=$?

echo "Proposed/Pyro exit=$STATUS_PROPOSED"
echo "FedAvg exit=$STATUS_FEDAVG"

if [ "$STATUS_PROPOSED" -ne 0 ] || [ "$STATUS_FEDAVG" -ne 0 ]; then
    exit 1
fi
