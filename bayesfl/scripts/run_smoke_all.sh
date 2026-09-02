#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

for cfg in \
  smoke_fedavg_mnist.yaml \
  smoke_fedavg_cifar10.yaml \
  smoke_bbb_mnist.yaml \
  smoke_bbb_cifar10.yaml \
  smoke_fola_mnist.yaml \
  smoke_fola_cifar10.yaml
do
  echo "================================================================"
  echo "Running $cfg"
  echo "================================================================"
  python -m bayesfl.main --config "scripts/configs/$cfg"
done
