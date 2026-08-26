#!/usr/bin/env bash
set -euo pipefail
python main.py --config configs/smoke.yaml --experiment fig2 --methods fedavg,proposed
python utils.py --input results --figure fig2
