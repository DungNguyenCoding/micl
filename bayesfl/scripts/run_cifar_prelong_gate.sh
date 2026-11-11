#!/usr/bin/env bash
# Run ONCE before considering longer 100/150-round CIFAR-10 pilots.
# This script runs tests and one shared-preparation R11 paired evaluation.
# It never creates training queues, launches a long simulation or edits protected runs.
set -Eeuo pipefail

GPU="${1:-0}"
if ! [[ "$GPU" =~ ^[0-9]+$ ]]; then
  echo 'ERROR: GPU must be a nonnegative integer, e.g. 0' >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
PY=/home/micl/anaconda3/envs/dungndh/bin/python
if [[ ! -x "$PY" ]]; then
  echo "ERROR: Pinned python unavailable: $PY" >&2
  exit 2
fi
for f in \
  scripts/probe_cifar_same_posterior.py \
  scripts/probe_cifar_paired_r11.py \
  scripts/report_cifar_prelong_gate.py \
  docs/tools/export_handoff.py \
  scripts/check_cifar_wireless_scenario_v1.py \
  tests/wireless/test_paired_r11_probe.py
  do
    if [[ ! -f "$f" ]]; then
      echo "ERROR: Missing installed dependency: $f" >&2
      exit 2
    fi
  done

export PYTHONPATH="$ROOT/src:$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU"

echo "Repo: $ROOT"
echo "Python: $PY"
echo "GPU selection: CUDA_VISIBLE_DEVICES=$GPU"
echo 'Confirm no other training occupies this GPU before executing this script.'

TAG="bayesfl_cifar_prelong_gate_$(date +%Y%m%d_%H%M%S)_$$"
OUT="$ROOT/../$TAG"
mkdir -p "$OUT/logs"
trap 'echo "Pre-long gate stopped. Diagnostics (if any): $OUT" >&2' ERR

echo "New independent evidence directory: $OUT"
echo '[1/6] Wireless and new paired probe tests'
"$PY" -m pytest -q -rs tests/wireless 2>&1 | tee "$OUT/logs/01_wireless_pytest.txt"
echo '[2/6] Full repository regression tests'
"$PY" -m pytest -q -rs 2>&1 | tee "$OUT/logs/02_full_pytest.txt"
echo '[3/6] Baseline 24-endpoint communication check'
"$PY" docs/tools/export_handoff.py --check 2>&1 | tee "$OUT/logs/03_baseline_endpoints.txt"
echo '[4/6] Fixed-channel CIFAR radio/partition feasibility'
"$PY" scripts/check_cifar_wireless_scenario_v1.py 2>&1 | tee "$OUT/logs/04_cifar_radio_preflight.txt"
echo '[5/6] Original R10 checkpoint and locked CIFAR source preflight'
"$PY" scripts/probe_cifar_paired_r11.py --preflight --checkpoint-round 10 2>&1 | tee "$OUT/logs/05_paired_preflight.txt"
echo '[6/6] ONE real R11 local preparation, TWO independent packet/aggregate/evaluation branches'
"$PY" scripts/probe_cifar_paired_r11.py \
  --execute --checkpoint-round 10 --require-cuda \
  --output-dir "$OUT/paired_r11" 2>&1 | tee "$OUT/logs/06_paired_r11.txt"

echo '[REPORT] Checking every scalar/packet/accuracy gate and packaging privacy-limited evidence'
"$PY" scripts/report_cifar_prelong_gate.py --gate-dir "$OUT"

echo '=================================='
echo 'PASS: CIFAR-10 PRE-LONG GATE'
echo "Evidence: $OUT"
echo "Upload: $OUT.zip"
echo 'No R100/R150 training job was launched.'
