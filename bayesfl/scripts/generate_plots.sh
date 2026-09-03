#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "usage: $0 outputs/<run_directory>" >&2
  exit 2
fi
python -m bayesfl.utils --run-dir "$1"
