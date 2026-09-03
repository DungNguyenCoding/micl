#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELECTED="$ROOT/scripts/configs/fola_cifar10_selected.yaml"
if [[ ! -f "$SELECTED" ]]; then
  cat >&2 <<'EOF'
FOLA prior lambda must be selected from the 20-round paper-style sweep while
keeping E=10 and the requested optimizer schedule.

Run:
  bash scripts/run_paper_fola_lambda_sweep.sh
Then, after it finishes:
  bash scripts/run_fola_cifar10.sh
EOF
  exit 2
fi
exec "$ROOT/scripts/_run_nohup.sh" "$SELECTED" "$@"
