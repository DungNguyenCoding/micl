#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/_run_nohup.sh" "$ROOT/scripts/configs/fola_cifar10_full300.yaml" "$@"
