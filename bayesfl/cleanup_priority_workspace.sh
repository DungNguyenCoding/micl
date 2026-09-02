#!/usr/bin/env bash
set -euo pipefail

# Safe cleanup for the current Linux CIFAR priority workspace.
# Preserves completed results/dense_*, results/sparse_*, results/plot_*,
# configs/cifar_final/, configs/cifar_priority/, source files, docs, and tests.

rm -rf .pytest_cache __pycache__ tests/__pycache__
find . -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

rm -f utils_v161_backup.py
rm -f scripts/run_dense_keep100_seeds_12025_12027.sh

# Obsolete tuning configurations; frozen/final/priority configs are preserved.
rm -rf configs/cifar_tune

# Temporary result namespaces only.
rm -rf results/tune_* results/debug_* results/archive_*

remove_dense_if_incomplete() {
    local dir="$1"
    local metrics="$dir/metrics.csv"
    [ -d "$dir" ] || return 0

    if [ ! -f "$metrics" ]; then
        echo "Removing incomplete dense directory without metrics: $dir"
        rm -rf "$dir"
        return 0
    fi

    local max_round
    max_round="$({ python - "$metrics" <<'PY'
import sys
import pandas as pd
p = sys.argv[1]
df = pd.read_csv(p)
print(int(df["round"].max()) if len(df) else -1)
PY
    } 2>/dev/null || echo -1)"

    if [ "$max_round" -lt 80 ]; then
        echo "Removing interrupted dense result: $dir (max logical round=$max_round)"
        rm -rf "$dir"
    else
        echo "Keeping completed dense result: $dir (max logical round=$max_round)"
    fi
}

# The uploaded snapshot showed these old campaign outputs stopped at round 14.
# The guard above prevents deletion if either has since become a complete run.
remove_dense_if_incomplete results/dense_cifar_keep100_seed12025
remove_dense_if_incomplete results/dense_cifar_keep100_seed12026

# Runtime logs are reproducible artifacts; keep an empty directory for nohup.
rm -rf logs
mkdir -p logs

echo 'Cleanup complete.'
echo 'Preserved result entries:'
find results -maxdepth 1 -mindepth 1 -print | sort
