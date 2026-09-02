#!/usr/bin/env bash
set -euo pipefail

# Source-oriented cleanup for AirCompBayesFL_cifar10_src.
#
# Default: DRY RUN only.
# Apply deletions with:
#   bash scripts/cleanup_priority_workspace.sh --apply
#
# Goal:
#   - keep Python source, tests, YAML configs, Linux priority scripts,
#     documentation, VERSION, LICENSE, requirements, and final results;
#   - remove obsolete Windows launchers, legacy root launchers/backups,
#     caches, temporary tuning/debug artifacts, and patch/archive clutter.

MODE="${1:---dry-run}"
if [[ "${MODE}" != "--dry-run" && "${MODE}" != "--apply" ]]; then
    echo "Usage: $0 [--dry-run|--apply]"
    exit 2
fi

ROOT="$(pwd)"

if [[ ! -f "${ROOT}/main.py" || ! -f "${ROOT}/main_cifar10.py" ]]; then
    echo "ERROR: run this from the AirCompBayesFL_cifar10_src project root."
    exit 1
fi

if pgrep -f "python.*main_cifar10.py" >/dev/null 2>&1; then
    echo "ERROR: an active main_cifar10.py simulation was detected."
    echo "Stop/wait for the simulation before cleaning the source tree."
    pgrep -af "python.*main_cifar10.py" || true
    exit 1
fi

declare -a TARGETS=()

add_if_exists() {
    local p="$1"
    [[ -e "$p" || -L "$p" ]] && TARGETS+=("$p")
}

# Obsolete top-level platform/legacy launchers.
for p in \
    run_cifar10_sparse_rep3.ps1 \
    run_sparse_proposed.ps1 \
    run_windows_gpu.ps1 \
    run_smoke.bat \
    run_smoke.sh
do
    add_if_exists "$p"
done

# Catch any other top-level Windows launchers.
while IFS= read -r -d '' p; do
    TARGETS+=("$p")
done < <(find . -maxdepth 1 -type f \( -name '*.ps1' -o -name '*.bat' -o -name '*.cmd' \) -print0)

# Explicit obsolete/backup/debug artifacts.
for p in \
    utils_v161_backup.py \
    debug_aircomp_optimizer_upgrade_bundle.txt \
    scripts/run_dense_keep100_seeds_12025_12027.sh
do
    add_if_exists "$p"
done

# Project-root patch/archive clutter after installation.
while IFS= read -r -d '' p; do
    TARGETS+=("$p")
done < <(
    find . -maxdepth 1 -type f \
      \( -name '*.patch' -o -name '*.diff' -o -name '*.zip' \
         -o -name '*.tar' -o -name '*.tar.gz' -o -name '*.tgz' \) \
      -print0
)

# Python/test/editor/OS caches.
while IFS= read -r -d '' p; do
    TARGETS+=("$p")
done < <(
    find . \
      \( -type d \( -name '__pycache__' -o -name '.pytest_cache' \
                    -o -name '.mypy_cache' -o -name '.ruff_cache' \
                    -o -name '.ipynb_checkpoints' \) \
         -o -type f \( -name '*.pyc' -o -name '*.pyo' \
                       -o -name '.DS_Store' -o -name 'Thumbs.db' \
                       -o -name 'desktop.ini' \) \) \
      -print0
)

# Old CIFAR tuning configs.
add_if_exists "configs/cifar_tune"

# Temporary result/log families only.
# KEEP: results/dense_*, results/sparse_*, results/plot_*.
for base in results logs; do
    [[ -d "$base" ]] || continue
    while IFS= read -r -d '' p; do
        TARGETS+=("$p")
    done < <(
        find "$base" -mindepth 1 -maxdepth 1 \
          \( -name 'tune_*' -o -name 'debug_*' -o -name 'archive_*' \) \
          -print0
    )
done

mapfile -t TARGETS < <(printf '%s\n' "${TARGETS[@]:-}" | sed '/^$/d' | sort -u)

echo "============================================================"
echo "AirCompBayesFL source-oriented cleanup"
echo "Mode: ${MODE}"
echo "Root: ${ROOT}"
echo "============================================================"
echo
echo "WILL KEEP intentionally:"
echo "  - *.py source files (except explicit obsolete backups)"
echo "  - tests/"
echo "  - configs/cifar_final/"
echo "  - configs/cifar_priority/"
echo "  - other non-tuning YAML configs"
echo "  - scripts/ current Linux priority launch/summary scripts"
echo "  - README.md, CHANGELOG.md, VALIDATION.md, PRIORITY_BASELINE_*.md"
echo "  - VERSION, LICENSE, requirements*.txt, .gitignore"
echo "  - results/dense_*, results/sparse_*, results/plot_*"
echo "  - data/download directories (not touched)"
echo
echo "PLANNED REMOVALS:"
if [[ "${#TARGETS[@]}" -eq 0 ]]; then
    echo "  (nothing)"
else
    printf '  %s\n' "${TARGETS[@]}"
fi
echo

if [[ "${MODE}" == "--dry-run" ]]; then
    echo "DRY RUN ONLY: nothing was deleted."
    echo
    echo "If this list looks correct, apply with:"
    echo "  bash scripts/cleanup_priority_workspace.sh --apply"
    exit 0
fi

for p in "${TARGETS[@]}"; do
    rm -rf -- "$p"
done

if [[ -d logs ]] && [[ -z "$(find logs -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    rmdir logs
fi

echo "Cleanup complete."
echo
echo "===== TOP-LEVEL FILES AFTER CLEANUP ====="
find . -maxdepth 1 -type f -printf '%f\n' | sort
echo
echo "===== TOP-LEVEL DIRECTORIES AFTER CLEANUP ====="
find . -mindepth 1 -maxdepth 1 -type d -printf '%f/\n' | sort
