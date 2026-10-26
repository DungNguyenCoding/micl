#!/usr/bin/env python3
"""BayesFL Stage 3B: conservative SAME-SSD organization of 13 historical files.

Keeps src/, plots, all main-paper scripts, ALL 898 configs, and all other queues
in place. Only moves 5 version snapshots, 2 obsolete EMPTY queues, and 6
historical/broken queue definitions. The two empty CIFAR R150 resume queues are
intentionally protected and remain in scripts/queues/.

Place at bayesfl/scripts/bayesfl_stage3b_organizer.py, then run:
    python scripts/bayesfl_stage3b_organizer.py --preview
    python scripts/bayesfl_stage3b_organizer.py --move --confirm MOVE_13_HISTORY_FILES
    python scripts/bayesfl_stage3b_organizer.py --verify
    python scripts/bayesfl_stage3b_organizer.py --restore --confirm RESTORE_13_HISTORY_FILES

NO DELETE OPERATION. Archives are stored beside the repo, on the same SSD;
this is an organization step, NOT an independent backup or disk-space saving.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT.parent / 'bayesfl_historical_archive' / 'STAGE3_LEGACY_SCRIPTS_QUEUES'
APPROVED = ROOT.parent / 'bayesfl_cleanup_DRY_RUN_20261026_194253.csv'
MANIFEST = ARCHIVE / 'stage3b_13_files_manifest.json'

FILES = {
    'HISTORICAL_VERSIONS': [
        'scripts/_run_gpu_queue.before_resume_support.sh',
        'scripts/migrate_multicast_accounting.before_legacy_mnist.py',
        'scripts/plot_ieee_multicast_results.py.before_cifar_f1_patch',
        'scripts/plot_ieee_multicast_results_updated_old.py',
        'scripts/plot_ieee_multicast_results_with_fedavg_curve.py.before_cifar_f1_patch',
    ],
    'OBSOLETE_EMPTY_QUEUES': [
        'scripts/queues/cifar_plot_recovery_gpu1.txt',
        'scripts/queues/mnist_densegate_stageB2_gpu1_EMPTY.txt',
    ],
    'HISTORICAL_RESUME_QUEUES': [
        'scripts/queues/mnist_newmlp_final_dense_confirm_seed0_gpu0.txt',
        'scripts/queues/mnist_stage11_r200400_seed0_gpu1.txt',
        'scripts/queues/multicast_resume_keep005_seed0_gpu0.txt',
        'scripts/queues/multicast_resume_keep005_seed0_gpu1.txt',
        'scripts/queues/multicast_resume_no005_seed0_gpu0.txt',
        'scripts/queues/multicast_resume_no005_seed0_gpu1.txt',
    ],
}

TARGETS = [(category, relative) for category, names in FILES.items() for relative in names]
assert len(TARGETS) == 13 and len(set(r for _, r in TARGETS)) == 13

PROTECTED = [
    'src/bayesfl/main.py',
    'scripts/_run_gpu_queue.sh',
    'scripts/plot_ieee_multicast_results_mnist_s10_stage22.py',
    'scripts/prepare_cifar_r150_sparse_resume.py',
    'scripts/mnist_stage21_sparse_s5_seed3.py',
    'scripts/mnist_stage22_sparse_s10_seed0.py',
    'scripts/queues/cifar_sparse_r150_resume_gpu0_kl_global_local.txt',
    'scripts/queues/cifar_sparse_r150_resume_gpu1_random.txt',
]


def fail(msg: str):
    raise RuntimeError(msg)


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def dest_for(category: str, rel: str) -> Path:
    return ARCHIVE / category / rel


def validate_project() -> int:
    if ROOT.name != 'bayesfl':
        fail(f'Unexpected project root: {ROOT}')
    if ROOT.is_symlink() or ARCHIVE.is_symlink():
        fail('Symlinked root/archive is not supported')
    for rel in PROTECTED:
        path = ROOT / rel
        if not path.is_file():
            fail('Protected tool missing; do not proceed: ' + str(path))
    if not (ROOT / 'outputs/partitions').is_dir():
        fail('Protected partition directory missing')
    if not APPROVED.is_file():
        fail('Required Stage 2 dry-run CSV not found: ' + str(APPROVED))

    with APPROVED.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    keep = [r.get('relative_path', '') for r in rows if r.get('action') == 'KEEP_PROTECTED']
    if len(keep) != 42 or any(not (ROOT / r).is_dir() for r in keep):
        missing = [r for r in keep if not (ROOT / r).is_dir()]
        fail(f'Protected output validation failed ({len(keep)}/42 listed); missing={missing}')
    return len(keep)


def validate_files(file_data: dict, desired: str | None = None):
    states = []
    for category, rel in TARGETS:
        src, dst = ROOT / rel, dest_for(category, rel)
        if src.is_symlink() or dst.is_symlink():
            fail('Symlink target refused: ' + rel)
        a, b = src.is_file(), dst.is_file()
        if a and b:
            fail('Ambiguous: both source and archive contain ' + rel)
        if not a and not b:
            fail('Missing from both source and archive: ' + rel)
        path = src if a else dst
        info = file_data.get(rel)
        if info:
            if info['category'] != category or path.stat().st_size != info['bytes'] or hash_file(path) != info['sha256']:
                fail('Changed file; refusing to proceed: ' + str(path))
        elif not a:
            fail('Archive exists without a recorded manifest: ' + str(dst))
        if category == 'OBSOLETE_EMPTY_QUEUES':
            # A queue may contain comments but must not contain runnable records.
            if any(line.strip() and not line.lstrip().startswith('#') for line in path.read_text(encoding='utf-8').splitlines()):
                fail('Expected empty queue now has runnable content: ' + rel)
        state = 'SOURCE' if a else 'ARCHIVED'
        if desired and state != desired:
            fail(f'Expected {desired}, found {state}: {rel}')
        states.append((category, rel, path, state))
    return states


def read_manifest() -> dict | None:
    if not MANIFEST.exists():
        return None
    data = json.loads(MANIFEST.read_text(encoding='utf-8'))
    actual = set(data.get('files', {}))
    expect = set(r for _, r in TARGETS)
    if actual != expect:
        fail('Archive manifest has unexpected file set')
    return data


def new_manifest(states) -> dict:
    if any(state != 'SOURCE' for _, _, _, state in states):
        fail('Initial move needs all 13 source files present')
    return {
        'schema_version': 1,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'project': str(ROOT),
        'archive': str(ARCHIVE),
        'files': {
            rel: {
                'category': cat,
                'bytes': path.stat().st_size,
                'sha256': hash_file(path),
                'modified_ns': path.stat().st_mtime_ns,
            }
            for cat, rel, path, _ in states
        },
    }


def store_manifest(data: dict):
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_name(MANIFEST.name + '.tmp')
    with tmp.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, MANIFEST)


def ensure_same_filesystem():
    # Both directories are beneath ROOT.parent; guard against nonstandard mounts.
    if ROOT.stat().st_dev != ROOT.parent.stat().st_dev:
        fail('Project and parent are on different filesystems')
    if ARCHIVE.exists() and ROOT.stat().st_dev != ARCHIVE.stat().st_dev:
        fail('Archive and project are on different filesystems; move refused')


def show(states, phase):
    total = sum(p.stat().st_size for _, _, p, _ in states)
    moved = sum(st == 'ARCHIVED' for _, _, _, st in states)
    print('=' * 84)
    print('BAYESFL STAGE 3B — ' + phase)
    print('=' * 84)
    print('Project          :', ROOT)
    print('Archive          :', ARCHIVE)
    print('Protected outputs: 42 / 42')
    print('Exact target files:', len(states))
    print('Already archived :', moved)
    print('Still in scripts :', len(states) - moved)
    print('Total logical bytes:', format(total, ','))
    for category, rel, path, state in states:
        print(f'{state:8s} {category:26s} {rel}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_mutually_exclusive_group(required=True)
    operations.add_argument('--preview', action='store_true')
    operations.add_argument('--move', action='store_true')
    operations.add_argument('--verify', action='store_true')
    operations.add_argument('--restore', action='store_true')
    parser.add_argument('--confirm', default='', help='Required exact token for move/restore')
    args = parser.parse_args()

    validate_project()
    existing = read_manifest()
    records = existing['files'] if existing else {}
    states = validate_files(records)
    ensure_same_filesystem()
    phase = 'PREVIEW' if args.preview else 'MOVE' if args.move else 'VERIFY' if args.verify else 'RESTORE'
    show(states, phase)

    if args.preview:
        print('PASS: PREVIEW ONLY. No repository changes.')
        return

    if args.verify:
        if not existing:
            fail('No prior move manifest; cannot verify archived files')
        validate_files(existing['files'], 'ARCHIVED')
        print('PASS: ALL 13 ARCHIVED FILES VALIDATED; PROTECTED OUTPUTS PRESERVED')
        return

    if args.move:
        if args.confirm != 'MOVE_13_HISTORY_FILES':
            fail('Explicit confirmation required: --confirm MOVE_13_HISTORY_FILES')
        if not existing:
            existing = new_manifest(states)
            # Manifest written before any files move so a partial move is reversible.
            store_manifest(existing)
        validate_files(existing['files'])
        for category, rel, _, state in states:
            if state == 'ARCHIVED':
                continue
            src, dst = ROOT / rel, dest_for(category, rel)
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
            print('MOVED:', rel)
        validate_files(existing['files'], 'ARCHIVED')
        print('PASS: MOVE + SHA-256 VERIFICATION COMPLETE')
        return

    if args.restore:
        if args.confirm != 'RESTORE_13_HISTORY_FILES':
            fail('Explicit confirmation required: --confirm RESTORE_13_HISTORY_FILES')
        if not existing:
            fail('No move manifest; no archive to restore')
        validate_files(existing['files'])
        for category, rel, _, state in states:
            if state == 'SOURCE':
                continue
            src, dst = ROOT / rel, dest_for(category, rel)
            src.parent.mkdir(parents=True, exist_ok=True)
            dst.rename(src)
            print('RESTORED:', rel)
        validate_files(existing['files'], 'SOURCE')
        print('PASS: RESTORE + SHA-256 VERIFICATION COMPLETE')
        return


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('ERROR:', exc, file=sys.stderr)
        raise SystemExit(1)
