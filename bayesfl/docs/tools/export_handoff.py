#!/usr/bin/env python3
"""Read primary BayesFL endpoints or export a lightweight new-chat snapshot.

Default/--check: terminal only. --export: creates one NEW ZIP beside the repo.
Never trains, imports BayesFL, loads weights, edits metrics or changes Git.
The ZIP is NOT a complete training/resume backup. Standard library only.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from decimal import Decimal, InvalidOperation
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import stat
import sys
import zipfile

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
TEXT_SUFFIXES = {'.py', '.sh', '.yaml', '.yml', '.json', '.md', '.txt', '.toml', '.ini', '.cfg'}
EXCLUDED_PARTS = {'.git', '__pycache__', '.pytest_cache', '.venv', 'venv', 'node_modules'}
ROOT_FILES = ('README.md', 'AGENTS.md', 'SPARSE_README.md', 'IMPLEMENTATION_REPORT.md',
              'VALIDATION.md', 'VALIDATION_SPARSE.md', 'pyproject.toml', 'requirements.txt',
              'CHANGE_MANIFEST.json', 'PACKAGE_MANIFEST.txt', 'DOCS_UPDATE_MANIFEST.json', '.gitignore')


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_path(root: Path, rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        fail(f'Unsafe relative path: {rel}')
    path = root / p
    current = root
    for part in p.parts:
        current = current / part
        if current.is_symlink():
            fail(f'Symbolic link not followed: {current}')
    return path


def int_value(value: str, field: str) -> int:
    try:
        dec = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f'Invalid {field}: {value!r}') from exc
    if not dec.is_finite() or dec != dec.to_integral_value():
        fail(f'{field} must be a finite integer: {value!r}')
    return int(dec)


def stable_read(path: Path) -> bytes:
    if path.is_symlink():
        fail(f'Symbolic file refused: {path}')
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        fail(f'Not a regular file: {path}')
    data = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        fail(f'File changed while reading; stop writers and retry: {path}')
    return data


def read_endpoint(root: Path, run: dict, profiles: dict, unit: str) -> dict:
    rel = run['relative_output'] + '/' + run['metrics_relative_path']
    path = safe_path(root, rel)
    if not path.is_file():
        fail(f'Missing exact metric file (no wildcard substitution): {rel}')
    raw = stable_read(path)
    rows = {}
    for row in csv.DictReader(raw.decode('utf-8-sig').splitlines()):
        if not row.get('round', '').strip():
            continue
        rd = int_value(row['round'], 'round')
        if rd in rows:
            fail(f'Duplicate metric round R{rd}: {rel}')
        rows[rd] = row
    target = run['plot_target_round']
    if target not in rows:
        fail(f'INCOMPLETE/MISSING CHECKPOINT: {run["id"]}; need R{target}; last={max(rows, default=-1)}')
    row = rows[target]
    keys = (['fola_mean_accuracy'] if run['method'] == 'fola' else []) + [
        'global_test_accuracy', 'global_accuracy', 'accuracy', 'test_accuracy',
        'centralized_accuracy', 'centralized_acc', 'mean_accuracy', 'mean_acc']
    key = next((k for k in keys if row.get(k) not in (None, '')), None)
    if key is None:
        fail(f'No recognized global point-estimate accuracy field in {rel}; columns={list(row)}')
    acc = float(row[key])
    high = 1.0 if unit == 'fraction' else 100.0
    if not math.isfinite(acc) or not 0 <= acc <= high:
        fail(f'Accuracy is invalid for explicit unit={unit}: {acc} in {rel}')
    percent = acc * 100 if unit == 'fraction' else acc
    if 'cumulative_all_array_bytes' not in row:
        fail(f'Missing cumulative_all_array_bytes in {rel}; no estimated substitute allowed')
    cost = int_value(row['cumulative_all_array_bytes'], 'cumulative_all_array_bytes')
    p = profiles[run['profile']]
    d, k, budget = p['d'], p['participating_clients'], p['budget_bytes']
    keep = run['keep_ratio']
    if keep is None:
        per_round = (4 if run['method'] == 'fedavg' else 8) * d * (k + 1)
    else:
        per_round = 8*d + k*(8*math.ceil(keep*d) + math.ceil(d/8))
        if target != budget // per_round:
            fail(f'Registry sparse target is not last affordable round: {run["id"]}')
    if cost != target * per_round or cost > budget:
        fail(f'Byte mismatch {run["id"]}: recorded={cost}, expected={target*per_round}, budget={budget}')
    if keep is None and cost != budget:
        fail(f'Dense endpoint does not equal budget: {run["id"]}')
    return {'id': run['id'], 'profile': run['profile'], 'role': run['role'],
            'keep_ratio': keep, 'round': target, 'accuracy_percent': percent,
            'accuracy_column': key, 'input_accuracy_unit': unit, 'communication_bytes': cost,
            'metrics_relative_path': rel, 'metrics_sha256': sha(raw),
            'previous_documented_accuracy_percent': run.get('accuracy_percent_from_supplied_result')}


def print_results(results: list[dict]) -> None:
    for profile in ('mnist_s10', 'cifar10_a0p01_f1'):
        rs = [x for x in results if x['profile'] == profile]
        dense = next(x for x in rs if x['role'] == 'dense_fola')
        avg = next(x for x in rs if x['role'] == 'fedavg_baseline')
        print('\n' + '='*105)
        print(f'{profile.upper()} — RAW COMMUNICATION-MATCHED ENDPOINTS')
        print('='*105)
        for label, r in [('Full-posterior BayesFL', dense), ('FedAvg', avg)]:
            print(f'{label:<25} R{r["round"]:<4} {r["accuracy_percent"]:8.3f}%  {r["communication_bytes"]/1e9:12.9f} GB')
        print(f'Dense minus FedAvg: {dense["accuracy_percent"]-avg["accuracy_percent"]:+.3f} pp')
        print(f'{"Keep":>6} {"Round":>6} {"Comm GB":>14} {"KL %":>9} {"Random %":>10} {"KL-Random":>12} {"KL-Dense":>11} {"KL-FedAvg":>12}')
        for keep in (.75, .5, .25, .1, .05):
            a = next(x for x in rs if x['role']=='kl_sparse' and x['keep_ratio']==keep)
            b = next(x for x in rs if x['role']=='random_sparse' and x['keep_ratio']==keep)
            if a['communication_bytes'] != b['communication_bytes']:
                fail('Selector costs differ at equal keep ratio')
            aa, bb = a['accuracy_percent'], b['accuracy_percent']
            print(f'{keep:6.2f} {a["round"]:6d} {a["communication_bytes"]/1e9:14.9f} {aa:9.3f} {bb:10.3f} {aa-bb:+12.3f} {aa-dense["accuracy_percent"]:+11.3f} {aa-avg["accuracy_percent"]:+12.3f}')
    for r in results:
        prior = r['previous_documented_accuracy_percent']
        if prior is not None and abs(prior-r['accuracy_percent']) > 0.0051:
            print(f'NOTE: {r["id"]} differs from old documented endpoint ({prior:.3f}%); inspect provenance before updating docs.')
    print('\nPASS: 24 primary endpoints and exact byte totals read from current CSV files.')
    print('No simulation, model loading, metric modification, or inferred missing accuracy.')


def selected_export_files(root: Path, registry: dict) -> dict[str, Path]:
    files = {}
    for name in ROOT_FILES:
        p = safe_path(root, name)
        if p.is_file():
            files[name] = p
    for base in ('src', 'scripts', 'tests', 'docs'):
        folder = safe_path(root, base)
        if not folder.is_dir():
            fail(f'Missing expected repository directory: {folder}')
        for current, dirs, names in os.walk(folder, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_PARTS)
            for d in dirs:
                if (Path(current)/d).is_symlink():
                    fail(f'Export refuses symlinked directory: {Path(current)/d}')
            for name in sorted(names):
                p = Path(current)/name
                rel = p.relative_to(root).as_posix()
                is_fixture = rel.startswith('tests/fixtures/') and p.suffix == '.npz'
                if p.suffix not in TEXT_SUFFIXES and not is_fixture:
                    continue
                safe_path(root, rel)
                if p.stat().st_size > 20*1024*1024:
                    fail(f'Unexpectedly large text/fixture; review before sharing: {rel}')
                files[rel] = p
    for run in registry['runs']:
        if not run['primary_figure']:
            continue
        directory = safe_path(root, run['relative_output'])
        required = ('metrics/global_metrics.csv', 'resolved_config.yaml')
        optional = ('source_config.yaml', 'partition_metadata.json', 'run_summary.json',
                    'environment.json', 'environment.txt', 'communication/layout_manifest.json',
                    'communication/protocol.json')
        for name in required + optional:
            rel = run['relative_output']+'/'+name
            p = safe_path(root, rel)
            if p.is_file():
                files[rel] = p
            elif name in required:
                fail(f'Cannot create provenance export: required file missing: {rel}')
    for p in sorted(safe_path(root, 'outputs/partitions').glob('*.json')):
        rel = p.relative_to(root).as_posix()
        safe_path(root, rel)
        files[rel] = p
    return files


def export(root: Path, registry: dict, results: list[dict]) -> Path:
    for rel in ('src/bayesfl/main.py', 'scripts/_run_gpu_queue.sh',
                'scripts/plot_ieee_multicast_results_mnist_s10_stage22.py'):
        if not safe_path(root, rel).is_file():
            fail('Required current source/plot tool missing: '+rel)
    files = selected_export_files(root, registry)
    now = datetime.now().astimezone()
    dest = root.parent / ('bayesfl_LIVE_HANDOFF_'+now.strftime('%Y%m%d_%H%M%S_%f')+'.zip')
    versions = {}
    for package in ('torch', 'torchvision', 'flwr', 'ray', 'bayesian-torch', 'numpy', 'matplotlib', 'PyYAML', 'pytest'):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    summary = {'captured_at': now.isoformat(), 'repository': str(root),
               'python_executable': sys.executable, 'python_version': platform.python_version(),
               'installed_distribution_versions': versions, 'primary_endpoints': results,
               'scope': 'Current selected code/docs/configs/tests and 24 primary metrics; not a full backup.',
               'excluded': ['datasets', 'model weights', 'resume arrays', 'communication events ledgers',
                            'logs', '.git', 'partition NPZs', 'historical experiment trees', 'environment secrets'],
               'snapshot_warning': 'Per-file stability is checked. Whole-tree atomicity is not guaranteed; stop writers before export.'}
    observed_hashes = {r['metrics_relative_path']: r['metrics_sha256'] for r in results}
    index = []
    with zipfile.ZipFile(dest, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as out:
        for rel, path in sorted(files.items()):
            data = stable_read(path)
            digest = sha(data)
            if rel in observed_hashes and digest != observed_hashes[rel]:
                fail(f'Metrics changed after checking: {rel}; incomplete ZIP retained, do not use it.')
            out.writestr('bayesfl/'+rel, data)
            index.append({'path':'bayesfl/'+rel, 'bytes':len(data), 'sha256':digest})
        out.writestr('HANDOFF_CAPTURE.json', json.dumps(summary, indent=2)+'\n')
        out.writestr('HANDOFF_FILE_HASHES.json', json.dumps(index, indent=2)+'\n')
        out.writestr('READ_ME_FIRST.txt',
            'Read bayesfl/docs/CHAT_HANDOFF.md and HANDOFF_CAPTURE.json first.\n'
            'This is a lightweight source/metrics snapshot, not a resume backup.\n'
            'No model weights, raw data, partition NPZs or full resume ledger are included.\n'
            'Review paths, code, configs and metadata before sharing externally.\n')
    with zipfile.ZipFile(dest) as z:
        bad = z.testzip()
        if bad:
            fail('ZIP CRC verification failed: '+bad)
    print(f'\nEXPORT CREATED: {dest}\nFiles copied: {len(index)}\nZIP size: {dest.stat().st_size:,} bytes')
    print('Review before sharing. Existing repository files were not modified.')
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='Read/print endpoints only (default).')
    mode.add_argument('--export', action='store_true', help='Check and create a NEW lightweight ZIP beside the repo.')
    parser.add_argument('--accuracy-unit', choices=('fraction','percent'), default='fraction',
                        help='Explicit stored CSV accuracy unit; never inferred from an individual value.')
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    path = safe_path(root, 'docs/EXPERIMENT_REGISTRY.json')
    registry = json.loads(stable_read(path))
    selected = [r for r in registry['runs'] if r['primary_figure']]
    if len(selected) != 24:
        fail(f'Expected 24 explicitly registered primary runs; found {len(selected)}')
    results = [read_endpoint(root, r, registry['profiles'], args.accuracy_unit) for r in selected]
    print_results(results)
    if args.export:
        export(root, registry, results)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError, csv.Error, zipfile.BadZipFile) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)
