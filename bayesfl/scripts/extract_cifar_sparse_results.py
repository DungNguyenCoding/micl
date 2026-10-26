#!/usr/bin/env python3
"""Extract CIFAR-10 final selected sparse experiment accuracies.

Place this script anywhere, and run it from the BayesFL repo root:
  PYTHONPATH="$PWD/src" python extract_cifar_sparse_results.py

Does NOT launch training, alter metrics, or invent missing accuracy values.
Only accepts outputs with the expected R120 multicast communication counts.
"""
from pathlib import Path
import argparse
import csv
import math


D, CLIENTS, DENSE_ROUNDS = 878_538, 20, 120
RATIOS = (0.75, 0.50, 0.25, 0.10, 0.05)
TAGS = {0.75: '0p75', 0.50: '0p5', 0.25: '0p25', 0.10: '0p1', 0.05: '0p05'}
DENSE_PREFIX = (
    'run_confirm_newsrc_cifar_basiccnn_a0p01_fola_cos400_n20_f1_'
    'seed0_e10_b32_lr002_lam0p01_r150'
)
FEDAVG_PREFIX = (
    'run_budgetmatch_cifar_basiccnn_a0p01_fedavg_n20_f1_'
    'seed0_e10_b32_lr001_budget150densefola'
)
SPARSE_PATTERN = (
    'run_sparse_cifar_basiccnn_a0p01_f1_fola_'
    '{rule}_keep{tag}_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense'
)
ACC_KEYS = (
    'global_test_accuracy', 'global_accuracy', 'centralized_accuracy',
    'centralized_acc', 'mean_accuracy', 'mean_acc', 'fola_mean_accuracy',
    'accuracy', 'test_accuracy',
)
BUDGET = 8 * D * (CLIENTS + 1) * DENSE_ROUNDS
assert BUDGET == 17_711_326_080


def accuracy_percent(row):
    for key in ACC_KEYS:
        raw = row.get(key)
        if raw in (None, ''):
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            if v < 0 or v > 100:
                raise ValueError(f'Invalid accuracy for {key}: {v}')
            return 100 * v if v <= 1.5 else v
    raise ValueError('No global accuracy column: ' + ', '.join(row))


def get_row_at(path, round_target):
    with path.open(newline='', encoding='utf-8') as f:
        rows = [row for row in csv.DictReader(f) if row.get('round')]
    matches = [
        row for row in rows
        if int(float(row['round'])) == round_target
    ]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one R{round_target} row, got {len(matches)}')
    return matches[0]


def get_valid_output(root, prefix, round_target, expected_bytes):
    output_root = root / 'outputs'
    candidates = [output_root / prefix, *output_root.glob(prefix + '_*')]
    candidates = sorted(
        {p for p in candidates if p.is_dir()},
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    rejected = []
    for p in candidates:
        path = p / 'metrics/global_metrics.csv'
        if not path.is_file():
            rejected.append(f'{p.name}: metrics CSV missing')
            continue
        try:
            row = get_row_at(path, round_target)
            observed = int(float(row['cumulative_all_array_bytes']))
            if abs(observed - expected_bytes) > 1:
                raise ValueError(
                    f'byte mismatch at R{round_target}: {observed:,} vs expected '
                    f'{expected_bytes:,}; output may use old unicast accounting'
                )
            acc = accuracy_percent(row)
            return p, acc, observed
        except (ValueError, KeyError, OSError) as e:
            rejected.append(f'{p.name}: {e}')
    msg = f'Could not verify completed R{round_target} CIFAR run:\n  {prefix}'
    if rejected:
        msg += '\nCandidate issues:\n  ' + '\n  '.join(rejected)
    raise RuntimeError(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='.', help='BayesFL repository root')
    parser.add_argument('--output', default='reports/cifar_sparse_verified_results_for_paper.txt')
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    assert (root / 'outputs').is_dir(), f'Missing repo outputs/: {root}'

    fola_path, fola_acc, fola_bytes = get_valid_output(
        root, DENSE_PREFIX, 120, BUDGET
    )
    avg_path, avg_acc, avg_bytes = get_valid_output(
        root, FEDAVG_PREFIX, 240, BUDGET
    )

    lines = [
        'VERIFIED CIFAR-10 SPARSE RESULTS - MATCHED MULTICAST BYTES',
        '=================================================================',
        'N=20, alpha=0.01, local fraction=1.0, seed=0, BasicCNN (878538 params)',
        'Local E=10, B=32; FOLA LR=0.02 cosine horizon400; FedAvg LR=0.01 constant',
        'Both historical CIFAR baselines use gradient clipping norm=10.',
        f'Equal communication budget: {BUDGET:,} bytes = {BUDGET/1e9:.9f} GB',
        f'Dense FOLA R120: {fola_acc:.3f}% (bytes={fola_bytes:,})',
        f'Dense FedAvg R240: {avg_acc:.3f}% (bytes={avg_bytes:,})',
        f'Equal-byte Dense FOLA - FedAvg: {fola_acc-avg_acc:+.3f} pp',
        '',
        'Keep   Sparsity   Rounds   Comm GB      KL acc%   Random acc%   KL-Random pp   KL-Dense pp',
    ]
    provenance = [f'Dense FOLA: {fola_path}', f'FedAvg: {avg_path}']
    for keep in RATIOS:
        m = math.ceil(keep * D)
        c = 8*D + CLIENTS*(8*m + math.ceil(D/8))
        rounds = BUDGET // c
        vals = {}
        for rule in ('kl_global_local', 'random'):
            prefix = SPARSE_PATTERN.format(rule=rule, tag=TAGS[keep])
            vals[rule] = get_valid_output(root, prefix, rounds, rounds*c)
        kp, ka, kb = vals['kl_global_local']
        rp, ra, rb = vals['random']
        assert kb == rb
        lines.append(
            f'{keep:4.2f}      {100*(1-keep):5.0f}%   {rounds:6d}    '
            f'{kb/1e9:9.6f}    {ka:8.3f}     {ra:8.3f}       '
            f'{ka-ra:+8.3f}      {ka-fola_acc:+8.3f}'
        )
        provenance.append(f'KL keep {keep:.2f}: {kp}')
        provenance.append(f'Random keep {keep:.2f}: {rp}')
    lines.extend(['', 'Output run directories:', *provenance, '',
                  'All rows were verified against cumulative_all_array_bytes.',
                  'Accuracies use raw checkpoint metrics, not smoothed plot curves.',
                  'This is a selected seed, not a multiseed confidence interval.'])
    result = '\n'.join(lines) + '\n'
    dest = Path(args.output)
    if not dest.is_absolute():
        dest = root / dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(result, encoding='utf-8')
    print(result)
    print('SAVED:', dest)


if __name__ == '__main__':
    main()
