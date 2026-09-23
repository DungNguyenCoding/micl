#!/usr/bin/env python
"""Print observed results for an explicit generated study; optionally make plots.

Run directories are matched by exact configured names. Multiple matching runs
require --latest, which never silently falls back from an incomplete newest run.
No test accuracies or communication histories are inferred from filenames.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

from bayesfl.communication_plots import load_curve, plot_comparison, print_summary
from bayesfl.config import load_config


def collect(study_dir: Path, *, latest: bool=False, outputs_dir: Path | None=None):
    manifest=json.loads((study_dir/'study_manifest.json').read_text())
    curves=[]
    for run in manifest['runs']:
        cfg=load_config(run['config'])
        root=outputs_dir or Path(cfg.output.outputs_dir)
        pattern=re.compile(re.escape(cfg.run_name)+r'_\d{8}_\d{6}$')
        candidates=[p for p in root.glob(cfg.run_name+'_*') if p.is_dir() and pattern.fullmatch(p.name)]
        if not candidates:
            raise FileNotFoundError(f'No actual run found for {cfg.run_name} under {root}')
        if len(candidates)>1 and not latest:
            raise ValueError(f'{cfg.run_name}: {len(candidates)} runs; use explicit --run-dir plotting or --latest')
        path=max(candidates,key=lambda p:p.stat().st_mtime)
        actual=load_config(path/'resolved_config.yaml')
        if actual.to_dict()!=cfg.to_dict():
            raise ValueError(f'{path}: resolved configuration differs from the selected study (including limits)')
        summary_path=path/'run_summary.json'
        if not summary_path.exists():raise ValueError(f'{path}: no successful completion summary')
        summary=json.loads(summary_path.read_text())
        if summary.get('stop_reason') not in ('max_rounds','communication_budget'):
            raise ValueError(f'{path}: run is failed/incomplete')
        curve=load_curve(path)
        if [r['round'] for r in curve.rows]!=list(range(int(summary['round_id'])+1)):
            raise ValueError(f'{path}: missing evaluation rounds')
        curves.append(curve)
    return curves


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir',type=Path,required=True)
    parser.add_argument('--outputs-dir',type=Path)
    parser.add_argument('--latest',action='store_true')
    parser.add_argument('--budget-bytes',type=int)
    parser.add_argument('--plot-dir',type=Path)
    args=parser.parse_args()
    curves=collect(args.study_dir,latest=args.latest,outputs_dir=args.outputs_dir)
    for c in curves: print('Observed run:',c.run_dir)
    print_summary(curves,budget=args.budget_bytes)
    if args.plot_dir:
        for path in plot_comparison(curves,args.plot_dir):print(path)


if __name__=='__main__':main()
