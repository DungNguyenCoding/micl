"""Plot recorded accuracy against rounds/bytes and print common-budget results.

Never infer byte histories from rounds, use best-test accuracy as a checkpoint,
or extrapolate a round-capped run to a larger communication budget.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class RunCurve:
    run_dir: Path
    rows: list[dict]
    label: str
    metric: str
    coverage_end: int
    seed: int

    def at_budget(self, budget: int) -> dict:
        if budget < 0 or budget > self.coverage_end:
            raise ValueError(f"{self.run_dir.name}: budget {budget} outside covered range [0,{self.coverage_end}]")
        eligible = [r for r in self.rows if r[self.metric] <= budget]
        if not eligible:
            raise ValueError(f"{self.run_dir.name}: no evaluated checkpoint at/below budget {budget}")
        return eligible[-1]


def load_curve(run_dir: Path, *, label: str | None = None,
               metric: str = "cumulative_all_array_bytes") -> RunCurve:
    run_dir = Path(run_dir)
    path = run_dir / "metrics" / "global_metrics.csv"
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if metric not in fields:
            raise ValueError(f"{path}: no {metric}. Historical accuracy-only logs are not byte-aware runs.")
        accuracy_key = ("fola_mean_accuracy" if "fola_mean_accuracy" in fields else
                        "global_accuracy" if "global_accuracy" in fields else "accuracy")
        rows = []
        for raw in reader:
            round_float = float(raw.get("round_id") or raw["round"])
            if not math.isfinite(round_float) or not round_float.is_integer():
                raise ValueError(f"Invalid round in {path}")
            rnd = int(round_float)
            cost = int(raw[metric])  # Never parse integer byte counters through float32/float64.
            acc = float(raw[accuracy_key])
            if cost < 0 or not math.isfinite(acc) or not 0 <= acc <= 1:
                raise ValueError(f"Invalid accuracy/bytes at round {rnd} in {path}")
            if rows and (rnd <= rows[-1]["round"] or cost < rows[-1][metric]):
                raise ValueError(f"Duplicate/nonmonotone evaluation history in {path}")
            rows.append({**raw, "round": rnd, metric: cost, "accuracy": acc})
    if not rows:
        raise ValueError(f"Empty metric history: {path}")
    coverage = rows[-1][metric]
    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary.get("array_accounting_complete") is False:
            raise ValueError("Run contains undecodable packets with unknown array payload; not an exact byte comparison")
        if summary.get("stop_reason") == "communication_budget" and summary.get("budget_metric") == metric:
            # The requested budget was exhausted to within one whole round. The
            # final actual checkpoint is the model achieved under that budget.
            coverage = max(coverage, int(summary["budget_bytes"]))
    method = str(rows[-1].get("method") or run_dir.name)
    ratio = float(rows[-1].get("keep_ratio_requested") or 1)
    default_label = method + (f" (keep={ratio:g})" if "sparse" in method else "")
    return RunCurve(run_dir, rows, label or default_label, metric, coverage,
                    int(rows[-1].get("seed") or 0))


def print_summary(curves: list[RunCurve], *, budget: int | None = None) -> None:
    if not curves:
        raise ValueError("At least one run is required")
    if len({c.metric for c in curves}) != 1:
        raise ValueError('All runs must use the same communication accounting metric')
    common = min(c.coverage_end for c in curves) if budget is None else budget
    print(f"Common budget: {common:,} bytes ({common/1_000_000:g} MB), metric={curves[0].metric}")
    print("Accuracy is the latest evaluated model at/below the budget, not best-test accuracy.")
    print(f"{'Run / method':62} {'Round':>7} {'Used bytes':>17} {'Accuracy':>11}")
    for curve in curves:
        row = curve.at_budget(common)
        print(f"{curve.label[:62]:62} {row['round']:7d} {row[curve.metric]:17,d} {100*row['accuracy']:10.3f}%")


def _groups(curves: list[RunCurve], average_seeds: bool):
    if not average_seeds:
        return [(c.label, [c]) for c in curves]
    grouped = {}
    for curve in curves:
        grouped.setdefault(curve.label, []).append(curve)
    seed_sets = []
    for label, runs in grouped.items():
        seeds = [c.seed for c in runs]
        if len(set(seeds)) != len(seeds) or len(seeds) < 2:
            raise ValueError(f"Seed averaging needs >=2 distinct, explicitly matched seeds for {label}")
        seed_sets.append(set(seeds))
        signature_keys = ("method", "keep_ratio_requested", "d", "precision_policy", "score_precision_floor")
        expected = tuple(runs[0].rows[-1].get(k) for k in signature_keys)
        for run in runs[1:]:
            if tuple(run.rows[-1].get(k) for k in signature_keys) != expected:
                raise ValueError(f"Mismatched method/protocol settings in group {label}")
            if [(r['round'],r[run.metric]) for r in run.rows] != [(r['round'],r[runs[0].metric]) for r in runs[0].rows]:
                raise ValueError("Seed averaging requires identical evaluated round/byte grids; no missing-point interpolation")
        # Resolved configs add a stricter data/model/trainer check when available.
        import yaml
        configs = []
        for run in runs:
            path = run.run_dir / 'resolved_config.yaml'
            if not path.exists():
                raise ValueError("Seed averaging requires each run's resolved_config.yaml")
            data = yaml.safe_load(path.read_text())
            configs.append({key: data.get(key) for key in ('data','federation','model','training','fola','compression')})
        if any(c != configs[0] for c in configs[1:]):
            raise ValueError(f"Data/model/training settings differ across seeds for {label}")
    if any(s != seed_sets[0] for s in seed_sets[1:]):
        raise ValueError("Compared methods must contain the same seed IDs")
    return list(grouped.items())


def plot_comparison(curves: list[RunCurve], output_dir: Path, *, average_seeds: bool = False) -> list[Path]:
    import matplotlib.pyplot as plt
    if not curves or len({c.metric for c in curves}) != 1:
        raise ValueError('Plot requires runs with one shared communication metric')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = _groups(curves, average_seeds)
    paths = []
    for axis in ('round', 'communication'):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for label, runs in groups:
            first = runs[0]
            x = np.array([r['round'] if axis == 'round' else r[first.metric]/1_000_000 for r in first.rows])
            values = np.array([[100*r['accuracy'] for r in c.rows] for c in runs])
            mean = values.mean(axis=0)
            suffix = f" (mean, {len(runs)} seeds)" if len(runs)>1 else f" [seed {first.seed}]"
            if axis == 'communication':
                ax.step(x, mean, where='post', label=label+suffix, marker='.', markersize=3)
            else:
                ax.plot(x, mean, label=label+suffix)
            if len(runs)>1:
                sd = values.std(axis=0, ddof=1)
                ax.fill_between(x, mean-sd, mean+sd, alpha=.14,
                                step='post' if axis=='communication' else None)
        ax.set_ylabel('Global accuracy (%)')
        ax.set_xlabel('Completed global round' if axis=='round' else
                      'Cumulative logical array payload (MB; uplink + downlink)')
        scope = 'All model arrays, including any initialization/evaluation arrays' if curves[0].metric=='cumulative_all_array_bytes' else 'Training model arrays only'
        ax.set_title('Accuracy versus rounds' if axis=='round' else 'Accuracy versus communication\n'+scope)
        ax.grid(True, alpha=.2)
        ax.legend(fontsize=7, loc='best')
        fig.tight_layout()
        path = output_dir / f'accuracy_vs_{axis}.png'
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    csv_path = output_dir / 'observed_accuracy_communication.csv'
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['run_dir','label','seed','round','array_bytes','accuracy'])
        writer.writeheader()
        for curve in curves:
            for row in curve.rows:
                writer.writerow(dict(run_dir=str(curve.run_dir.resolve()),label=curve.label,seed=curve.seed,
                                     round=row['round'],array_bytes=row[curve.metric],accuracy=row['accuracy']))
    paths.append(csv_path)
    manifest=dict(accounting='logical_array_payload_not_on_wire', unit='MB = 1,000,000 bytes',
                  metric=curves[0].metric, average_seeds=average_seeds,
                  uncertainty_band='sample standard deviation, ddof=1' if average_seeds else None,
                  source_runs=[str(c.run_dir.resolve()) for c in curves],
                  no_extrapolation=True, accuracy='latest evaluated model, never best-so-far')
    (output_dir/'comparison_manifest.json').write_text(json.dumps(manifest,indent=2))
    return paths


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',action='append',type=Path,required=True)
    parser.add_argument('--label',action='append')
    parser.add_argument('--metric',choices=['cumulative_all_array_bytes','cumulative_train_total_array_bytes'],
                        default='cumulative_all_array_bytes')
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/plots/sparse_comparison'))
    parser.add_argument('--budget-bytes',type=int)
    parser.add_argument('--summary-only',action='store_true')
    parser.add_argument('--average-seeds',action='store_true',help='Average matching repeated labels; requires matched distinct seeds and full grids')
    args=parser.parse_args()
    if args.label is not None and len(args.label)!=len(args.run_dir):
        parser.error('Provide exactly one --label per --run-dir')
    labels=args.label or [None]*len(args.run_dir)
    curves=[load_curve(p,label=label,metric=args.metric) for p,label in zip(args.run_dir,labels)]
    print_summary(curves,budget=args.budget_bytes)
    if not args.summary_only:
        for path in plot_comparison(curves,args.output_dir,average_seeds=args.average_seeds):
            print(path)


if __name__=='__main__':
    main()
