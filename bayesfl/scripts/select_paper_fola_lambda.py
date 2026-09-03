#!/usr/bin/env python3
"""Select the best 20-round FOLA lambda under the hybrid CIFAR paper environment.

The data/model/client environment follows the Online Laplace paper, while the
user-requested optimizer settings remain fixed (lr=.05 cosine H=400, E=10,
batch=128, momentum=.9). The selected lambda is then promoted to 300 rounds.
"""

from __future__ import annotations

import csv
from pathlib import Path
import re
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
CONFIGS = ROOT / "scripts" / "configs" / "paper_sweep"

candidates = []
for run_dir in OUTPUTS.glob("paper_sweep_fola_cifar10_E10_lambda*_r20_*"):
    metrics = run_dir / "metrics" / "global_metrics.csv"
    if not metrics.exists():
        continue
    with metrics.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        continue
    final = rows[-1]
    acc = float(final.get("accuracy", final.get("fola_mean_accuracy", "nan")))
    m = re.search(r"E10_lambda([0-9]+)_r20", run_dir.name)
    if m:
        candidates.append((acc, int(m.group(1)), run_dir))

if not candidates:
    raise SystemExit(
        "No completed paper_sweep_fola_cifar10_E10_lambda*_r20_* outputs found. "
        "Run the lambda sweep first."
    )

candidates.sort(reverse=True)
print("20-round FOLA lambda ranking (E=10, kept optimizer settings):")
for acc, lam, run_dir in candidates:
    print(f"  lambda={lam:<5d} accuracy={acc:.6f}  {run_dir.name}")

best_acc, best_lambda, _ = candidates[0]
src = CONFIGS / f"fola_cifar10_E10_lambda{best_lambda}_r20.yaml"
with src.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
cfg["run_name"] = f"fola_cifar10_paper_env_keep_optim_selected_lambda{best_lambda}"
cfg["training"]["rounds"] = 300
cfg["output"]["checkpoint_every"] = 10

dst = ROOT / "scripts" / "configs" / "fola_cifar10_selected.yaml"
with dst.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print()
print(f"Selected lambda={best_lambda} (R20 accuracy={best_acc:.6f})")
print(f"Created: {dst.relative_to(ROOT)}")
print("Run with:")
print("  bash scripts/run_fola_cifar10.sh")
