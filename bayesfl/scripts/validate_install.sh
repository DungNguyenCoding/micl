#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

bash scripts/check_environment.sh
python -m pytest -q

python - <<'PY'
from pathlib import Path
from bayesfl.config import load_config
from bayesfl.models.factory import build_model, count_bayesian_random_variables

root = Path.cwd()
cfg = load_config(root / "scripts/configs/bbb_cifar10.yaml")
model = build_model(cfg)
d = count_bayesian_random_variables(model)
assert d == 851_514, d
print(f"CIFAR-10 BBB Bayesian dimension: {d:,} [OK]")
PY
