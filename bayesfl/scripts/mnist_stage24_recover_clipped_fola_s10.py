#!/usr/bin/env python3
"""Recover historical clipped dense MNIST FOLA with a new run_* name.

From ~/DungNDH/micl/bayesfl:
  PYTHONPATH="$PWD/src" /home/micl/anaconda3/envs/dungndh/bin/python \
    scripts/mnist_stage24_recover_clipped_fola_s10.py
"""
from pathlib import Path
import copy
import csv
import math
import yaml

from bayesfl.config import load_config
from bayesfl.communication import core_round_bytes
from bayesfl.data.datasets import partition_stem
from bayesfl.models.factory import initialize_model

ROOT = Path.cwd().resolve()
OLD_NAME = (
    "debug_mnist_stage12_cnn16_32_fixed1c_s10_n20_"
    "fola_seed0_e5_b10_lr0p10_lam0p01_r200"
)
NEW_NAME = (
    "run_mnist_stage24_cnn16_32_fixed1c_s10_n20_"
    "fola_clip_seed0_e5_b10_lr0p10_lam0p01_r200"
)
SRC = (ROOT / "scripts/configs/mnist_stage12_arch_fixed_s10_n20_seed0"
       / f"{OLD_NAME}.yaml")
DEST = (ROOT / "scripts/configs/mnist_stage24_recover_clipped_fola_s10"
        / f"{NEW_NAME}.yaml")
QUEUE = ROOT / "scripts/queues/mnist_stage24_recover_s10_clipped_fola_gpu0.txt"

if not SRC.is_file():
    raise FileNotFoundError(
        f"Original Stage-12 YAML not found: {SRC}\n"
        "Do not reconstruct optimizer settings from memory; locate the original YAML first."
    )
old = load_config(SRC)
assert old.run_name == OLD_NAME
assert old.method == "fola"
assert old.model.name == "mnist_cnn_16_32_128"
assert old.data.dataset == "mnist"
assert old.data.partition == {
    "type": "fixed_labels", "samples_per_client": 10, "labels_per_client": 1
}
assert (old.federation.num_clients, old.federation.clients_per_round) == (20, 20)
assert old.runtime.seed == 0
assert old.training.optimizer == "sgd"
assert old.training.local_epochs == 5
assert old.training.batch_size == 10
assert old.training.rounds == 200
assert old.training.lr == 0.10
assert old.training.lr_schedule == "constant"
assert old.training.lr_min == 0.10
assert old.training.momentum == 0.0
assert old.training.weight_decay == 0.0
assert old.training.grad_clip_norm == 10.0
assert old.fola.mode == "paper_reference"
assert old.fola.prior_lambda == 0.01
assert old.compression.selection_rule == "dense"
assert old.compression.keep_ratio == 1.0
assert old.communication.downlink_mode == "multicast"
assert old.communication.deterministic_client_schedule is True
assert old.communication.max_communication_bytes is None

d = sum(p.numel() for p in initialize_model(old).parameters())
assert d == 80_202
b = core_round_bytes(d, 20, method="fola", downlink_mode="multicast")["total_bytes"]
assert b == 13_473_936
assert 150 * b == 2_021_090_400
assert 200 * b == 2_694_787_200

raw = yaml.safe_load(SRC.read_text(encoding="utf-8"))
assert raw["run_name"] == OLD_NAME
raw["run_name"] = NEW_NAME   # Only modification to the historical configuration
DEST.parent.mkdir(parents=True, exist_ok=True)
QUEUE.parent.mkdir(parents=True, exist_ok=True)
if DEST.exists():
    assert yaml.safe_load(DEST.read_text(encoding="utf-8")) == raw, (
        f"Existing YAML differs; will not overwrite: {DEST}"
    )
else:
    DEST.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
new = load_config(DEST)
a = copy.deepcopy(old.to_dict())
c = copy.deepcopy(new.to_dict())
a.pop("run_name")
c.pop("run_name")
assert a == c, "Historical configuration was changed besides run_name"
assert partition_stem(old) == partition_stem(new)

# Avoid starting another run if this newly named recovery already exists.
outputs = ROOT / "outputs"
existing = sorted({p for p in [outputs / NEW_NAME, *outputs.glob(NEW_NAME + "_*")]
                   if p.is_dir()})
complete = False
for run in existing:
    resolved = run / "resolved_config.yaml"
    assert resolved.is_file(), f"Existing run has no resolved config: {run}"
    assert load_config(resolved).to_dict() == new.to_dict(), (
        f"Existing output has different experiment settings: {run}"
    )
    metrics = run / "metrics/global_metrics.csv"
    if not metrics.exists():
        continue
    with metrics.open(newline="", encoding="utf-8") as f:
        rows = {int(float(r["round"])): r for r in csv.DictReader(f)
                if r.get("round")}
    if all(r in rows for r in range(1, 201)):
        last_bytes = float(rows[200]["cumulative_all_array_bytes"])
        if math.isclose(last_bytes, 200 * b, rel_tol=0.0, abs_tol=1.0):
            complete = True
            break
if existing and not complete:
    raise RuntimeError(
        f"A recovery output already exists but is incomplete: {existing}\n"
        "Preserve it and resume; do not restart the run."
    )

entry = DEST.relative_to(ROOT).as_posix() + "\n"
if QUEUE.exists() and QUEUE.read_text(encoding="utf-8") not in (entry, ""):
    raise RuntimeError(f"Queue conflict; inspect this file: {QUEUE}")
QUEUE.write_text("" if complete else entry, encoding="utf-8")
print("PASS:", "Already complete (skipped)" if complete else "One job queued")
print("Run name:", NEW_NAME)
print("Config  :", DEST.relative_to(ROOT))
print("Queue   :", QUEUE.relative_to(ROOT))
print("Method  : Dense FOLA, clipped SGD norm=10, seed 0, N=20, s10")
print("Train   : R200; plot at R150 = 2.0210904 GB")
