#!/usr/bin/env python3
"""Prepare safe checkpoint-resume queues for the locked CIFAR-10 sparse study.

Run from the BayesFL repository root with PYTHONPATH=$PWD/src.
The only output is two RESUME queues; no experiments start here.
"""
from pathlib import Path
import csv
import json
import math
import sys

from bayesfl.config import load_config
from bayesfl.communication import core_round_bytes
from bayesfl.experiment_state import config_fingerprint

ROOT = Path.cwd().resolve()
OUTPUTS = ROOT / "outputs"
QUEUES = ROOT / "scripts" / "queues"
D, N = 878_538, 20
DENSE_R120 = 120 * 8 * D * (N + 1)
DENSE_R150 = 150 * 8 * D * (N + 1)
assert DENSE_R120 == 17_711_326_080
assert DENSE_R150 == 22_139_157_600

# Retention, exact label used by existing CIFAR runs, prior R120-budget endpoint,
# timestamp of existing KL run, timestamp of existing Random run.
STUDIES = [
    (0.75, "0p75", 154, "20260924_194221", "20260925_032226"),
    (0.50, "0p5",  222, "20260924_194221", "20260925_043532"),
    (0.25, "0p25", 399, "20260925_090138", "20260925_194658"),
    (0.10, "0p1",  760, "20260925_071300", "20260925_192526"),
    (0.05, "0p05",1089, "20260927_193202", "20260928_024119"),
]

planned = {"kl_global_local": [], "random": []}
print("=" * 103)
print("CIFAR-10 R150 BUDGET: SAFE SPARSE RESUME PREFLIGHT")
print("Dense FOLA R150 budget:", f"{DENSE_R150:,}", "bytes")
print("=" * 103)

for keep, tag, old_target, kl_stamp, random_stamp in STUDIES:
    m = math.ceil(keep * D)
    per_round = core_round_bytes(
        D, N, method="fola", sparse=True, m=m,
        downlink_mode="multicast",
    )["total_bytes"]
    expected_old = DENSE_R120 // per_round
    new_target = DENSE_R150 // per_round
    assert expected_old == old_target, (keep, expected_old, old_target)
    assert (new_target + 1) * per_round > DENSE_R150

    for rule, stamp in [
        ("kl_global_local", kl_stamp), ("random", random_stamp)
    ]:
        config_name = (
            "run_sparse_cifar_basiccnn_a0p01_f1_fola_"
            f"{rule}_keep{tag}_cos400_n20_seed0_e10_b32_lr002_lam0p01_"
            "budget150dense"
        )
        name = f"{config_name}_{stamp}"
        run_dir = OUTPUTS / name
        if not run_dir.is_dir():
            raise FileNotFoundError(
                f"Existing {rule}, keep={keep:.2f} output not found: {run_dir}. "
                "DO NOT start a new run: check the directory name."
            )
        resolved = run_dir / "resolved_config.yaml"
        if not resolved.is_file():
            raise RuntimeError(f"Missing saved config: {resolved}")
        cfg = load_config(resolved)
        assert cfg.run_name == config_name, (cfg.run_name, config_name)
        assert cfg.method == "fola"
        assert cfg.data.dataset == "cifar10"
        assert cfg.data.partition == {
            "type": "paper_dirichlet",
            "dirichlet_alpha": 0.01,
            "local_data_fraction": 1.0,
        }
        assert cfg.model.name == "paper_basiccnn"
        assert cfg.federation.num_clients == N
        assert cfg.federation.clients_per_round == N
        assert cfg.runtime.seed == 0
        assert cfg.training.local_epochs == 10
        assert cfg.training.batch_size == 32
        assert cfg.training.lr == 0.02
        assert cfg.training.lr_schedule == "cosine"
        assert cfg.training.lr_decay_rounds == 400
        assert cfg.training.grad_clip_norm == 10.0
        assert cfg.fola.mode == "paper_reference"
        assert cfg.fola.prior_lambda == 0.01
        assert cfg.compression.selection_rule == rule
        assert math.isclose(cfg.compression.keep_ratio, keep, abs_tol=1e-12)
        assert cfg.compression.precision_policy == "floor_for_score"
        assert cfg.communication.deterministic_client_schedule
        assert cfg.communication.budget_metric == "cumulative_all_array_bytes"
        if cfg.communication.downlink_mode != "multicast":
            raise RuntimeError(
                f"Saved run is not using multicast accounting: {run_dir}. "
                "Do not switch accounting by editing files."
            )

        resume_dir = run_dir / "resume"
        pointer_path = resume_dir / "latest.json"
        if not pointer_path.is_file():
            raise RuntimeError(f"Missing resume pointer: {pointer_path}")
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        for key in ("state_file", "arrays_file"):
            if Path(pointer[key]).name != pointer[key]:
                raise RuntimeError(f"Unsafe pointer entry: {key}")
            if not (resume_dir / pointer[key]).is_file():
                raise RuntimeError(f"Missing committed {key}: {run_dir}")
        state = json.loads(
            (resume_dir / pointer["state_file"]).read_text(encoding="utf-8")
        )
        completed_round = int(state["round_id"])
        if completed_round < old_target:
            raise RuntimeError(
                f"Saved checkpoint only R{completed_round} for {run_dir.name}; "
                f"expected at least R{old_target}. Please inspect before continuing."
            )
        if completed_round > new_target:
            raise RuntimeError(
                f"Saved R{completed_round} exceeds new target R{new_target}: {run_dir}"
            )
        if state.get("downlink_mode") != "multicast":
            raise RuntimeError(
                f"Checkpoint is not committed with multicast accounting: {run_dir}"
            )
        if state["config_fingerprint"] != config_fingerprint(cfg):
            raise RuntimeError(f"Saved optimization fingerprint mismatch: {run_dir}")
        meta_path = run_dir / "partition_metadata.json"
        if not meta_path.is_file():
            raise RuntimeError(f"Missing partition metadata: {run_dir}")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if state["partition_sha256"] != metadata["sha256"]:
            raise RuntimeError(f"Saved partition SHA mismatch: {run_dir}")

        csv_file = run_dir / "metrics/global_metrics.csv"
        if not csv_file.is_file():
            raise RuntimeError(f"Missing metrics: {csv_file}")
        with csv_file.open(newline="", encoding="utf-8") as f:
            rows = [row for row in csv.DictReader(f) if row.get("round")]
        recorded_rounds = [int(float(row["round"])) for row in rows]
        if not recorded_rounds or recorded_rounds[-1] != completed_round:
            raise RuntimeError(f"Metrics end differs from saved R{completed_round}: {csv_file}")
        if len(recorded_rounds) != len(set(recorded_rounds)):
            raise RuntimeError(f"Duplicate round IDs: {csv_file}")
        if not set(range(1, completed_round + 1)).issubset(recorded_rounds):
            raise RuntimeError(f"Missing completed metrics rounds: {csv_file}")
        recorded_bytes = int(float(rows[-1]["cumulative_all_array_bytes"]))
        if recorded_bytes != completed_round * per_round:
            raise RuntimeError(
                f"Byte accounting mismatch in {csv_file}: "
                f"recorded={recorded_bytes:,}, expected={completed_round * per_round:,}"
            )

        status = "SKIP COMPLETE" if completed_round == new_target else "RESUME"
        print(
            f"{rule:16s} keep={keep:.2f} | R{completed_round:4d} -> R{new_target:4d} "
            f"| +{new_target-completed_round:3d} | {status}"
        )
        if completed_round < new_target:
            # The budget (not the round cap) is the effective stop condition.
            # Keep a generous safety cap as in the repository's older R150 queue.
            round_cap = max(2000, int(cfg.training.rounds), new_target + 1)
            planned[rule].append(
                f"RESUME|{run_dir}|{round_cap}|{DENSE_R150}|{new_target}"
            )

# Write queues only after ALL ten preflight checks pass.
STATUS_ONLY = (len(sys.argv) == 2 and sys.argv[1] == "--status")
if len(sys.argv) > 1 and not STATUS_ONLY:
    raise SystemExit("Usage: python scripts/prepare_cifar_r150_sparse_resume.py [--status]")
if not STATUS_ONLY:
    QUEUES.mkdir(parents=True, exist_ok=True)
for gpu, rule in [(0, "kl_global_local"), (1, "random")]:
    path = QUEUES / f"cifar_sparse_r150_resume_gpu{gpu}_{rule}.txt"
    content = "\n".join(planned[rule])
    if content:
        content += "\n"
    if not STATUS_ONLY:
        if path.is_file() and path.read_text(encoding="utf-8") != content:
            raise RuntimeError(
                f"Queue already exists with different content: {path}. "
                "Archive it before regenerating."
            )
        if not path.is_file():
            path.write_text(content, encoding="utf-8")
    print(f"GPU {gpu}: {len(planned[rule])} remaining jobs -> {path.relative_to(ROOT)}")

print("PASS: " + ("status checked" if STATUS_ONLY else "queues generated") + "; no training has started")
