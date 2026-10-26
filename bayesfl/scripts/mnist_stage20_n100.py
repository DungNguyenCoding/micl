#!/usr/bin/env python3
"""N=100 CNN FOLA/FedAvg, with and without gradient clipping."""

from collections import Counter, defaultdict
from pathlib import Path
import argparse
import csv
import json
import math
import yaml

from bayesfl.config import load_config
from bayesfl.models.factory import initialize_model
from bayesfl.data.datasets import prepare_partition, partition_stem
from bayesfl.data.partition import load_partition
from bayesfl.communication import core_round_bytes

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
CONFIG_DIR = ROOT / "scripts/configs/mnist_stage20_n100_cnn"
QUEUE_DIR = ROOT / "scripts/queues"

SAMPLES = (5, 10)
SEEDS = range(5)
METHODS = ("fola", "fedavg")
POLICIES = ("noclip", "clip")

N = 100
DIM = 80202

MILESTONES = (
    (120, 240),
    (150, 300),
    (200, 400),
)


# ============================================================
# Historical templates
# ============================================================

def base_path(samples, seed, method):

    if method == "fola":
        name = (
            "run_mnist_stage18_cnn16_32_fola_noclip_"
            f"fixed1c_s{samples}_n20_seed{seed}_"
            f"e5_b{samples}_lr0p10_lam0p01_r200"
        )

        return (
            ROOT
            / "scripts/configs/mnist_stage18_fola_noclip_cnn"
            / (name + ".yaml")
        )

    name = (
        "run_stage16_paper00_fedavg_cnn16_32_"
        f"s{samples}_n20_seed{seed}_"
        f"e5_b{samples}_lr0p10_r400"
    )

    return (
        ROOT
        / "scripts/configs/mnist_stage16_paper00_fedavg"
        / (name + ".yaml")
    )


def case_name(s, seed, method, policy):

    cap = 200 if method == "fola" else 400

    return (
        f"run_mnist_stage20_n100_cnn_s{s}_"
        f"seed{seed}_{method}_{policy}_r{cap}"
    )


def desired_config(s, seed, method, policy):

    template = base_path(s, seed, method)

    if not template.is_file():
        raise FileNotFoundError(
            f"Missing historical template: {template}"
        )

    with template.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    assert raw["method"] == method
    assert raw["runtime"]["seed"] == seed

    assert raw["model"]["name"] == "mnist_cnn_16_32_128"

    assert raw["data"]["partition"] == {
        "type": "fixed_labels",
        "samples_per_client": s,
        "labels_per_client": 1,
    }

    assert raw["training"]["local_epochs"] == 5
    assert raw["training"]["batch_size"] == s
    assert raw["training"]["lr"] == 0.10
    assert raw["training"]["lr_schedule"] == "constant"
    assert raw["training"]["momentum"] == 0.0
    assert raw["training"]["weight_decay"] == 0.0

    assert raw["compression"]["selection_rule"] == "dense"
    assert raw["compression"]["keep_ratio"] == 1.0

    assert raw["communication"]["downlink_mode"] == "multicast"

    if method == "fola":
        assert raw["fola"]["mode"] == "paper_reference"
        assert raw["fola"]["prior_lambda"] == 0.01
        assert raw["training"]["grad_clip_norm"] is None

    else:
        assert raw["training"]["paper00_strict"] is True
        assert raw["training"]["grad_clip_norm"] is None

    # Change the number of clients.
    raw["run_name"] = case_name(s, seed, method, policy)

    raw["federation"]["num_clients"] = N
    raw["federation"]["clients_per_round"] = N

    # Keep the existing milestone horizons.
    raw["training"]["rounds"] = (
        200 if method == "fola" else 400
    )

    # Compare clipped and unclipped training.
    raw["training"]["grad_clip_norm"] = (
        10.0 if policy == "clip" else None
    )

    raw["training"]["paper00_strict"] = (
        method == "fedavg" and policy == "noclip"
    )

    # Dense communication, multicast downlink.
    raw["communication"]["max_communication_bytes"] = None
    raw["communication"]["deterministic_client_schedule"] = True
    raw["communication"]["downlink_mode"] = "multicast"

    raw["compression"]["selection_rule"] = "dense"
    raw["compression"]["keep_ratio"] = 1.0

    raw["output"]["checkpoint_every"] = 10
    raw["output"]["save_full_client_posteriors"] = False

    return raw


# ============================================================
# Compare complete experiment configurations
# ============================================================

def profile(cfg):

    d = cfg.to_dict()

    # Names and output directories do not affect training.
    d.pop("run_name", None)
    d.pop("output", None)

    # A longer completed run may contain an earlier milestone.
    d["training"].pop("rounds", None)

    # Communication budget is a stopping condition.
    d["communication"].pop(
        "max_communication_bytes", None
    )

    return json.dumps(
        d,
        sort_keys=True,
        allow_nan=False,
    )


def make_index():

    by_profile = defaultdict(list)
    incompatible = 0

    for path in OUTPUTS.glob("*/resolved_config.yaml"):

        try:
            cfg = load_config(path)

            by_profile[
                profile(cfg)
            ].append(path.parent)

        except Exception:
            incompatible += 1

    print(
        "\nEXISTING OUTPUT SCAN:"
        f"\n  Readable configs: "
        f"{sum(map(len, by_profile.values()))}"
        f"\n  Incompatible historical configs: {incompatible}"
    )

    return by_profile


def completed(run_dir, cfg, expected_sha):

    metrics = (
        run_dir / "metrics/global_metrics.csv"
    )

    if not metrics.is_file():
        return False

    try:
        with metrics.open(
            newline="",
            encoding="utf-8",
        ) as f:

            rows = {
                int(float(row["round"])): row
                for row in csv.DictReader(f)
                if row.get("round")
            }

        cap = int(cfg.training.rounds)

        if not all(
            r in rows
            for r in range(1, cap + 1)
        ):
            return False

        checkpoints = (
            (120, 150, 200)
            if cfg.method == "fola"
            else (240, 300, 400)
        )

        cost = core_round_bytes(
            DIM,
            N,
            method=cfg.method,
            downlink_mode="multicast",
        )["total_bytes"]

        # Verify recorded, not just theoretical, communication.
        for r in checkpoints:

            recorded = float(
                rows[r]["cumulative_all_array_bytes"]
            )

            if not math.isfinite(recorded):
                return False

            if not math.isclose(
                recorded,
                cost * r,
                rel_tol=1e-10,
                abs_tol=1.0,
            ):
                return False

        # Verify the actual partition.
        metadata = run_dir / "partition_metadata.json"

        if not metadata.is_file():
            return False

        saved_sha = json.loads(
            metadata.read_text()
        ).get("sha256")

        if saved_sha != expected_sha:
            return False

        # Verify checkpoint if available.
        pointer = run_dir / "resume/latest.json"

        if pointer.is_file():

            info = json.loads(pointer.read_text())

            state = json.loads(
                (pointer.parent / info["state_file"]).read_text()
            )

            if int(state["round_id"]) < cap:
                return False

        return True

    except (
        ValueError,
        KeyError,
        OSError,
        json.JSONDecodeError,
    ):
        return False


# ============================================================
# Verify all settings, initializations and partitions
# ============================================================

def check_configs(cases):

    import torch
    import numpy as np
    from torchvision.datasets import MNIST

    partition_sha = {}
    partition_stems = set()

    for s in SAMPLES:
        for seed in SEEDS:

            group = []

            for method in METHODS:
                for policy in POLICIES:

                    cfg = cases[(s, seed, method, policy)]

                    assert cfg.federation.num_clients == 100
                    assert cfg.federation.clients_per_round == 100

                    assert cfg.runtime.seed == seed
                    assert cfg.method == method

                    assert cfg.model.name == "mnist_cnn_16_32_128"

                    assert cfg.training.lr == 0.10
                    assert cfg.training.local_epochs == 5
                    assert cfg.training.batch_size == s

                    assert cfg.training.rounds == (
                        200 if method == "fola" else 400
                    )

                    assert cfg.training.grad_clip_norm == (
                        10.0 if policy == "clip" else None
                    )

                    assert cfg.training.paper00_strict is (
                        method == "fedavg"
                        and policy == "noclip"
                    )

                    assert cfg.communication.downlink_mode == "multicast"
                    assert cfg.communication.max_communication_bytes is None

                    assert cfg.compression.selection_rule == "dense"
                    assert cfg.compression.keep_ratio == 1.0

                    if method == "fola":
                        assert cfg.fola.mode == "paper_reference"
                        assert cfg.fola.prior_lambda == 0.01

                    group.append(cfg)

            # All four methods must use the same partition.
            stems = {
                partition_stem(c)
                for c in group
            }

            assert len(stems) == 1
            partition_stems.update(stems)

            # All four methods start with identical parameters.
            states = [
                initialize_model(c).state_dict()
                for c in group
            ]

            assert all(
                sum(v.numel() for v in x.values()) == DIM
                for x in states
            )

            for other in states[1:]:
                assert all(
                    torch.equal(states[0][k], other[k])
                    for k in states[0]
                )

            # Cache and verify this N=100 partition.
            path, metadata = prepare_partition(group[0])
            parts = load_partition(path)

            labels = MNIST(
                root=group[0].data.root,
                train=True,
                download=False,
            ).targets.numpy()

            assert len(parts) == N
            assert all(len(p) == s for p in parts)

            all_indices = np.concatenate(parts)

            assert (
                len(all_indices)
                == len(np.unique(all_indices))
                == N * s
            )

            class_counts = Counter(
                int(labels[p[0]])
                for p in parts
            )

            assert all(
                len(np.unique(labels[p])) == 1
                for p in parts
            )

            assert all(
                class_counts.get(digit, 0) == 10
                for digit in range(10)
            )

            partition_sha[(s, seed)] = metadata["sha256"]

            print(
                f"PARTITION PASS | s{s}, seed{seed} | "
                f"{N*s} unique samples | "
                "10 clients per digit"
            )

    assert len(partition_stems) == 10

    # Verify communication-matched milestones.
    fb = core_round_bytes(
        DIM, N,
        method="fola",
        downlink_mode="multicast",
    )["total_bytes"]

    ab = core_round_bytes(
        DIM, N,
        method="fedavg",
        downlink_mode="multicast",
    )["total_bytes"]

    assert fb == 2 * ab

    for rf, ra in MILESTONES:

        assert fb * rf == ab * ra

        print(
            f"MATCHED COST | FOLA R{rf} / FedAvg R{ra} | "
            f"{rf*fb/1e9:.3f} GB"
        )

    return partition_sha


# ============================================================
# Generate the 40 intended experiment profiles
# ============================================================

def all_cases():

    return [
        (s, seed, method, policy)
        for s in SAMPLES
        for seed in SEEDS
        for method in METHODS
        for policy in POLICIES
    ]


def main(action):

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)

    configs = {}
    paths = {}

    for s, seed, method, policy in all_cases():

        raw = desired_config(
            s, seed, method, policy
        )

        path = (
            CONFIG_DIR / (raw["run_name"] + ".yaml")
        )

        if path.exists():

            assert yaml.safe_load(
                path.read_text()
            ) == raw, (
                f"Configuration conflict: {path}"
            )

        elif action == "prepare":

            path.write_text(
                yaml.safe_dump(
                    raw,
                    sort_keys=False,
                )
            )

        if not path.is_file():
            raise RuntimeError(
                f"Run prepare first: {path}"
            )

        cfg = load_config(path)

        key = (s, seed, method, policy)

        configs[key] = cfg
        paths[key] = path.relative_to(ROOT).as_posix()

    # Check matching partitions and initializations.
    sha = check_configs(configs)

    # Search all previously completed experiments.
    existing = make_index()

    gpu_jobs = {0: [], 1: []}

    reused = []
    blocked = []

    for s, seed, method, policy in all_cases():

        key = (s, seed, method, policy)
        cfg = configs[key]

        matches = existing.get(
            profile(cfg),
            [],
        )

        good = [
            directory
            for directory in matches
            if completed(
                directory,
                cfg,
                sha[(s, seed)],
            )
        ]

        label = (
            f"s{s} seed{seed} "
            f"{method:6s} {policy:6s} "
            f"R{cfg.training.rounds}"
        )

        if good:

            directory = max(
                good,
                key=lambda x: x.stat().st_mtime,
            )

            reused.append((key, directory))

            print(
                f"REUSE COMPLETE | {label} | {directory}"
            )

            continue

        # Prevent duplicate training of partially started runs.
        collisions = [
            d for d in [
                OUTPUTS / cfg.run_name,
                *OUTPUTS.glob(cfg.run_name + "_*"),
            ]
            if d.is_dir()
        ]

        if matches or collisions:

            blocked.append(
                (key, matches or collisions)
            )

            print(
                f"BLOCK PARTIAL | {label} | "
                f"{matches or collisions}"
            )

            continue

        # Assign paired experiments by seed and sample count.
        # With no existing completions: 20 jobs per GPU.
        gpu = (seed + (s == 10)) % 2

        gpu_jobs[gpu].append(paths[key])

        print(
            f"QUEUE GPU{gpu} | {label}"
        )

    if blocked:
        raise RuntimeError(
            f"{len(blocked)} partial/conflicting runs found. "
            "No queue was overwritten. "
            "Inspect or resume these runs before proceeding."
        )

    # Save the two filtered queues.
    for gpu in (0, 1):

        queue = QUEUE_DIR / (
            f"mnist_stage20_n100_gpu{gpu}.txt"
        )

        jobs = gpu_jobs[gpu]

        contents = (
            "\n".join(jobs)
            + ("\n" if jobs else "")
        )

        if action == "prepare":

            if (
                queue.exists()
                and queue.read_text() != contents
            ):
                raise RuntimeError(
                    f"Existing queue differs: {queue}. "
                    "Archive the old queue before regenerating."
                )

            queue.write_text(contents)

        print(
            f"GPU{gpu}: {len(jobs)} new jobs | "
            f"{queue.relative_to(ROOT)}"
        )

    print()
    print("=" * 72)
    print("N=100 CNN EXPERIMENT PREPARATION")
    print("=" * 72)

    print("Possible experiments :", len(all_cases()))
    print("Already completed    :", len(reused))
    print(
        "New jobs             :",
        len(gpu_jobs[0]) + len(gpu_jobs[1])
    )
    print("Blocked partial runs :", len(blocked))

    print("GPU 0 jobs           :", len(gpu_jobs[0]))
    print("GPU 1 jobs           :", len(gpu_jobs[1]))

    print("PASS")

    if action == "status":
        print(
            "Status only — existing queue files "
            "were not modified."
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "action",
        choices=("prepare", "status"),
    )

    main(parser.parse_args().action)
