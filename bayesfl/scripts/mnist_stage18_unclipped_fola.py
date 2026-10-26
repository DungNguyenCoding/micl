#!/usr/bin/env python3
"""Unclipped FOLA R200 versus strict Paper-00 FedAvg R400."""

from pathlib import Path
from statistics import mean, stdev, pstdev
from math import sqrt, isclose, isfinite
import argparse
import csv
import json
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scripts/configs/mnist_stage18_fola_noclip_cnn"

SAMPLES = (5, 10)
SEEDS = range(5)
MILESTONES = ((120, 240), (150, 300), (200, 400))
PARAMS = 80202
T95 = 2.7764451051977987  # Student t, df=4


# ============================================================
# Exact experiment configuration paths
# ============================================================

def historical_path(s, seed):
    if seed == 0:
        stage = 12 if s == 10 else 14
        directory = f"mnist_stage{stage}_arch_fixed_s{s}_n20_seed0"
        name = (
            f"debug_mnist_stage{stage}_cnn16_32_"
            f"fixed1c_s{s}_n20_fola_seed0_e5_b{s}_"
            "lr0p10_lam0p01_r200"
        )
    else:
        directory = "mnist_stage15_cnn_multiseed"
        name = (
            "run_mnist_stage15_cnn16_32_"
            f"fixed1c_s{s}_n20_fola_seed{seed}_e5_b{s}_"
            "lr0p10_lam0p01_r150"
        )

    return ROOT / "scripts/configs" / directory / (name + ".yaml")


def strict_path(s, seed):
    name = (
        "run_stage16_paper00_fedavg_cnn16_32_"
        f"s{s}_n20_seed{seed}_e5_b{s}_lr0p10_r400"
    )
    return (
        ROOT / "scripts/configs/mnist_stage16_paper00_fedavg"
        / (name + ".yaml")
    )


def new_name(s, seed):
    return (
        "run_mnist_stage18_cnn16_32_fola_noclip_"
        f"fixed1c_s{s}_n20_seed{seed}_e5_b{s}_"
        "lr0p10_lam0p01_r200"
    )


def new_path(s, seed):
    return OUT / (new_name(s, seed) + ".yaml")


# ============================================================
# Validate settings and pair compatibility
# ============================================================

def check_pair(f, a, s, seed, *, clipped):
    from bayesfl.data.datasets import partition_stem

    assert f.method == "fola" and a.method == "fedavg"
    assert f.model.name == a.model.name == "mnist_cnn_16_32_128"
    assert f.runtime.seed == a.runtime.seed == seed
    assert partition_stem(f) == partition_stem(a)

    assert f.data.partition == a.data.partition == {
        "type": "fixed_labels",
        "samples_per_client": s,
        "labels_per_client": 1,
    }

    assert f.data.dataset == a.data.dataset == "mnist"
    assert f.federation.num_clients == a.federation.num_clients == 20
    assert f.federation.clients_per_round == a.federation.clients_per_round == 20

    assert f.training.lr == a.training.lr == 0.10
    assert f.training.local_epochs == a.training.local_epochs == 5
    assert f.training.batch_size == a.training.batch_size == s
    assert f.training.lr_schedule == a.training.lr_schedule == "constant"

    assert f.training.momentum == a.training.momentum == 0.0
    assert f.training.weight_decay == a.training.weight_decay == 0.0
    assert f.training.optimizer.lower() == a.training.optimizer.lower() == "sgd"

    assert f.training.grad_clip_norm == (10.0 if clipped else None)
    assert f.training.paper00_strict is False

    assert f.fola.mode == "paper_reference"
    assert f.fola.prior_lambda == 0.01

    assert a.training.grad_clip_norm is None
    assert a.training.paper00_strict is True

    assert f.compression.selection_rule == a.compression.selection_rule == "dense"
    assert f.compression.keep_ratio == a.compression.keep_ratio == 1.0

    assert f.communication.downlink_mode == a.communication.downlink_mode == "multicast"
    assert f.communication.max_communication_bytes is None
    assert a.communication.max_communication_bytes is None


# ============================================================
# PREPARE — Generate and validate 10 new configs
# ============================================================

def prepare():
    import torch
    from bayesfl.config import load_config
    from bayesfl.models.factory import initialize_model
    from bayesfl.communication import core_round_bytes
    from bayesfl.data.datasets import partition_stem

    OUT.mkdir(parents=True, exist_ok=True)
    jobs = {0: [], 1: []}

    for s in SAMPLES:
        for seed in SEEDS:

            source = historical_path(s, seed)
            strict = strict_path(s, seed)

            assert source.is_file(), f"Missing historical FOLA YAML: {source}"
            assert strict.is_file(), f"Missing strict FedAvg YAML: {strict}"

            raw = yaml.safe_load(source.read_text())

            assert raw["training"]["grad_clip_norm"] == 10.0

            # Only change run identity, round cap, and clipping.
            raw["run_name"] = new_name(s, seed)
            raw["training"]["rounds"] = 200
            raw["training"]["grad_clip_norm"] = None
            raw["training"]["paper00_strict"] = False

            dest = new_path(s, seed)

            if dest.exists():
                assert yaml.safe_load(dest.read_text()) == raw, (
                    f"Configuration conflict: {dest}"
                )
            else:
                assert not list(
                    (ROOT / "outputs").glob(new_name(s, seed) + "_*")
                ), f"Existing output for {new_name(s, seed)}"

                dest.write_text(yaml.safe_dump(raw, sort_keys=False))

            cfg = load_config(dest)
            hist = load_config(source)
            fed = load_config(strict)

            check_pair(cfg, fed, s, seed, clipped=False)
            check_pair(hist, fed, s, seed, clipped=True)

            assert cfg.training.rounds == 200
            assert fed.training.rounds == 400
            assert partition_stem(cfg) == partition_stem(hist)

            # Verify exact intended YAML changes.
            expected = yaml.safe_load(source.read_text())
            expected["run_name"] = new_name(s, seed)
            expected["training"].update(
                rounds=200,
                grad_clip_norm=None,
                paper00_strict=False,
            )
            assert raw == expected

            # Identical initial parameters.
            mf = initialize_model(cfg)
            ma = initialize_model(fed)

            assert sum(p.numel() for p in mf.parameters()) == PARAMS
            assert mf.state_dict().keys() == ma.state_dict().keys()
            assert all(
                torch.equal(mf.state_dict()[k], ma.state_dict()[k])
                for k in mf.state_dict()
            ), f"Initialization mismatch: s{s}, seed{seed}"

            # Verify all three equal-communication budgets.
            fb = core_round_bytes(
                PARAMS, 20,
                method="fola",
                downlink_mode="multicast",
            )["total_bytes"]

            ab = core_round_bytes(
                PARAMS, 20,
                method="fedavg",
                downlink_mode="multicast",
            )["total_bytes"]

            assert fb == 2 * ab

            for rf, ra in MILESTONES:
                assert rf * fb == ra * ab

            # Balance each GPU to five runs.
            gpu = (seed + (s == 10)) % 2
            jobs[gpu].append(dest.relative_to(ROOT).as_posix())

            print(
                f"PASS | s{s:2d} seed{seed} | "
                f"FOLA no-clip R200 | strict FedAvg R400 | "
                f"budget={200*fb/1e9:.3f} GB | GPU{gpu}"
            )

    all_jobs = jobs[0] + jobs[1]
    assert len(all_jobs) == len(set(all_jobs)) == 10

    for gpu in (0, 1):
        assert len(jobs[gpu]) == 5

        q = ROOT / (
            f"scripts/queues/mnist_stage18_fola_noclip_gpu{gpu}.txt"
        )
        q.parent.mkdir(parents=True, exist_ok=True)

        contents = "\n".join(jobs[gpu]) + "\n"

        if q.exists():
            assert q.read_text() == contents, f"Queue conflict: {q}"
        else:
            q.write_text(contents)

        print(
            f"GPU{gpu}: {len(jobs[gpu])} jobs => "
            f"{q.relative_to(ROOT)}"
        )

    print("\nPASS: 10 validated FOLA runs, five jobs on each GPU")


# ============================================================
# Read completed experiment outputs
# ============================================================

def read_run(cfg, method, rounds):
    from bayesfl.config import load_config

    output_root = ROOT / "outputs"

    directories = [
        p for p in [
            output_root / cfg.run_name,
            *output_root.glob(cfg.run_name + "_*"),
        ]
        if p.is_dir()
        and (p / "metrics/global_metrics.csv").is_file()
    ]

    directories = sorted(
        set(directories),
        key=lambda p: (
            p / "metrics/global_metrics.csv"
        ).stat().st_mtime,
        reverse=True,
    )

    for directory in directories:

        resolved = directory / "resolved_config.yaml"

        if resolved.is_file():
            actual = load_config(resolved)
            if actual.run_name != cfg.run_name or actual.method != cfg.method:
                continue

        with (
            directory / "metrics/global_metrics.csv"
        ).open(newline="", encoding="utf-8") as handle:
            rows = {
                int(float(row["round"])): row
                for row in csv.DictReader(handle)
                if row.get("round")
            }

        if not all(r in rows for r in range(1, rounds + 1)):
            continue

        keys = (
            ("fola_mean_accuracy", "accuracy",
             "global_accuracy", "test_accuracy")
            if method == "fola"
            else
            ("accuracy", "global_accuracy", "test_accuracy")
        )

        accuracy = {}
        bytes_used = {}

        for r in range(1, rounds + 1):

            row = rows[r]

            for col in keys:
                val = row.get(col)

                if val not in (None, ""):
                    v = float(val)

                    if isfinite(v):
                        accuracy[r] = (
                            100 * v if v <= 1.000001 else v
                        )
                        break

            if r not in accuracy:
                raise ValueError(
                    f"Accuracy missing: {directory.name}, R{r}"
                )

            bytes_used[r] = float(
                row["cumulative_all_array_bytes"]
            )

        assert all(
            bytes_used[r] > bytes_used[r-1]
            for r in range(2, rounds + 1)
        )

        metadata = directory / "partition_metadata.json"

        sha = (
            json.loads(metadata.read_text()).get("sha256")
            if metadata.is_file()
            else None
        )

        return {
            "acc": accuracy,
            "bytes": bytes_used,
            "sha": sha,
            "directory": directory,
        }

    raise RuntimeError(
        f"Missing or incomplete R{rounds} run: {cfg.run_name}"
    )


def avg(values, start, end):
    return mean(values[r] for r in range(start, end + 1))


def sd_ci(gaps):
    assert len(gaps) == 5
    m = mean(gaps)
    sd = stdev(gaps)
    half = T95 * sd / sqrt(5)

    return m, sd, m-half, m+half


# ============================================================
# RESULTS — Compare all five seeds
# ============================================================

def results():
    from bayesfl.config import load_config

    D = {}

    for s in SAMPLES:
        for seed in SEEDS:

            fcfg = load_config(new_path(s, seed))
            acfg = load_config(strict_path(s, seed))
            hcfg = load_config(historical_path(s, seed))

            check_pair(fcfg, acfg, s, seed, clipped=False)
            check_pair(hcfg, acfg, s, seed, clipped=True)

            f = read_run(fcfg, "fola", 200)
            a = read_run(acfg, "fedavg", 400)
            h = read_run(
                hcfg,
                "fola",
                200 if seed == 0 else 150,
            )

            for rf, ra in MILESTONES:
                assert isclose(
                    f["bytes"][rf],
                    a["bytes"][ra],
                    rel_tol=1e-9,
                    abs_tol=1.0,
                ), f"Communication mismatch: s{s}, seed{seed}"

            # Check actual saved partition SHA when available.
            if f["sha"] and a["sha"]:
                assert f["sha"] == a["sha"], (
                    f"Partition mismatch: s{s}, seed{seed}"
                )

            if f["sha"] and h["sha"]:
                assert f["sha"] == h["sha"], (
                    f"Historical partition mismatch: s{s}, seed{seed}"
                )

            D[s, seed] = {
                "f": f["acc"],
                "a": a["acc"],
                "h": h["acc"],
                "budget": f["bytes"][200] / 1e9,
            }

    print("\nPASS: All new FOLA, strict FedAvg and historical FOLA runs loaded")

    # --------------------------------------------------------
    # 1. Per-seed equal-communication endpoints
    # --------------------------------------------------------

    print("\n" + "=" * 110)
    print("1. PER-SEED EQUAL-COMMUNICATION ENDPOINTS")
    print("FOLA = unclipped | FedAvg = strict Paper-00")
    print("=" * 110)

    print(
        " S Seed | F120/A240 gap | "
        "F150/A300 gap | F200/A400 gap"
    )

    for s in SAMPLES:
        for seed in SEEDS:

            f = D[s, seed]["f"]
            a = D[s, seed]["a"]

            cells = [
                f"{f[rf]:5.2f}/{a[ra]:5.2f} "
                f"({f[rf]-a[ra]:+6.2f})"
                for rf, ra in MILESTONES
            ]

            print(
                f"{s:2d} {seed:4d} | "
                + " | ".join(cells)
            )

    # --------------------------------------------------------
    # 2. Five-seed endpoints and confidence intervals
    # --------------------------------------------------------

    print("\n" + "=" * 120)
    print("2. FIVE-SEED EQUAL-COMMUNICATION ENDPOINT SUMMARY")
    print("=" * 120)

    print(
        " S   Rounds   FOLA mean±SD    FedAvg mean±SD "
        "   Gap mean±SD       95% CI         Wins"
    )

    for s in SAMPLES:
        for rf, ra in MILESTONES:

            fv = [D[s, i]["f"][rf] for i in SEEDS]
            av = [D[s, i]["a"][ra] for i in SEEDS]
            gaps = [f-a for f, a in zip(fv, av)]

            m, sd, lo, hi = sd_ci(gaps)

            print(
                f"{s:2d}  {rf:3d}/{ra:<3d}  "
                f"{mean(fv):6.2f}±{stdev(fv):5.2f}  "
                f"{mean(av):6.2f}±{stdev(av):5.2f}  "
                f"{m:+6.2f}±{sd:5.2f} "
                f"[{lo:+6.2f},{hi:+6.2f}] "
                f"{sum(g > 0 for g in gaps)}/5"
            )

    # --------------------------------------------------------
    # 3. Same-round comparison
    # --------------------------------------------------------

    print("\n" + "=" * 100)
    print("3. FIVE-SEED SAME-ROUND COMPARISON")
    print("Same rounds are NOT the same communication cost")
    print("=" * 100)

    print(
        " S  Round    FOLA mean   FedAvg mean  "
        "Paired gap      95% CI"
    )

    for s in SAMPLES:
        for r in (120, 150, 200):

            fv = [D[s, i]["f"][r] for i in SEEDS]
            av = [D[s, i]["a"][r] for i in SEEDS]

            m, sd, lo, hi = sd_ci(
                [f-a for f, a in zip(fv, av)]
            )

            print(
                f"{s:2d}  R{r:<3d} "
                f"{mean(fv):8.2f}% "
                f"{mean(av):8.2f}% "
                f"{m:+7.2f} pp "
                f"[{lo:+6.2f},{hi:+6.2f}]"
            )

    # --------------------------------------------------------
    # 4. Communication-matched tails and convergence
    # --------------------------------------------------------

    print("\n" + "=" * 135)
    print("4. COMMUNICATION-MATCHED FINAL-TAIL SUMMARY")
    print("FOLA last 20 rounds vs FedAvg last 40 rounds")
    print("Delta = current window mean minus previous equal-width window")
    print("=" * 135)

    print(
        " S   Rounds    F-tail  A-tail "
        " Gap mean±SD       95% CI       "
        "F-delta A-delta F-winSD A-winSD"
    )

    for s in SAMPLES:
        for rf, ra in MILESTONES:

            rows = []

            for seed in SEEDS:

                f = D[s, seed]["f"]
                a = D[s, seed]["a"]

                ft = avg(f, rf-19, rf)
                at = avg(a, ra-39, ra)

                rows.append((
                    ft,
                    at,
                    ft-at,
                    ft-avg(f, rf-39, rf-20),
                    at-avg(a, ra-79, ra-40),
                    pstdev(
                        f[r] for r in range(rf-19, rf+1)
                    ),
                    pstdev(
                        a[r] for r in range(ra-39, ra+1)
                    ),
                ))

            m, sd, lo, hi = sd_ci(
                [r[2] for r in rows]
            )

            print(
                f"{s:2d} {rf:3d}/{ra:<3d} "
                f"{mean(r[0] for r in rows):7.2f} "
                f"{mean(r[1] for r in rows):7.2f} "
                f"{m:+6.2f}±{sd:5.2f} "
                f"[{lo:+6.2f},{hi:+6.2f}] "
                f"{mean(r[3] for r in rows):+7.2f} "
                f"{mean(r[4] for r in rows):+7.2f} "
                f"{mean(r[5] for r in rows):7.3f} "
                f"{mean(r[6] for r in rows):7.3f}"
            )

    # --------------------------------------------------------
    # 5. Ablation: remove clipping from FOLA
    # --------------------------------------------------------

    print("\n" + "=" * 110)
    print("5. FOLA GRADIENT-CLIPPING ABLATION")
    print("New unclipped FOLA minus historical clipped FOLA")
    print("=" * 110)

    print(
        " S  Round    Difference mean±SD       "
        "95% CI       Wins   Per-seed gaps"
    )

    for s in SAMPLES:
        for rf in (120, 150):

            gaps = [
                D[s, i]["f"][rf] - D[s, i]["h"][rf]
                for i in SEEDS
            ]

            m, sd, lo, hi = sd_ci(gaps)

            print(
                f"{s:2d} R{rf:<3d} "
                f"{m:+6.2f}±{sd:5.2f} "
                f"[{lo:+6.2f},{hi:+6.2f}] "
                f"{sum(g > 0 for g in gaps)}/5 "
                + ", ".join(f"{g:+.2f}" for g in gaps)
            )

        # Historical FOLA seeds 1-4 only reach R150.
        seed0 = D[s, 0]

        print(
            f"s{s} R200 seed-0 only | "
            f"unclipped={seed0['f'][200]:.2f}% | "
            f"clipped={seed0['h'][200]:.2f}% | "
            f"difference={seed0['f'][200]-seed0['h'][200]:+.2f} pp"
        )

    print("\n" + "=" * 90)
    print("PASS: STAGE-18 FIVE-SEED ANALYSIS COMPLETE")
    print("=" * 90)

    print("FOLA: paper_reference, NO clipping, R200")
    print("FedAvg: strict Paper-00, NO clipping, R400")
    print("Sample counts: 5 and 10")
    print("Seeds: 0-4")
    print("Matched milestones: 120/240, 150/300, 200/400")
    print("95% CI: paired five-seed Student-t interval")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("prepare", "results"),
    )

    action = parser.parse_args().action

    if action == "prepare":
        prepare()
    else:
        results()
