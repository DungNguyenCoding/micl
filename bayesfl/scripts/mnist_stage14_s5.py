from pathlib import Path
from collections import Counter
from statistics import mean, pstdev
import argparse
import copy
import csv
import math
import yaml

ROOT = Path(__file__).resolve().parents[1]

STAGE12 = ROOT / (
    "scripts/configs/mnist_stage12_arch_fixed_s10_n20_seed0"
)

OUT = ROOT / (
    "scripts/configs/mnist_stage14_arch_fixed_s5_n20_seed0"
)

QUEUES = [
    ROOT / (
        f"scripts/queues/"
        f"mnist_stage14_arch_fixed_s5_n20_seed0_gpu{i}.txt"
    )
    for i in (0, 1)
]

MODELS = {
    "mlp200x2": ("mlp_784_200_200_10", 199210),
    "cnn16_32": ("mnist_cnn_16_32_128", 80202),
    "mlp256x3": ("mlp_784_256x3_10", 335114),
}

LRS = (0.01, 0.03, 0.10)

MILESTONES = (
    (120, 240),
    (150, 300),
    (200, 400),
)


def run_name(stage, model, lr, method, samples, batch, rounds):
    lr_tag = f"{lr:.2f}".replace(".", "p")
    lam_tag = "_lam0p01" if method == "fola" else ""

    return (
        f"debug_mnist_stage{stage}_{model}_"
        f"fixed1c_s{samples}_n20_{method}_"
        f"seed0_e5_b{batch}_lr{lr_tag}"
        f"{lam_tag}_r{rounds}"
    )


def queue_configs():
    from bayesfl.config import load_config

    configs = []

    for queue in QUEUES:
        assert queue.is_file(), queue

        for line in queue.read_text().splitlines():
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            path = ROOT / line
            assert path.is_file(), path

            configs.append(load_config(path))

    assert len(configs) == 18
    assert len({c.run_name for c in configs}) == 18

    return configs


def prepare():
    from bayesfl.config import load_config
    from bayesfl.communication import core_round_bytes
    from bayesfl.models.factory import build_model, initialize_model
    from bayesfl.data.datasets import partition_stem, prepare_partition
    from bayesfl.data.partition import load_partition
    from torchvision.datasets import MNIST
    import torch

    OUT.mkdir(parents=True, exist_ok=True)

    for queue in QUEUES:
        queue.parent.mkdir(parents=True, exist_ok=True)

    paths = []
    configs = {}

    # ========================================================
    # Generate 18 configurations from Stage-12 templates
    # ========================================================

    for short, (model_name, expected_params) in MODELS.items():
        for lr in LRS:
            for method in ("fola", "fedavg"):

                rounds = 200 if method == "fola" else 400

                old_name = run_name(
                    12, short, lr, method, 10, 10, rounds
                )

                template = STAGE12 / f"{old_name}.yaml"

                assert template.is_file(), (
                    f"Missing Stage-12 template: {template}"
                )

                cfg = yaml.safe_load(template.read_text())

                # Validate original Stage-12 settings.
                assert cfg["model"]["name"] == model_name
                assert cfg["method"] == method
                assert cfg["runtime"]["seed"] == 0

                assert cfg["data"]["partition"] == {
                    "type": "fixed_labels",
                    "samples_per_client": 10,
                    "labels_per_client": 1,
                }

                assert cfg["federation"]["num_clients"] == 20
                assert cfg["federation"]["clients_per_round"] == 20

                assert cfg["training"]["rounds"] == rounds
                assert cfg["training"]["lr"] == lr
                assert cfg["training"]["batch_size"] == 10
                assert cfg["training"]["local_epochs"] == 5
                assert cfg["training"]["lr_schedule"] == "constant"

                assert cfg["fola"]["mode"] == "paper_reference"
                assert cfg["fola"]["prior_lambda"] == 0.01

                assert cfg["compression"]["selection_rule"] == "dense"
                assert cfg["compression"]["keep_ratio"] == 1.0

                assert cfg["communication"]["downlink_mode"] == "multicast"

                # Only change local sample count and batch size.
                cfg = copy.deepcopy(cfg)

                cfg["run_name"] = run_name(
                    14, short, lr, method, 5, 5, rounds
                )

                cfg["data"]["partition"]["samples_per_client"] = 5
                cfg["training"]["batch_size"] = 5

                path = OUT / f"{cfg['run_name']}.yaml"

                if path.exists():
                    assert yaml.safe_load(path.read_text()) == cfg, (
                        f"Existing config differs: {path}"
                    )
                else:
                    assert not list(
                        (ROOT / "outputs").glob(
                            cfg["run_name"] + "_*"
                        )
                    ), f"Existing experiment output: {cfg['run_name']}"

                    path.write_text(
                        yaml.safe_dump(cfg, sort_keys=False)
                    )

                loaded = load_config(path)

                model = build_model(loaded)

                count = sum(
                    p.numel() for p in model.parameters()
                )

                assert count == expected_params, (
                    short, count, expected_params
                )

                assert loaded.training.batch_size == 5
                assert loaded.data.partition["samples_per_client"] == 5

                configs[(short, lr, method)] = loaded
                paths.append(path.relative_to(ROOT).as_posix())

    # ========================================================
    # Validate paired initialization and communication costs
    # ========================================================

    partition_ids = set()

    for short, (_, nparams) in MODELS.items():
        for lr in LRS:

            f = configs[(short, lr, "fola")]
            a = configs[(short, lr, "fedavg")]

            pf = partition_stem(f)
            pa = partition_stem(a)

            assert pf == pa
            partition_ids.add(pf)

            mf = initialize_model(f)
            ma = initialize_model(a)

            assert all(
                torch.equal(x, y)
                for x, y in zip(
                    mf.parameters(), ma.parameters()
                )
            ), f"Initialization mismatch: {short}, LR={lr}"

            for rf, ra in MILESTONES:

                fb = rf * core_round_bytes(
                    nparams,
                    20,
                    method="fola",
                    downlink_mode="multicast",
                )["total_bytes"]

                ab = ra * core_round_bytes(
                    nparams,
                    20,
                    method="fedavg",
                    downlink_mode="multicast",
                )["total_bytes"]

                assert fb == ab

            print(
                f"PAIR PASS | {short:9s} LR={lr:.2f} | "
                f"R200/R400 budget={fb/1e9:.3f} GB"
            )

    assert len(partition_ids) == 1

    # ========================================================
    # Create and validate the common 5-sample partition
    # ========================================================

    pcfg = configs[("mlp200x2", 0.01, "fola")]

    part_path, metadata = prepare_partition(pcfg)
    parts = load_partition(part_path)

    labels = MNIST(
        root=pcfg.data.root,
        train=True,
        download=True,
    ).targets.numpy()

    assert len(parts) == 20
    assert all(len(p) == 5 for p in parts)

    assigned = [int(labels[p[0]]) for p in parts]
    counts = Counter(assigned)

    assert all(
        len(set(labels[p].tolist())) == 1
        for p in parts
    )

    assert all(
        counts[c] == 2 for c in range(10)
    )

    indices = [
        int(i) for p in parts for i in p
    ]

    assert len(indices) == len(set(indices)) == 100

    print(
        "\nPARTITION PASS | 20 clients | "
        "5 samples/client | 100 unique samples | "
        "1 class/client"
    )

    print("SHA256:", metadata.get("sha256"))

    # ========================================================
    # Split into two GPU queues
    # ========================================================

    assert len(paths) == len(set(paths)) == 18

    splits = [
        [
            job for i, job in enumerate(paths)
            if i % 4 in (0, 3)
        ],
        [
            job for i, job in enumerate(paths)
            if i % 4 in (1, 2)
        ],
    ]

    for gpu, (queue, jobs) in enumerate(
        zip(QUEUES, splits)
    ):
        assert len(jobs) == 9

        content = "\n".join(jobs) + "\n"

        if queue.exists():
            assert queue.read_text() == content, (
                f"Existing queue differs: {queue}"
            )
        else:
            queue.write_text(content)

        print(
            f"GPU {gpu}: {len(jobs)} jobs | "
            f"{queue.relative_to(ROOT)}"
        )

    assert len(queue_configs()) == 18

    print("\nPASS: 18 jobs, 9 matched pairs, both GPU queues ready")


# ============================================================
# Result extraction
# ============================================================

def load_result(cfg):

    target = cfg.training.rounds

    candidates = sorted(
        [
            p for p in (ROOT / "outputs").glob(
                cfg.run_name + "_*"
            )
            if p.is_dir()
            and (p / "metrics/global_metrics.csv").is_file()
        ],
        key=lambda p: (
            p / "metrics/global_metrics.csv"
        ).stat().st_mtime,
        reverse=True,
    )

    for directory in candidates:

        with (
            directory / "metrics/global_metrics.csv"
        ).open(newline="", encoding="utf-8") as f:

            rows = {
                int(float(row["round"])): row
                for row in csv.DictReader(f)
                if row.get("round")
            }

        if not all(
            r in rows for r in range(1, target + 1)
        ):
            continue

        acc = {}
        comm = {}

        for r in range(1, target + 1):

            row = rows[r]

            keys = (
                (
                    "fola_mean_accuracy",
                    "accuracy",
                    "global_accuracy",
                    "test_accuracy",
                )
                if cfg.method == "fola"
                else (
                    "accuracy",
                    "global_accuracy",
                    "test_accuracy",
                )
            )

            for key in keys:
                if row.get(key) not in (None, ""):

                    value = float(row[key])

                    if math.isfinite(value):
                        acc[r] = (
                            value * 100
                            if value <= 1.000001
                            else value
                        )
                        break

            if r not in acc:
                raise ValueError(
                    f"Missing accuracy: {directory}, R{r}"
                )

            comm[r] = float(
                row["cumulative_all_array_bytes"]
            )

        assert all(
            comm[r] > comm[r-1]
            for r in range(2, target + 1)
        )

        return acc, comm

    raise RuntimeError(
        f"Incomplete run: {cfg.run_name}, expected R{target}"
    )


def results():

    records = {}

    for cfg in queue_configs():

        short = next(
            m for m, (name, _) in MODELS.items()
            if name == cfg.model.name
        )

        key = (
            short,
            round(float(cfg.training.lr), 2),
            cfg.method,
        )

        assert key not in records

        records[key] = load_result(cfg)

    assert len(records) == 18

    def avg(x, a, b):
        return mean(x[r] for r in range(a, b + 1))

    def sd(x, a, b):
        return pstdev(x[r] for r in range(a, b + 1))

    # ========================================================
    # 1. Endpoint comparisons
    # ========================================================

    print("\n" + "=" * 115)
    print("1. EQUAL-COMMUNICATION ENDPOINT COMPARISON")
    print("Each entry: FOLA% / FedAvg% (gap pp)")
    print("=" * 115)

    for short in MODELS:
        for lr in LRS:

            f, fc = records[(short, lr, "fola")]
            a, ac = records[(short, lr, "fedavg")]

            print(f"\nMODEL={short}, LR={lr:.2f}")

            for rf, ra in MILESTONES:

                assert math.isclose(
                    fc[rf], ac[ra],
                    rel_tol=1e-9,
                    abs_tol=1.0,
                )

                print(
                    f" FOLA R{rf:3d} vs FedAvg R{ra:3d} | "
                    f"{f[rf]:6.2f}% / {a[ra]:6.2f}% | "
                    f"gap={f[rf]-a[ra]:+7.2f} pp | "
                    f"budget={fc[rf]/1e9:.3f} GB"
                )

    # ========================================================
    # 2. Tail accuracy and convergence
    # ========================================================

    print("\n" + "=" * 130)
    print("2. COMMUNICATION-MATCHED TAIL AND CONVERGENCE")
    print("Tail: FOLA last20 rounds vs FedAvg last40 rounds")
    print("Delta: current tail mean minus preceding equal-width window")
    print("=" * 130)

    print(
        "Milestone Model       LR  BudgetGB "
        " F-tail  A-tail   Gap   F-std A-std "
        " F-delta A-delta"
    )

    rankings = {}

    for rf, ra in MILESTONES:

        rankings[rf] = []

        for short in MODELS:
            for lr in LRS:

                f, fc = records[(short, lr, "fola")]
                a, ac = records[(short, lr, "fedavg")]

                ft = avg(f, rf-19, rf)
                at = avg(a, ra-39, ra)

                fs = sd(f, rf-19, rf)
                ast = sd(a, ra-39, ra)

                fd = ft - avg(f, rf-39, rf-20)
                ad = at - avg(a, ra-79, ra-40)

                gap = ft - at

                rankings[rf].append(
                    (gap, short, lr, ft, at, fs, ast)
                )

                print(
                    f"{rf:3d}/{ra:<3d} "
                    f"{short:11s} {lr:4.2f} "
                    f"{fc[rf]/1e9:8.3f} "
                    f"{ft:7.2f} {at:7.2f} "
                    f"{gap:+6.2f} "
                    f"{fs:6.2f} {ast:5.2f} "
                    f"{fd:+7.2f} {ad:+7.2f}"
                )

        print("-" * 105)

    # ========================================================
    # 3. Ranking at each budget
    # ========================================================

    print("\n" + "=" * 100)
    print("3. RANKING BY EQUAL-COMMUNICATION TAIL GAP")
    print("=" * 100)

    for rf, ra in MILESTONES:

        print(f"\nFOLA R{rf} vs FedAvg R{ra}")

        for rank, record in enumerate(
            sorted(rankings[rf], reverse=True),
            1,
        ):
            gap, short, lr, ft, at, fs, ast = record

            print(
                f"{rank:2d}. {short:11s} "
                f"LR={lr:.2f} | "
                f"gap={gap:+6.2f} pp | "
                f"FOLA={ft:.2f}% | "
                f"FedAvg={at:.2f}% | "
                f"std={fs:.2f}/{ast:.2f}"
            )

    print(
        "\nPASS: all 18 experiments analyzed "
        "at R120/240, R150/300 and R200/400"
    )


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
