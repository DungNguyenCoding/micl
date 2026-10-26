from pathlib import Path
from statistics import mean, stdev, pstdev
from math import sqrt, isclose, isfinite
import csv
import json

from bayesfl.config import load_config
from bayesfl.data.datasets import partition_stem

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

SAMPLES = (5, 10)
SEEDS = range(5)
MILESTONES = ((120, 240), (150, 300))

T95 = 2.7764451051977987  # Student t, df=4


# ============================================================
# 1. Locate the exact experiment configurations
# ============================================================

def config_path(samples, seed, method):

    if method == "strict":
        name = (
            "run_stage16_paper00_fedavg_cnn16_32_"
            f"s{samples}_n20_seed{seed}_e5_b{samples}_"
            "lr0p10_r400"
        )

        directory = (
            "mnist_stage16_paper00_fedavg"
        )

    elif seed == 0:

        stage = 12 if samples == 10 else 14
        rounds = 200 if method == "fola" else 400
        lam = "_lam0p01" if method == "fola" else ""

        name = (
            f"debug_mnist_stage{stage}_cnn16_32_"
            f"fixed1c_s{samples}_n20_{method}_"
            f"seed0_e5_b{samples}_lr0p10"
            f"{lam}_r{rounds}"
        )

        directory = (
            "mnist_stage12_arch_fixed_s10_n20_seed0"
            if samples == 10
            else "mnist_stage14_arch_fixed_s5_n20_seed0"
        )

    else:

        rounds = 150 if method == "fola" else 300
        lam = "_lam0p01" if method == "fola" else ""

        name = (
            "run_mnist_stage15_cnn16_32_"
            f"fixed1c_s{samples}_n20_{method}_"
            f"seed{seed}_e5_b{samples}_lr0p10"
            f"{lam}_r{rounds}"
        )

        directory = "mnist_stage15_cnn_multiseed"

    return (
        ROOT / "scripts/configs" / directory / f"{name}.yaml"
    )


# ============================================================
# 2. Read completed trajectories
# ============================================================

def accuracy(row, method):

    keys = (
        (
            "fola_mean_accuracy",
            "accuracy",
            "global_accuracy",
            "test_accuracy",
        )
        if method == "fola"
        else (
            "accuracy",
            "global_accuracy",
            "test_accuracy",
        )
    )

    for key in keys:
        value = row.get(key)

        if value in (None, ""):
            continue

        try:
            x = float(value)
        except (TypeError, ValueError):
            continue

        if isfinite(x):
            return x * 100 if x <= 1.000001 else x

    raise ValueError(
        f"Missing accuracy column: {list(row)}"
    )


def load_run(cfg, method, target):

    name = cfg.run_name

    candidates = [
        OUTPUTS / name,
        *OUTPUTS.glob(name + "_*"),
    ]

    candidates = sorted(
        {
            p for p in candidates
            if p.is_dir()
            and (p / "metrics/global_metrics.csv").is_file()
        },
        key=lambda p: (
            p / "metrics/global_metrics.csv"
        ).stat().st_mtime,
        reverse=True,
    )

    last_seen = 0

    for directory in candidates:

        metrics = directory / "metrics/global_metrics.csv"

        with metrics.open(
            newline="", encoding="utf-8"
        ) as file:
            rows = {
                int(float(row["round"])): row
                for row in csv.DictReader(file)
                if row.get("round")
            }

        if rows:
            last_seen = max(last_seen, max(rows))

        if not all(
            r in rows for r in range(1, target + 1)
        ):
            continue

        acc = {
            r: accuracy(rows[r], method)
            for r in range(1, target + 1)
        }

        comm = {
            r: float(
                rows[r]["cumulative_all_array_bytes"]
            )
            for r in range(1, target + 1)
        }

        assert all(
            comm[r] > comm[r - 1]
            for r in range(2, target + 1)
        )

        return {
            "cfg": cfg,
            "acc": acc,
            "comm": comm,
            "directory": directory,
        }

    raise RuntimeError(
        f"Missing/incomplete: {name}\n"
        f"Required R{target}; latest available R{last_seen}"
    )


def partition_hash(run):

    path = run["directory"] / "partition_metadata.json"

    if not path.is_file():
        return None

    return json.loads(
        path.read_text()
    ).get("sha256")


def average(acc, lo, hi):
    return mean(acc[r] for r in range(lo, hi + 1))


def window_std(acc, lo, hi):
    return pstdev(acc[r] for r in range(lo, hi + 1))


def mean_sd(values):
    return mean(values), stdev(values)


def ci95(values):

    values = list(values)

    assert len(values) == 5

    m = mean(values)
    s = stdev(values)
    half = T95 * s / sqrt(5)

    return m, m - half, m + half


# ============================================================
# 3. Load all five seeds
# ============================================================

RUNS = {}
CLIPPED = {}
errors = []

for samples in SAMPLES:

    for seed in SEEDS:

        case = {}

        for method in ("fola", "strict"):

            path = config_path(samples, seed, method)

            if not path.is_file():
                errors.append(f"Missing config: {path}")
                continue

            cfg = load_config(path)

            target = (
                200 if method == "fola" and seed == 0
                else 150 if method == "fola"
                else 400
            )

            if method == "strict":

                assert cfg.method == "fedavg"
                assert cfg.training.paper00_strict is True
                assert cfg.training.grad_clip_norm is None

                read_method = "fedavg"

            else:

                assert cfg.method == "fola"
                assert cfg.fola.mode == "paper_reference"
                assert cfg.training.grad_clip_norm == 10.0

                read_method = "fola"

            assert cfg.runtime.seed == seed
            assert cfg.model.name == "mnist_cnn_16_32_128"
            assert cfg.training.lr == 0.10
            assert cfg.training.local_epochs == 5
            assert cfg.training.batch_size == samples
            assert cfg.federation.num_clients == 20
            assert cfg.federation.clients_per_round == 20

            try:
                case[method] = load_run(
                    cfg, read_method, target
                )
            except Exception as exc:
                errors.append(str(exc))

        if len(case) == 2:

            f = case["fola"]
            a = case["strict"]

            # Paired experiments must use identical partition.
            assert (
                partition_stem(f["cfg"])
                == partition_stem(a["cfg"])
            ), (samples, seed, "partition mismatch")

            fh = partition_hash(f)
            ah = partition_hash(a)

            if fh and ah:
                assert fh == ah, (
                    samples, seed, "partition SHA mismatch"
                )

            # Check recorded equal-communication budgets.
            for rf, ra in MILESTONES:

                assert isclose(
                    f["comm"][rf],
                    a["comm"][ra],
                    rel_tol=1e-9,
                    abs_tol=1.0,
                ), (
                    samples, seed,
                    f"FOLA R{rf} / FedAvg R{ra} budget mismatch"
                )

            RUNS[(samples, seed)] = case

        # Historical clipped FedAvg, for clipping ablation.
        old_path = config_path(
            samples, seed, "fedavg"
        )

        if old_path.is_file():

            old_cfg = load_config(old_path)

            try:
                old = load_run(
                    old_cfg, "fedavg", 300
                )

                if (samples, seed) in RUNS:

                    strict = RUNS[(samples, seed)]["strict"]

                    assert (
                        partition_stem(old_cfg)
                        == partition_stem(strict["cfg"])
                    )

                    for r in (240, 300):
                        assert isclose(
                            old["comm"][r],
                            strict["comm"][r],
                            rel_tol=1e-9,
                            abs_tol=1.0,
                        )

                CLIPPED[(samples, seed)] = old

            except Exception as exc:
                print(
                    "WARNING: clipped FedAvg unavailable:",
                    samples, seed, exc
                )


if errors:

    print("\nINCOMPLETE EXPERIMENTS:")

    for e in errors:
        print(" -", e)

    raise SystemExit(
        f"\nOnly {len(RUNS)}/10 paired runs available."
    )

assert len(RUNS) == 10

print()
print("=" * 105)
print("STAGE-16 STRICT FEDAVG VS HISTORICAL FOLA")
print("CNN 16-32 | LR=0.10 | N=20 | E=5 | one class/client")
print("=" * 105)

print("PASS: 10 paired trajectories loaded")
print("Conditions: 5 and 10 samples/client")
print("Seeds: 0, 1, 2, 3, 4")
print("FOLA: historical clipped, norm=10")
print("FedAvg: strict Paper-00, no clipping")


# ============================================================
# 4. Per-seed equal-communication comparisons
# ============================================================

print()
print("=" * 105)
print("1. PER-SEED EQUAL-COMMUNICATION RESULTS")
print("Gap = historical FOLA - strict FedAvg")
print("=" * 105)

print(
    "Samples Seed | F120  A240  Gap120 | "
    "F150  A300  Gap150 | "
    "A400"
)
print("-" * 90)

for samples in SAMPLES:

    for seed in SEEDS:

        f = RUNS[(samples, seed)]["fola"]["acc"]
        a = RUNS[(samples, seed)]["strict"]["acc"]

        print(
            f"{samples:7d} {seed:4d} | "
            f"{f[120]:5.2f} "
            f"{a[240]:5.2f} "
            f"{f[120]-a[240]:+7.2f} | "
            f"{f[150]:5.2f} "
            f"{a[300]:5.2f} "
            f"{f[150]-a[300]:+7.2f} | "
            f"{a[400]:5.2f}"
        )

    print("-" * 90)


# ============================================================
# 5. Five-seed accuracy summary with 95% CI
# ============================================================

print()
print("=" * 120)
print("2. FIVE-SEED ENDPOINT ACCURACY SUMMARY")
print("95% CI: Student-t interval for paired seed-wise gaps")
print("=" * 120)

print(
    "Samples Rounds    "
    "FOLA mean+/-SD   FedAvg mean+/-SD   "
    "Gap mean+/-SD     95% CI of gap        Wins"
)
print("-" * 120)

for samples in SAMPLES:

    for rf, ra in MILESTONES:

        fvalues = []
        avalues = []
        gaps = []

        for seed in SEEDS:

            f = RUNS[(samples, seed)]["fola"]["acc"]
            a = RUNS[(samples, seed)]["strict"]["acc"]

            fvalues.append(f[rf])
            avalues.append(a[ra])
            gaps.append(f[rf] - a[ra])

        fm, fs = mean_sd(fvalues)
        am, ads = mean_sd(avalues)
        gm, gs = mean_sd(gaps)

        _, low, high = ci95(gaps)

        wins = sum(g > 0 for g in gaps)

        print(
            f"{samples:7d} R{rf:<3d}/{ra:<3d} "
            f"{fm:7.2f}+/-{fs:5.2f} "
            f"{am:9.2f}+/-{ads:5.2f} "
            f"{gm:+8.2f}+/-{gs:5.2f} "
            f"[{low:+7.2f}, {high:+7.2f}] "
            f"{wins}/5"
        )


# ============================================================
# 6. Final-tail accuracy and late convergence
# ============================================================

print()
print("=" * 125)
print("3. FIVE-SEED COMMUNICATION-MATCHED FINAL-TAIL SUMMARY")
print("FOLA last 20 rounds vs FedAvg last 40 rounds")
print("=" * 125)

print(
    "Samples Rounds   F-tail   A-tail   "
    "TailGap   95% CI of tail gap      "
    "F-delta A-delta  F-std A-std"
)
print("-" * 125)

SUMMARY = []

for samples in SAMPLES:

    for rf, ra in MILESTONES:

        f_tail = []
        a_tail = []
        gaps = []

        f_delta = []
        a_delta = []

        f_stds = []
        a_stds = []

        for seed in SEEDS:

            f = RUNS[(samples, seed)]["fola"]["acc"]
            a = RUNS[(samples, seed)]["strict"]["acc"]

            ft = average(f, rf-19, rf)
            at = average(a, ra-39, ra)

            fp = average(f, rf-39, rf-20)
            ap = average(a, ra-79, ra-40)

            f_tail.append(ft)
            a_tail.append(at)
            gaps.append(ft - at)

            f_delta.append(ft - fp)
            a_delta.append(at - ap)

            f_stds.append(
                window_std(f, rf-19, rf)
            )

            a_stds.append(
                window_std(a, ra-39, ra)
            )

        _, low, high = ci95(gaps)

        print(
            f"{samples:7d} R{rf:<3d}/{ra:<3d} "
            f"{mean(f_tail):7.2f} "
            f"{mean(a_tail):8.2f} "
            f"{mean(gaps):+8.2f} "
            f"[{low:+7.2f}, {high:+7.2f}] "
            f"{mean(f_delta):+8.2f} "
            f"{mean(a_delta):+8.2f} "
            f"{mean(f_stds):5.3f} "
            f"{mean(a_stds):5.3f}"
        )

        SUMMARY.append({
            "samples": samples,
            "rf": rf,
            "ra": ra,
            "fola": mean(f_tail),
            "fedavg": mean(a_tail),
            "gap": mean(gaps),
            "ci": (low, high),
            "wins": sum(g > 0 for g in gaps),
        })


# ============================================================
# 7. Historical clipped vs strict unclipped FedAvg
# ============================================================

print()
print("=" * 110)
print("4. EFFECT OF REMOVING GRADIENT CLIPPING FROM FEDAVG")
print("Difference = strict/unclipped - historical/clipped")
print("Positive means unclipped FedAvg performs better")
print("=" * 110)

print(
    "Samples Seed | "
    "Strict R240 - Clipped R240 | "
    "Strict R300 - Clipped R300"
)
print("-" * 85)

for samples in SAMPLES:

    diffs240 = []
    diffs300 = []

    for seed in SEEDS:

        old = CLIPPED.get((samples, seed))

        if old is None:
            print(
                f"{samples:7d} {seed:4d} | N/A"
            )
            continue

        new = RUNS[(samples, seed)]["strict"]["acc"]
        old = old["acc"]

        d240 = new[240] - old[240]
        d300 = new[300] - old[300]

        diffs240.append(d240)
        diffs300.append(d300)

        print(
            f"{samples:7d} {seed:4d} | "
            f"{d240:+12.2f} pp            "
            f"{d300:+12.2f} pp"
        )

    if diffs240:
        print(
            f"s{samples} MEAN CHANGE | "
            f"R240={mean(diffs240):+.2f} pp | "
            f"R300={mean(diffs300):+.2f} pp | "
            f"n={len(diffs240)}"
        )

    print("-" * 85)


# ============================================================
# 8. Seed-0-only R200/R400 comparison
# ============================================================

print()
print("=" * 100)
print("5. SEED-0-ONLY R200/R400 COMPARISON")
print("Other FOLA seeds end at R150, so this is NOT five-seed.")
print("=" * 100)

for samples in SAMPLES:

    f = RUNS[(samples, 0)]["fola"]["acc"]
    a = RUNS[(samples, 0)]["strict"]["acc"]

    if 200 not in f:
        print(f"s{samples}: No FOLA R200 available")
        continue

    fcomm = RUNS[(samples, 0)]["fola"]["comm"][200]
    acomm = RUNS[(samples, 0)]["strict"]["comm"][400]

    assert isclose(
        fcomm, acomm,
        rel_tol=1e-9, abs_tol=1.0
    )

    print(
        f"s{samples}: "
        f"FOLA R200={f[200]:.2f}% | "
        f"strict FedAvg R400={a[400]:.2f}% | "
        f"gap={f[200]-a[400]:+.2f} pp | "
        f"budget={fcomm/1e9:.3f} GB"
    )


# ============================================================
# 9. Ranking and completion
# ============================================================

print()
print("=" * 110)
print("6. RANKING BY FIVE-SEED EQUAL-COMMUNICATION TAIL GAP")
print("=" * 110)

for rank, row in enumerate(
    sorted(
        SUMMARY,
        key=lambda x: x["gap"],
        reverse=True
    ),
    1,
):
    low, high = row["ci"]

    print(
        f"{rank}. s{row['samples']} "
        f"FOLA R{row['rf']} / FedAvg R{row['ra']} | "
        f"FOLA={row['fola']:.2f}% | "
        f"FedAvg={row['fedavg']:.2f}% | "
        f"gap={row['gap']:+.2f} pp | "
        f"95% CI=[{low:+.2f}, {high:+.2f}] | "
        f"wins={row['wins']}/5"
    )

print()
print("=" * 85)
print("PASS: STAGE-16 CNN FIVE-SEED COMPARISON COMPLETE")
print("=" * 85)

print("Compared:")
print("  CNN, 5 samples/client, LR=0.10, seeds 0-4")
print("  CNN, 10 samples/client, LR=0.10, seeds 0-4")
print("  Historical clipped FOLA vs strict unclipped FedAvg")
print("  FOLA R120 vs FedAvg R240")
print("  FOLA R150 vs FedAvg R300")
print("  Seed-0-only FOLA R200 vs FedAvg R400")

print(
    "\nPaste sections 2, 3, 4, and 6 into ChatGPT."
)
