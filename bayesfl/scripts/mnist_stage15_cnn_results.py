from pathlib import Path
from statistics import mean, stdev, pstdev
from math import sqrt, isclose, isfinite
import csv

ROOT = Path(__file__).resolve().parents[1]

# samples/client, learning rate, seed-0 source stage
GROUPS = [
    (10, 0.10, 12),
    (5,  0.10, 14),
    (5,  0.03, 14),
]

SEEDS = range(5)

MILESTONES = [
    (120, 240),
    (150, 300),
]

# Two-sided 95% Student-t interval, df=4 (five seeds)
T_CRITICAL = 2.7764451051977987


# ============================================================
# Helpers
# ============================================================

def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_name(samples, lr, stage, seed, method):

    lr_tag = f"{lr:.2f}".replace(".", "p")
    lam = "_lam0p01" if method == "fola" else ""

    if seed == 0:
        # Existing Stage-12 or Stage-14 run.
        rounds = 200 if method == "fola" else 400

        return (
            f"debug_mnist_stage{stage}_cnn16_32_"
            f"fixed1c_s{samples}_n20_{method}_"
            f"seed0_e5_b{samples}_lr{lr_tag}"
            f"{lam}_r{rounds}"
        )

    # New Stage-15 run, seeds 1-4.
    rounds = 150 if method == "fola" else 300

    return (
        f"run_mnist_stage15_cnn16_32_"
        f"fixed1c_s{samples}_n20_{method}_"
        f"seed{seed}_e5_b{samples}_lr{lr_tag}"
        f"{lam}_r{rounds}"
    )


def get_accuracy(row, method):

    if method == "fola":
        keys = (
            "fola_mean_accuracy",
            "accuracy",
            "global_accuracy",
            "test_accuracy",
        )
    else:
        keys = (
            "accuracy",
            "global_accuracy",
            "test_accuracy",
        )

    for key in keys:

        value = row.get(key)

        if value in (None, ""):
            continue

        try:
            value = float(value)
        except (TypeError, ValueError):
            continue

        if isfinite(value):
            return value * 100 if value <= 1.000001 else value

    raise ValueError(
        f"No usable accuracy column for {method}: {list(row)}"
    )


def load_run(name, method, target):

    output_root = ROOT / "outputs"

    candidates = [
        p for p in output_root.glob(name + "_*")
        if p.is_dir()
        and (p / "metrics/global_metrics.csv").is_file()
    ]

    exact = output_root / name

    if exact.is_dir() and (
        exact / "metrics/global_metrics.csv"
    ).is_file():
        candidates.append(exact)

    candidates.sort(
        key=lambda p: (
            p / "metrics/global_metrics.csv"
        ).stat().st_mtime,
        reverse=True,
    )

    last_found = 0

    for directory in candidates:

        rows = {
            int(float(row["round"])): row
            for row in read_csv(
                directory / "metrics/global_metrics.csv"
            )
            if row.get("round")
        }

        if rows:
            last_found = max(last_found, max(rows))

        if not all(r in rows for r in range(1, target + 1)):
            continue

        acc = {
            r: get_accuracy(rows[r], method)
            for r in range(1, target + 1)
        }

        comm = {
            r: float(
                rows[r]["cumulative_all_array_bytes"]
            )
            for r in range(1, target + 1)
        }

        assert all(
            comm[r] > comm[r-1]
            for r in range(2, target + 1)
        )

        assert all(
            isfinite(acc[r])
            for r in acc
        )

        return {
            "acc": acc,
            "comm": comm,
            "directory": directory,
        }

    raise RuntimeError(
        f"Missing/incomplete run:\n"
        f"  {name}\n"
        f"  Required R{target}, found through R{last_found}"
    )


def avg(acc, start, end):
    return mean(acc[r] for r in range(start, end + 1))


def std_window(acc, start, end):
    return pstdev(
        acc[r] for r in range(start, end + 1)
    )


def mean_sd(values):

    values = list(values)

    return mean(values), stdev(values)


def ci95(values):

    values = list(values)
    assert len(values) == 5

    m = mean(values)
    s = stdev(values)

    half_width = T_CRITICAL * s / sqrt(5)

    return m, m - half_width, m + half_width


def matched_metrics(f_run, a_run, rf, ra):

    assert ra == 2 * rf

    f = f_run["acc"]
    a = a_run["acc"]

    fb = f_run["comm"][rf]
    ab = a_run["comm"][ra]

    assert isclose(
        fb, ab,
        rel_tol=1e-9,
        abs_tol=1.0,
    ), (
        f"Communication mismatch: "
        f"FOLA R{rf}={fb}, FedAvg R{ra}={ab}"
    )

    # Communication-matched last windows:
    # FOLA: 20 rounds
    # FedAvg: 40 rounds
    f_tail = avg(f, rf-19, rf)
    a_tail = avg(a, ra-39, ra)

    # Preceding windows of the same communication width.
    f_previous = avg(f, rf-39, rf-20)
    a_previous = avg(a, ra-79, ra-40)

    return {
        "f": f[rf],
        "a": a[ra],
        "same_round_a": a[rf],

        "gap": f[rf] - a[ra],
        "same_round_gap": f[rf] - a[rf],

        "f_tail": f_tail,
        "a_tail": a_tail,
        "tail_gap": f_tail - a_tail,

        "f_delta": f_tail - f_previous,
        "a_delta": a_tail - a_previous,

        "f_std": std_window(f, rf-19, rf),
        "a_std": std_window(a, ra-39, ra),

        "budget_gb": fb / 1e9,
    }


# ============================================================
# Load all 30 experiments
# ============================================================

print("\n" + "=" * 85)
print("STAGE-15 CNN MULTI-SEED RESULTS")
print("=" * 85)

runs = {}
errors = []

for samples, lr, stage in GROUPS:
    for seed in SEEDS:
        for method in ("fola", "fedavg"):

            name = run_name(
                samples, lr, stage, seed, method
            )

            target = 150 if method == "fola" else 300

            try:
                runs[(samples, lr, seed, method)] = load_run(
                    name, method, target
                )
            except Exception as e:
                errors.append(str(e))

if errors:

    print("\nMISSING OR INCOMPLETE RUNS:")

    for error in errors:
        print(error)
        print("-" * 70)

    raise SystemExit(
        f"\nOnly {len(runs)}/30 runs available."
    )

assert len(runs) == 30

print("PASS: All 30 experiment trajectories loaded")
print("  Existing seed-0 runs: 6")
print("  New Stage-15 runs  : 24")
print("  Conditions         : 3")
print("  Independent seeds  : 5 per condition")


# ============================================================
# Construct seed-wise, communication-matched results
# ============================================================

records = {}

for samples, lr, _ in GROUPS:
    for rf, ra in MILESTONES:

        rows = []

        for seed in SEEDS:

            f_run = runs[(samples, lr, seed, "fola")]
            a_run = runs[(samples, lr, seed, "fedavg")]

            result = matched_metrics(
                f_run, a_run, rf, ra
            )

            result["seed"] = seed
            rows.append(result)

        records[(samples, lr, rf)] = rows


# ============================================================
# 1. Per-seed endpoint results
# ============================================================

print("\n" + "=" * 120)
print("1. PER-SEED ENDPOINT ACCURACY")
print("Same-round gap: FOLA R vs FedAvg R")
print("Equal-communication gap: FOLA R vs FedAvg 2R")
print("=" * 120)

print(
    "Samples LR    Rounds   Seed "
    " FOLA%  FedAvg%  SameRoundGap "
    " EqualCommGap  BudgetGB"
)

print("-" * 105)

for samples, lr, _ in GROUPS:
    for rf, ra in MILESTONES:

        for x in records[(samples, lr, rf)]:

            print(
                f"{samples:7d} "
                f"{lr:4.2f} "
                f"{rf:3d}/{ra:<3d} "
                f"{x['seed']:5d} "
                f"{x['f']:6.2f} "
                f"{x['a']:8.2f} "
                f"{x['same_round_gap']:+12.2f} "
                f"{x['gap']:+13.2f} "
                f"{x['budget_gb']:9.3f}"
            )

        print("-" * 105)


# ============================================================
# 2. Across-seed endpoint summary and 95% CI
# ============================================================

print("\n" + "=" * 145)
print("2. FIVE-SEED ENDPOINT SUMMARY")
print("95% CI is calculated from the five paired seed-wise gaps.")
print("=" * 145)

print(
    "Samples LR    Rounds  "
    "FOLA mean±SD    FedAvg mean±SD  "
    "Gap mean±SD   95% CI of gap       "
    "Wins"
)

print("-" * 145)

for samples, lr, _ in GROUPS:
    for rf, ra in MILESTONES:

        rows = records[(samples, lr, rf)]

        f = [x["f"] for x in rows]
        a = [x["a"] for x in rows]
        gaps = [x["gap"] for x in rows]

        fm, fs = mean_sd(f)
        am, ass = mean_sd(a)
        gm, gs = mean_sd(gaps)

        _, lo, hi = ci95(gaps)

        wins = sum(g > 0 for g in gaps)

        print(
            f"{samples:7d} "
            f"{lr:4.2f} "
            f"{rf:3d}/{ra:<3d} "
            f"{fm:6.2f}±{fs:5.2f} "
            f"{am:9.2f}±{ass:5.2f} "
            f"{gm:+7.2f}±{gs:5.2f} "
            f"[{lo:+7.2f}, {hi:+7.2f}] "
            f"{wins}/5"
        )


# ============================================================
# 3. Communication-matched final-tail summary
# ============================================================

print("\n" + "=" * 145)
print("3. FIVE-SEED FINAL-TAIL SUMMARY")
print("FOLA last 20 rounds vs FedAvg last 40 rounds")
print("Mean/SD across seeds; 95% CI for paired tail gaps")
print("=" * 145)

print(
    "Samples LR    Rounds "
    " FOLA tail±SD   FedAvg tail±SD "
    " Tail gap±SD    95% CI of tail gap   "
    "Wins"
)

print("-" * 145)

for samples, lr, _ in GROUPS:
    for rf, ra in MILESTONES:

        rows = records[(samples, lr, rf)]

        f = [x["f_tail"] for x in rows]
        a = [x["a_tail"] for x in rows]
        gaps = [x["tail_gap"] for x in rows]

        fm, fs = mean_sd(f)
        am, ass = mean_sd(a)
        gm, gs = mean_sd(gaps)

        _, lo, hi = ci95(gaps)

        wins = sum(g > 0 for g in gaps)

        print(
            f"{samples:7d} "
            f"{lr:4.2f} "
            f"{rf:3d}/{ra:<3d} "
            f"{fm:7.2f}±{fs:5.2f} "
            f"{am:10.2f}±{ass:5.2f} "
            f"{gm:+8.2f}±{gs:5.2f} "
            f"[{lo:+7.2f}, {hi:+7.2f}] "
            f"{wins}/5"
        )


# ============================================================
# 4. Late convergence and within-run stability
# ============================================================

print("\n" + "=" * 125)
print("4. LATE CONVERGENCE AND STABILITY")
print("Delta = current matched-byte tail minus preceding window")
print("Std = within-run accuracy variation, averaged across seeds")
print("=" * 125)

print(
    "Samples LR    Rounds "
    " FOLA delta±SD  FedAvg delta±SD "
    " FOLA windowSTD  FedAvg windowSTD"
)

print("-" * 125)

for samples, lr, _ in GROUPS:
    for rf, ra in MILESTONES:

        rows = records[(samples, lr, rf)]

        fd = [x["f_delta"] for x in rows]
        ad = [x["a_delta"] for x in rows]

        fs = [x["f_std"] for x in rows]
        ass = [x["a_std"] for x in rows]

        fdm, fds = mean_sd(fd)
        adm, ads = mean_sd(ad)

        print(
            f"{samples:7d} "
            f"{lr:4.2f} "
            f"{rf:3d}/{ra:<3d} "
            f"{fdm:+8.2f}±{fds:5.2f} "
            f"{adm:+11.2f}±{ads:5.2f} "
            f"{mean(fs):12.3f} "
            f"{mean(ass):15.3f}"
        )


# ============================================================
# 5. Ranking at the selected R150/R300 milestone
# ============================================================

print("\n" + "=" * 115)
print("5. RANKING AT FOLA R150 VS FEDAVG R300")
print("Ranked by mean paired final-tail gap across five seeds")
print("=" * 115)

ranking = []

for samples, lr, _ in GROUPS:

    rows = records[(samples, lr, 150)]

    tail_gaps = [x["tail_gap"] for x in rows]
    endpoint_gaps = [x["gap"] for x in rows]

    _, tail_lo, tail_hi = ci95(tail_gaps)

    ranking.append({
        "samples": samples,
        "lr": lr,
        "f_mean": mean(x["f"] for x in rows),
        "a_mean": mean(x["a"] for x in rows),
        "endpoint_gap": mean(endpoint_gaps),
        "tail_gap": mean(tail_gaps),
        "tail_ci": (tail_lo, tail_hi),
        "wins": sum(x > 0 for x in tail_gaps),
        "budget": rows[0]["budget_gb"],
    })

ranking.sort(
    key=lambda x: x["tail_gap"],
    reverse=True,
)

for rank, x in enumerate(ranking, 1):

    print(
        f"{rank}. "
        f"CNN | {x['samples']:2d} samples/client | "
        f"LR={x['lr']:.2f} | "
        f"FOLA={x['f_mean']:6.2f}% | "
        f"FedAvg={x['a_mean']:6.2f}% | "
        f"endpoint={x['endpoint_gap']:+6.2f} pp | "
        f"tail={x['tail_gap']:+6.2f} pp | "
        f"95%CI=[{x['tail_ci'][0]:+.2f}, "
        f"{x['tail_ci'][1]:+.2f}] | "
        f"wins={x['wins']}/5 | "
        f"budget={x['budget']:.3f} GB"
    )


# ============================================================
# 6. Overall completion
# ============================================================

print("\n" + "=" * 85)
print("PASS: STAGE-15 FIVE-SEED ANALYSIS COMPLETE")
print("=" * 85)

print("Compared settings:")
print("  CNN, 10 samples/client, LR=0.10")
print("  CNN,  5 samples/client, LR=0.10")
print("  CNN,  5 samples/client, LR=0.03")

print("\nCompared communication-matched milestones:")
print("  FOLA R120 vs FedAvg R240")
print("  FOLA R150 vs FedAvg R300")

print("\nSend sections 2, 3, 4 and 5 to ChatGPT.")
