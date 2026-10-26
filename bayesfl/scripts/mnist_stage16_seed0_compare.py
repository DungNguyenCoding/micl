from pathlib import Path
from statistics import mean, pstdev
import csv
import math
import json

from bayesfl.config import load_config
from bayesfl.data.datasets import partition_stem

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CONFIG = ROOT / "scripts/configs/mnist_stage16_paper00_fedavg"

ROUNDS = (120, 150, 200, 240, 300, 400)
MATCHED = ((120, 240), (150, 300), (200, 400))

CASES = [
    ("oldmlp", 5,  0.01, "OldMLP s5"),
    ("oldmlp", 10, 0.01, "OldMLP s10"),
    ("cnn16_32", 5,  0.03, "CNN s5 lr.03"),
    ("cnn16_32", 5,  0.10, "CNN s5 lr.10"),
    ("cnn16_32", 10, 0.03, "CNN s10 lr.03"),
    ("cnn16_32", 10, 0.10, "CNN s10 lr.10"),
]


def lr_tag(lr):
    return f"{lr:.2f}".replace(".", "p")


def strict_name(model, samples, lr):
    batch = 128 if model == "oldmlp" else samples

    return (
        f"run_stage16_paper00_fedavg_{model}_"
        f"s{samples}_n20_seed0_e5_b{batch}_"
        f"lr{lr_tag(lr)}_r400"
    )


def history_name(model, samples, lr, method):

    if model == "oldmlp":

        if samples == 5:
            return None

        lam = "_lam0p01" if method == "fola" else ""
        rounds = 200 if method == "fola" else 400

        return (
            f"debug_mnist_stage11_oldfixed_{method}_"
            f"n20_seed0_e5_b128_lr0p01{lam}_r{rounds}"
        )

    stage = 12 if samples == 10 else 14
    lam = "_lam0p01" if method == "fola" else ""
    rounds = 200 if method == "fola" else 400

    return (
        f"debug_mnist_stage{stage}_cnn16_32_"
        f"fixed1c_s{samples}_n20_{method}_"
        f"seed0_e5_b{samples}_lr{lr_tag(lr)}"
        f"{lam}_r{rounds}"
    )


def parse_accuracy(row, method):

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

        if math.isfinite(value):
            return value * 100 if value <= 1.000001 else value

    raise ValueError(
        f"No valid accuracy column for {method}: {list(row)}"
    )


def load_run(name, method, required_rounds, required=False):

    if name is None:
        return None

    candidates = [
        OUT / name,
        *OUT.glob(name + "_*"),
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

    for directory in candidates:

        resolved = directory / "resolved_config.yaml"

        cfg = load_config(resolved) if resolved.is_file() else None

        if cfg is not None:
            if cfg.run_name != name or cfg.method != method:
                continue

        with (
            directory / "metrics/global_metrics.csv"
        ).open(newline="", encoding="utf-8") as file:

            rows = {
                int(float(row["round"])): row
                for row in csv.DictReader(file)
                if row.get("round")
            }

        if not all(
            r in rows for r in range(1, required_rounds + 1)
        ):
            continue

        accuracy = {
            r: parse_accuracy(rows[r], method)
            for r in range(1, required_rounds + 1)
        }

        communication = {
            r: float(rows[r]["cumulative_all_array_bytes"])
            for r in range(1, required_rounds + 1)
        }

        assert all(
            math.isfinite(communication[r])
            and communication[r] > 0
            for r in communication
        )

        assert all(
            communication[r] > communication[r-1]
            for r in range(2, required_rounds + 1)
        )

        return {
            "acc": accuracy,
            "comm": communication,
            "config": cfg,
            "directory": directory,
        }

    if required:
        raise RuntimeError(
            f"Missing or incomplete R{required_rounds}: {name}"
        )

    return None


def window(run, start, end):
    return [
        run["acc"][r]
        for r in range(start, end + 1)
    ]


def verify_pair(fola, fedavg, model, samples, lr):

    fc = fola["config"]
    ac = fedavg["config"]

    if fc is not None and ac is not None:

        assert fc.model.name == ac.model.name
        assert fc.runtime.seed == ac.runtime.seed == 0

        assert partition_stem(fc) == partition_stem(ac), (
            model, samples, lr, "PARTITION MISMATCH"
        )

        assert fc.training.lr == ac.training.lr == lr
        assert fc.training.local_epochs == ac.training.local_epochs == 5

        assert (
            fc.federation.num_clients
            == ac.federation.num_clients
            == 20
        )

    # Verify actual partition hashes, when available.
    fp = fola["directory"] / "partition_metadata.json"
    ap = fedavg["directory"] / "partition_metadata.json"

    if fp.is_file() and ap.is_file():

        fh = json.loads(fp.read_text()).get("sha256")
        ah = json.loads(ap.read_text()).get("sha256")

        if fh and ah:
            assert fh == ah, (
                model, samples, lr, "PARTITION HASH MISMATCH"
            )

    # Verify actual recorded equal-communication budgets.
    for rf, ra in MATCHED:

        fb = fola["comm"][rf]
        ab = fedavg["comm"][ra]

        assert math.isclose(
            fb, ab, rel_tol=1e-9, abs_tol=1.0
        ), (
            model, samples, lr,
            f"Budget mismatch FOLA R{rf}/FedAvg R{ra}",
            fb, ab,
        )


# ============================================================
# LOAD SIX STRICT PAPER-00 FEDAVG RUNS
# ============================================================

strict = {}
old_fola = {}
old_fedavg = {}
missing = []

for model, samples, lr, label in CASES:

    key = (model, samples, lr)

    name = strict_name(model, samples, lr)
    cfg_path = CONFIG / f"{name}.yaml"

    if not cfg_path.is_file():
        missing.append(f"Missing config: {cfg_path}")
        continue

    cfg = load_config(cfg_path)

    assert cfg.method == "fedavg"
    assert cfg.runtime.seed == 0
    assert cfg.training.rounds == 400
    assert cfg.training.paper00_strict is True
    assert cfg.training.grad_clip_norm is None
    assert cfg.training.momentum == 0.0
    assert cfg.training.weight_decay == 0.0

    try:
        strict[key] = load_run(
            name, "fedavg", 400, required=True
        )
    except RuntimeError as error:
        missing.append(str(error))
        continue

    old_fola[key] = load_run(
        history_name(model, samples, lr, "fola"),
        "fola",
        200,
    )

    old_fedavg[key] = load_run(
        history_name(model, samples, lr, "fedavg"),
        "fedavg",
        400,
    )

    if old_fola[key] is not None:
        verify_pair(
            old_fola[key],
            strict[key],
            model,
            samples,
            lr,
        )

    if old_fedavg[key] is not None:

        old = old_fedavg[key]
        new = strict[key]

        if old["config"] is not None:

            assert old["config"].training.grad_clip_norm == 10.0

            if new["config"] is not None:
                assert (
                    partition_stem(old["config"])
                    == partition_stem(new["config"])
                )

        for r in ROUNDS:
            assert math.isclose(
                old["comm"][r],
                new["comm"][r],
                rel_tol=1e-9,
                abs_tol=1.0,
            )


if missing:

    print("\nMISSING OR INCOMPLETE STRICT FEDAVG RUNS:")

    for item in missing:
        print(" -", item)

    raise SystemExit(
        f"Only {len(strict)}/6 seed-0 runs completed."
    )


# ============================================================
# SECTION 1: STRICT FEDAVG ACCURACY
# ============================================================

print("\n" + "=" * 120)
print("1. STRICT PAPER-00 FEDAVG — SEED 0")
print("Plain SGD, no clipping, 400 rounds")
print("=" * 120)

print(
    f"{'Setting':15s} "
    f"{'R120':>7s} {'R150':>7s} {'R200':>7s} "
    f"{'R240':>7s} {'R300':>7s} {'R400':>7s} "
    f"{'Best@R':>14s} {'TailSD':>8s} {'Delta400':>9s}"
)

for model, samples, lr, label in CASES:

    d = strict[(model, samples, lr)]
    a = d["acc"]

    best_round = max(a, key=a.get)

    tail_std = pstdev(window(d, 361, 400))

    delta = (
        mean(window(d, 381, 400))
        - mean(window(d, 341, 360))
    )

    row = (
        f"{label:15s} "
        + " ".join(f"{a[r]:7.2f}" for r in ROUNDS)
    )

    row += (
        f" {a[best_round]:6.2f}@R{best_round:<4d}"
        f" {tail_std:8.2f}"
        f" {delta:+9.2f}"
    )

    print(row)


# ============================================================
# SECTION 2: FOLA VS STRICT FEDAVG AT EQUAL BYTES
# ============================================================

print("\n" + "=" * 125)
print("2. HISTORICAL FOLA VS STRICT PAPER-00 FEDAVG")
print("Equal-communication comparison: R120/240, R150/300, R200/400")
print("FOLA = historical clipped | FedAvg = new unclipped")
print("=" * 125)

print(
    f"{'Setting':15s} {'Rounds':12s} "
    f"{'FOLA':>8s} {'FedAvg':>8s} "
    f"{'Gap':>9s} {'TailGap':>9s} "
    f"{'BudgetGB':>10s} "
    f"{'F-delta':>9s} {'A-delta':>9s}"
)

rank150 = []

for model, samples, lr, label in CASES:

    key = (model, samples, lr)

    f = old_fola[key]
    a = strict[key]

    if f is None:

        print(
            f"{label:15s} "
            "N/A — no completed matching FOLA run"
        )
        continue

    for rf, ra in MATCHED:

        f_tail = mean(window(f, rf-19, rf))
        a_tail = mean(window(a, ra-39, ra))

        f_previous = mean(
            window(f, rf-39, rf-20)
        )

        a_previous = mean(
            window(a, ra-79, ra-40)
        )

        gap = f["acc"][rf] - a["acc"][ra]
        tail_gap = f_tail - a_tail

        budget = f["comm"][rf] / 1e9

        f_delta = f_tail - f_previous
        a_delta = a_tail - a_previous

        print(
            f"{label:15s} R{rf:<3d}/R{ra:<3d}   "
            f"{f['acc'][rf]:8.2f} "
            f"{a['acc'][ra]:8.2f} "
            f"{gap:+9.2f} "
            f"{tail_gap:+9.2f} "
            f"{budget:10.3f} "
            f"{f_delta:+9.2f} "
            f"{a_delta:+9.2f}"
        )

        if rf == 150:

            rank150.append((
                tail_gap,
                label,
                gap,
                f_tail,
                a_tail,
            ))


# ============================================================
# SECTION 3: EFFECT OF REMOVING GRADIENT CLIPPING
# ============================================================

print("\n" + "=" * 115)
print("3. GRADIENT CLIPPING ABLATION")
print("New strict FedAvg MINUS historical clipped FedAvg")
print("Positive = unclipped FedAvg achieves higher accuracy")
print("=" * 115)

print(
    f"{'Setting':15s} "
    f"{'R120':>8s} {'R150':>8s} {'R200':>8s} "
    f"{'R240':>8s} {'R300':>8s} {'R400':>8s} "
    f"{'TailDiff':>9s}"
)

for model, samples, lr, label in CASES:

    key = (model, samples, lr)

    old = old_fedavg[key]
    new = strict[key]

    if old is None:

        print(
            f"{label:15s} "
            "N/A — no completed matching historical FedAvg R400"
        )
        continue

    differences = [
        new["acc"][r] - old["acc"][r]
        for r in ROUNDS
    ]

    tail_difference = (
        mean(window(new, 361, 400))
        - mean(window(old, 361, 400))
    )

    print(
        f"{label:15s} "
        + " ".join(f"{d:+8.2f}" for d in differences)
        + f" {tail_difference:+9.2f}"
    )


# ============================================================
# SECTION 4: SAME-ROUND COMPARISON
# ============================================================

print("\n" + "=" * 100)
print("4. SAME-ROUND FOLA VS STRICT PAPER-00 FEDAVG")
print("FOLA - FedAvg at R120, R150 and R200")
print("Note: same rounds, NOT same communication cost")
print("=" * 100)

print(
    f"{'Setting':15s} "
    f"{'R120 gap':>10s} "
    f"{'R150 gap':>10s} "
    f"{'R200 gap':>10s}"
)

for model, samples, lr, label in CASES:

    key = (model, samples, lr)

    f = old_fola[key]
    a = strict[key]

    if f is None:
        print(f"{label:15s} N/A")
        continue

    print(
        f"{label:15s} "
        + " ".join(
            f"{f['acc'][r] - a['acc'][r]:+10.2f}"
            for r in (120, 150, 200)
        )
    )


# ============================================================
# SECTION 5: R150/R300 RANKING
# ============================================================

print("\n" + "=" * 115)
print("5. RANKING AT EQUAL COMMUNICATION COST")
print("FOLA R131–150 versus strict FedAvg R261–300")
print("Ranking uses final-tail gap, NOT one endpoint")
print("=" * 115)

for rank, record in enumerate(
    sorted(rank150, reverse=True),
    1,
):

    gap, label, endpoint_gap, f_tail, a_tail = record

    print(
        f"{rank}. {label:15s} | "
        f"tail gap={gap:+7.2f} pp | "
        f"endpoint gap={endpoint_gap:+7.2f} pp | "
        f"FOLA={f_tail:.2f}% | "
        f"FedAvg={a_tail:.2f}%"
    )


print("\n" + "=" * 85)
print("PASS: STAGE-16 SEED-0 COMPARISON COMPLETE")
print("=" * 85)

print("Six strict Paper-00 FedAvg R400 runs analyzed.")
print("Historical FOLA is still clipped at norm=10.")
print("No new simulations were run.")

print(
    "\nSend sections 1, 2, 3, 4 and 5 to ChatGPT."
)
