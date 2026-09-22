from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PLOT_DIR = OUT / "plots" / "basiccnn_lr_schedule_compare"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


CASES = {
    ".1": {
        "title": r"CIFAR-10 BasicCNN — $\alpha=0.1$",
        "output": "cifar10_basiccnn_a0p1_accuracy_vs_round",
        "runs": [
            (
                "FOLA cosine (LR=.02)",
                "fola",
                "debug_cifar_basiccnn_a0p1_"
                "fola_cos400_full_seed0_"
                "e10_b32_lr002_lam0p1_r50",
            ),
            (
                "FOLA constant (LR=.02)",
                "fola",
                "debug_cifar_basiccnn_a0p1_"
                "fola_constant_full_seed0_"
                "e10_b32_lr002_lam0p1_r50",
            ),
            (
                "FedAvg constant (LR=.02)",
                "fedavg",
                "debug_cifar_basiccnn_a0p1_"
                "fedavg_constant_full_seed0_"
                "e10_b32_lr002_r50",
            ),
        ],
    },

    ".01": {
        "title": r"CIFAR-10 BasicCNN — $\alpha=0.01$",
        "output": "cifar10_basiccnn_a0p01_accuracy_vs_round",
        "runs": [
            (
                "FOLA cosine (base LR=.02)",
                "fola",
                "debug_cifar_basiccnn_a0p01_"
                "fola_cos400_full_seed0_"
                "e10_b32_lr002_lam0p01_r50",
            ),
            (
                "FOLA constant (LR=.01)",
                "fola",
                "debug_cifar_basiccnn_a0p01_"
                "fola_constant_full_seed0_"
                "e10_b32_lr001_lam0p01_r50",
            ),
            (
                "FedAvg constant (LR=.01)",
                "fedavg",
                "debug_cifar_basiccnn_a0p01_"
                "fedavg_constant_full_seed0_"
                "e10_b32_lr001_r50",
            ),
        ],
    },
}


def latest_run(stem):
    exact = OUT / stem

    if exact.is_dir():
        return exact

    candidates = [
        p
        for p in OUT.glob(stem + "_*")
        if p.is_dir()
    ]

    if not candidates:
        raise RuntimeError(
            f"Could not find output for:\n{stem}"
        )

    return max(
        candidates,
        key=lambda p: p.stat().st_mtime,
    )


def find_metrics(run_dir):
    candidates = [
        run_dir / "metrics" / "global_metrics.csv",
        run_dir / "global_metrics.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    found = list(
        run_dir.rglob("global_metrics.csv")
    )

    if not found:
        raise RuntimeError(
            f"No global_metrics.csv under:\n{run_dir}"
        )

    return found[0]


def read_curve(stem, method):
    run_dir = latest_run(stem)
    metrics = find_metrics(run_dir)

    with metrics.open(
        newline="",
        encoding="utf-8",
    ) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(
            f"Empty metrics file: {metrics}"
        )

    if method == "fola":
        keys = [
            "fola_mean_accuracy",
            "accuracy",
            "global_accuracy",
            "test_accuracy",
        ]
    else:
        keys = [
            "accuracy",
            "global_accuracy",
            "test_accuracy",
        ]

    series = {}

    for row in rows:
        try:
            rnd = int(float(row["round"]))
        except Exception:
            continue

        value = None

        for key in keys:
            try:
                value = float(row[key])
                break
            except Exception:
                pass

        if value is None:
            continue

        if value > 1.5:
            value /= 100.0

        series[rnd] = value

    missing = [
        r
        for r in range(1, 51)
        if r not in series
    ]

    if missing:
        raise RuntimeError(
            f"{run_dir.name}: missing rounds {missing}"
        )

    rounds = np.arange(1, 51)

    accuracy = np.asarray(
        [
            100.0 * series[r]
            for r in rounds
        ]
    )

    return rounds, accuracy, run_dir


def summarize(acc):
    return {
        "R10": acc[9],
        "R20": acc[19],
        "R30": acc[29],
        "R40": acc[39],
        "R50": acc[49],
        "last10": acc[40:50].mean(),
        "mean": acc.mean(),
    }


for alpha, case in CASES.items():

    print()
    print("=" * 110)
    print(case["title"].replace("$", ""))
    print("=" * 110)

    fig, ax = plt.subplots(
        figsize=(7.4, 5.2)
    )

    all_acc = []

    for label, method, stem in case["runs"]:
        rounds, acc, run_dir = read_curve(
            stem,
            method,
        )

        all_acc.append(acc)

        ax.plot(
            rounds,
            acc,
            linewidth=2.2,
            label=label,
        )

        s = summarize(acc)

        print()
        print(label)
        print("  output  =", run_dir.name)
        print(f"  R10     = {s['R10']:.2f}%")
        print(f"  R20     = {s['R20']:.2f}%")
        print(f"  R30     = {s['R30']:.2f}%")
        print(f"  R40     = {s['R40']:.2f}%")
        print(f"  R50     = {s['R50']:.2f}%")
        print(f"  Last-10 = {s['last10']:.2f}%")
        print(f"  Mean    = {s['mean']:.2f}%")

    combined = np.concatenate(all_acc)

    lower = max(
        0.0,
        np.floor(combined.min() / 5.0) * 5.0 - 2.5,
    )

    upper = min(
        100.0,
        np.ceil(combined.max() / 5.0) * 5.0 + 2.5,
    )

    ax.set_xlim(1, 50)
    ax.set_ylim(lower, upper)

    ax.set_xticks(
        [1, 10, 20, 30, 40, 50]
    )

    ax.set_xlabel(
        "Communication round"
    )

    ax.set_ylabel(
        "Global test accuracy (%)"
    )

    ax.set_title(
        case["title"]
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend(
        frameon=False,
        loc="lower right",
    )

    fig.tight_layout()

    png = (
        PLOT_DIR /
        f"{case['output']}.png"
    )

    pdf = (
        PLOT_DIR /
        f"{case['output']}.pdf"
    )

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print("PNG =", png)
    print("PDF =", pdf)


print()
print("=" * 110)
print("DONE")
print("=" * 110)
