from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PART = OUT / "partitions"

PLOT_DIR = (
    OUT /
    "plots" /
    "official_accuracy_round"
)


# ======================================================================
# Exact official seed-0 runs
# ======================================================================

CASES = [
    {
        "dataset": "CIFAR-10",
        "dataset_key": "cifar10",
        "alpha": 0.01,
        "fraction": 1.0,
        "clients": 20,
        "tag": "cifar10_a0p01_full",
        "fedavg": (
            "run_cifar_basiccnn_a0p01_official_"
            "fedavg_n20_f1_seed0_e10_b32_lr001_r150"
        ),
        "fola": (
            "run_cifar_basiccnn_a0p01_official_"
            "fola_cos400_n20_f1_seed0_"
            "e10_b32_lr002_lam0p01_r150"
        ),
    },
    {
        "dataset": "CIFAR-10",
        "dataset_key": "cifar10",
        "alpha": 0.01,
        "fraction": 0.5,
        "clients": 20,
        "tag": "cifar10_a0p01_s1250",
        "fedavg": (
            "run_cifar_basiccnn_a0p01_official_"
            "fedavg_n20_f0p5_seed0_e10_b32_lr001_r150"
        ),
        "fola": (
            "run_cifar_basiccnn_a0p01_official_"
            "fola_cos400_n20_f0p5_seed0_"
            "e10_b32_lr002_lam0p01_r150"
        ),
    },
    {
        "dataset": "CIFAR-10",
        "dataset_key": "cifar10",
        "alpha": 0.1,
        "fraction": 1.0,
        "clients": 20,
        "tag": "cifar10_a0p1_full",
        "fedavg": (
            "run_cifar_basiccnn_a0p1_official_"
            "fedavg_n20_f1_seed0_e10_b32_lr002_r150"
        ),
        "fola": (
            "run_cifar_basiccnn_a0p1_official_"
            "fola_cos400_n20_f1_seed0_"
            "e10_b32_lr002_lam0p1_r150"
        ),
    },
    {
        "dataset": "CIFAR-10",
        "dataset_key": "cifar10",
        "alpha": 0.1,
        "fraction": 0.5,
        "clients": 20,
        "tag": "cifar10_a0p1_s1250",
        "fedavg": (
            "run_cifar_basiccnn_a0p1_official_"
            "fedavg_n20_f0p5_seed0_e10_b32_lr002_r150"
        ),
        "fola": (
            "run_cifar_basiccnn_a0p1_official_"
            "fola_cos400_n20_f0p5_seed0_"
            "e10_b32_lr002_lam0p1_r150"
        ),
    },
    {
        "dataset": "MNIST",
        "dataset_key": "mnist",
        "alpha": 0.01,
        "fraction": 0.125,
        "clients": 100,
        "tag": "mnist_a0p01_s75",
        "fedavg": (
            "run_mnist_a0p01_official_"
            "fedavg_n100_s75_f0p125_seed0_"
            "e10_b32_lr001_r150"
        ),
        "fola": (
            "run_mnist_a0p01_official_"
            "fola_n100_s75_f0p125_seed0_"
            "e10_b32_lr001_lam0p01_r150"
        ),
    },
    {
        "dataset": "MNIST",
        "dataset_key": "mnist",
        "alpha": 0.01,
        "fraction": 1.0 / 60.0,
        "clients": 100,
        "tag": "mnist_a0p01_s10",
        "fedavg": (
            "run_mnist_a0p01_official_"
            "fedavg_n100_s10_f0p016667_seed0_"
            "e10_b32_lr001_r150"
        ),
        "fola": (
            "run_mnist_a0p01_official_"
            "fola_n100_s10_f0p016667_seed0_"
            "e10_b32_lr001_lam0p01_r150"
        ),
    },
    {
        "dataset": "MNIST",
        "dataset_key": "mnist",
        "alpha": None,
        "fraction": None,
        "clients": 100,
        "fixed_distribution": True,
        "samples_per_client": 10,
        "labels_per_client": 1,
        "tag": "mnist_fixed1c_s10",
        "fedavg": (
            "run_mnist_fixed1c_s10_official_"
            "fedavg_n100_seed0_"
            "e10_b32_lr001_r150"
        ),
        "fola": (
            "run_mnist_fixed1c_s10_official_"
            "fola_n100_seed0_"
            "e10_b32_lr001_lam0p01_r150"
        ),
    },

]


# ======================================================================
# Output helpers
# ======================================================================

def latest_run(stem: str) -> Path:
    exact = OUT / stem

    if exact.is_dir():
        return exact

    candidates = [
        p
        for p in OUT.glob(stem + "_*")
        if p.is_dir()
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No output directory found for:\n{stem}"
        )

    return max(
        candidates,
        key=lambda p: p.stat().st_mtime,
    )


def find_metrics(run_dir: Path) -> Path:
    candidates = [
        run_dir /
        "metrics" /
        "global_metrics.csv",

        run_dir /
        "global_metrics.csv",
    ]

    for p in candidates:
        if p.exists():
            return p

    found = list(
        run_dir.rglob(
            "global_metrics.csv"
        )
    )

    if not found:
        raise FileNotFoundError(
            f"No global_metrics.csv under {run_dir}"
        )

    return found[0]


# ======================================================================
# Read actual realized mean samples/client from partition
# ======================================================================

def partition_stem(
    dataset: str,
    alpha: float,
    fraction: float,
    clients: int,
):
    alpha_s = f"{alpha:g}"

    if abs(fraction - 1.0) < 1e-12:
        return (
            f"{dataset}_paper_dirichlet_"
            f"a{alpha_s}_n{clients}_seed0"
        )

    fraction_s = f"{fraction:g}"

    return (
        f"{dataset}_paper_dirichlet_"
        f"a{alpha_s}_f{fraction_s}_"
        f"n{clients}_seed0"
    )


def realized_mean_samples(case):
    if case.get("fixed_distribution", False):
        return float(
            case["samples_per_client"]
        )

    stem = partition_stem(
        case["dataset_key"],
        case["alpha"],
        case["fraction"],
        case["clients"],
    )

    json_path = PART / f"{stem}.json"

    if json_path.exists():
        meta = json.loads(
            json_path.read_text(
                encoding="utf-8"
            )
        )

        for key in [
            "mean_size",
            "mean_samples_per_client",
        ]:
            if key in meta:
                return float(meta[key])

    # Fallback: calculate directly from npz.
    npz_path = PART / f"{stem}.npz"

    if npz_path.exists():
        with np.load(
            npz_path,
            allow_pickle=False,
        ) as z:
            sizes = [
                len(z[k])
                for k in z.files
            ]

        return float(
            np.mean(sizes)
        )

    raise FileNotFoundError(
        "Could not determine realized samples/client.\n"
        f"Expected partition stem:\n{stem}"
    )


# ======================================================================
# Accuracy loading
# ======================================================================

def read_curve(
    stem: str,
    method: str,
):
    run_dir = latest_run(stem)
    path = find_metrics(run_dir)

    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    if not rows:
        raise RuntimeError(
            f"Empty metrics file: {path}"
        )

    headers = set(
        rows[0].keys()
    )

    if method == "fola":
        candidates = [
            "fola_mean_accuracy",
            "accuracy",
            "global_accuracy",
            "test_accuracy",
        ]
    else:
        candidates = [
            "accuracy",
            "global_accuracy",
            "test_accuracy",
        ]

    acc_key = next(
        (
            key
            for key in candidates
            if key in headers
        ),
        None,
    )

    if acc_key is None:
        raise RuntimeError(
            f"No recognized accuracy field: {path}\n"
            f"columns={sorted(headers)}"
        )

    values = {}

    for row in rows:
        try:
            rnd = int(
                float(row["round"])
            )

            acc = float(
                row[acc_key]
            )
        except Exception:
            continue

        if acc > 1.5:
            acc /= 100.0

        values[rnd] = acc

    if max(values) < 150:
        raise RuntimeError(
            f"Incomplete run: {stem}\n"
            f"last round=R{max(values)}"
        )

    return values


# ======================================================================
# Plot one official comparison
# ======================================================================

def plot_case(case):

    fed = read_curve(
        case["fedavg"],
        "fedavg",
    )

    fola = read_curve(
        case["fola"],
        "fola",
    )

    rounds = sorted(
        set(fed)
        & set(fola)
        & set(range(1, 151))
    )

    x = np.asarray(
        rounds,
        dtype=int,
    )

    fed_y = 100.0 * np.asarray(
        [
            fed[r]
            for r in rounds
        ]
    )

    fola_y = 100.0 * np.asarray(
        [
            fola[r]
            for r in rounds
        ]
    )

    fed_final = 100.0 * fed[150]
    fola_final = 100.0 * fola[150]

    samples = realized_mean_samples(
        case
    )

    # ----------------------------------------------------------
    # Figure
    # ----------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7.6, 5.2)
    )

    fed_line = ax.plot(
        x,
        fed_y,
        linewidth=2.2,
        label=(
            f"FedAvg "
            f"(final: {fed_final:.2f}%)"
        ),
    )[0]

    fola_line = ax.plot(
        x,
        fola_y,
        linewidth=2.2,
        label=(
            f"FOLA "
            f"(final: {fola_final:.2f}%)"
        ),
    )[0]

    # Final-round markers.
    ax.scatter(
        [150],
        [fed_final],
        s=45,
        color=fed_line.get_color(),
        zorder=5,
    )

    ax.scatter(
        [150],
        [fola_final],
        s=45,
        color=fola_line.get_color(),
        zorder=5,
    )

    # ----------------------------------------------------------
    # Endpoint labels
    # ----------------------------------------------------------

    # Offset labels slightly so they remain readable.
    final_gap = abs(
        fola_final - fed_final
    )

    if final_gap < 2.0:
        fed_offset = -13
        fola_offset = 10
    else:
        fed_offset = -5
        fola_offset = 5

    ax.annotate(
        f"{fed_final:.2f}%",
        xy=(150, fed_final),
        xytext=(-8, fed_offset),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=9,
        color=fed_line.get_color(),
        fontweight="bold",
    )

    ax.annotate(
        f"{fola_final:.2f}%",
        xy=(150, fola_final),
        xytext=(-8, fola_offset),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=9,
        color=fola_line.get_color(),
        fontweight="bold",
    )

    # ----------------------------------------------------------
    # Labels/title
    # ----------------------------------------------------------

    ax.set_xlabel(
        "Communication round"
    )

    ax.set_ylabel(
        "Global test accuracy (%)"
    )

    if case.get("fixed_distribution", False):
        title = (
            f"{case['dataset']} | Fixed distribution"
            "\n"
            f"{samples:.0f} samples/client | "
            f"{case['labels_per_client']} class/client"
        )
    else:
        title = (
            f"{case['dataset']} | "
            rf"$\alpha={case['alpha']:g}$"
            "\n"
            f"~{samples:.0f} samples/client"
        )

    ax.set_title(title)

    ax.set_xlim(
        1,
        154,
    )

    ymin = min(
        fed_y.min(),
        fola_y.min(),
    )

    ymax = max(
        fed_y.max(),
        fola_y.max(),
    )

    margin = max(
        2.0,
        0.08 * (ymax - ymin),
    )

    ax.set_ylim(
        max(
            0.0,
            ymin - margin,
        ),
        min(
            100.0,
            ymax + margin,
        ),
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

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample_tag = (
        f"s{samples:.0f}"
    )

    png = (
        PLOT_DIR /
        (
            f"{case['tag']}_"
            f"{sample_tag}_"
            "accuracy_vs_round.png"
        )
    )

    pdf = (
        PLOT_DIR /
        (
            f"{case['tag']}_"
            f"{sample_tag}_"
            "accuracy_vs_round.pdf"
        )
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

    setting = (
        "fixed 1-class/client"
        if case.get("fixed_distribution", False)
        else f"alpha={case['alpha']:g}"
    )

    print(
        f"{case['dataset']:8s} | "
        f"{setting} | "
        f"samples/client={samples:8.2f} | "
        f"FedAvg final={fed_final:6.2f}% | "
        f"FOLA final={fola_final:6.2f}% | "
        f"gap={fola_final-fed_final:+6.2f} pp"
    )

    print(
        "  ",
        png,
    )


# ======================================================================
# Main
# ======================================================================

def main():

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove only figures generated by this script,
    # leaving unrelated plots untouched.
    for p in PLOT_DIR.glob(
        "*_accuracy_vs_round.*"
    ):
        p.unlink()

    print()
    print("=" * 125)
    print(
        "OFFICIAL ACCURACY-vs-ROUND FIGURES "
        "| seed=0 | R150"
    )
    print("=" * 125)

    for case in CASES:
        plot_case(case)

    print()
    print("=" * 125)
    print("GENERATED 6 FIGURES")
    print("=" * 125)

    for p in sorted(
        PLOT_DIR.glob(
            "*_accuracy_vs_round.png"
        )
    ):
        print(p)


if __name__ == "__main__":
    main()
