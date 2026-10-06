from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


# ============================================================
# Locked run-name references from the runs we already created
# ============================================================
DENSE_FEDAVG = {
    "cifar_f1":
        "run_confirm_newsrc_cifar_basiccnn_a0p01_fedavg_n20_f1_seed0_e10_b32_lr001_r150",
    "cifar_f0p5":
        "run_confirm_newsrc_cifar_basiccnn_a0p01_fedavg_n20_f0p5_seed0_e10_b32_lr001_r150",
    "mnist_fixed":
        "run_confirm_newsrc_mnist_fixed1c_s10_fedavg_n100_seed0_e10_b32_lr001_r150",
}

DENSE_FOLA = {
    "cifar_f1":
        "run_confirm_newsrc_cifar_basiccnn_a0p01_fola_cos400_n20_f1_seed0_e10_b32_lr002_lam0p01_r150",
    "cifar_f0p5":
        "run_confirm_newsrc_cifar_basiccnn_a0p01_fola_cos400_n20_f0p5_seed0_e10_b32_lr002_lam0p01_r150",
    "mnist_fixed":
        "run_confirm_newsrc_mnist_fixed1c_s10_fola_n100_seed0_e10_b32_lr001_lam0p01_r150",
}

SPARSE_PREFIX = {
    "cifar_f1":
        "run_sparse_cifar_basiccnn_a0p01_f1_fola",
    "cifar_f0p5":
        "run_sparse_cifar_basiccnn_a0p01_f0p5_fola",
    "mnist_fixed":
        "run_sparse_mnist_fixed1c_s10_fola",
}

PROFILE_TITLE = {
    "cifar_f1":
        "CIFAR-10  |  alpha=0.01  |  local fraction=1.0",
    "cifar_f0p5":
        "CIFAR-10  |  alpha=0.01  |  local fraction=0.5",
    "mnist_fixed":
        "MNIST  |  fixed 10 samples/client  |  1 class/client",
}

KEEP_RATIOS = [0.75, 0.50, 0.25, 0.10]
RULES = ["kl_global_local", "kl_local_global", "random"]

RULE_LABEL = {
    "kl_global_local": "KL(global→local)",
    "kl_local_global": "KL(local→global)",
    "random": "Random",
}

RULE_LINESTYLE = {
    "kl_global_local": "-",
    "kl_local_global": "-.",
    "random": "--",
}

KEEP_COLOR = {
    1.00: "black",
    0.75: "tab:blue",
    0.50: "tab:green",
    0.25: "tab:orange",
    0.10: "tab:red",
}

KEEP_MARKER = {
    1.00: "o",
    0.75: "s",
    0.50: "^",
    0.25: "D",
    0.10: "x",
}


def ratio_tag(r: float) -> str:
    return format(r, ".8g").replace(".", "p")


def latest_run_dir(outputs_root: Path, run_name: str) -> Path:
    candidates: List[Path] = []

    exact = outputs_root / run_name
    if exact.is_dir():
        candidates.append(exact)

    candidates.extend(
        p for p in outputs_root.glob(run_name + "_*")
        if p.is_dir()
    )

    if not candidates:
        raise FileNotFoundError(f"No output directory found for run_name={run_name}")

    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_metrics_csv(run_dir: Path) -> Path:
    preferred = run_dir / "metrics" / "global_metrics.csv"
    if preferred.exists():
        return preferred

    fallback = run_dir / "global_metrics.csv"
    if fallback.exists():
        return fallback

    matches = list(run_dir.rglob("global_metrics.csv"))
    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(f"Cannot uniquely find global_metrics.csv under {run_dir}")


def read_curve(run_dir: Path, method: str) -> Dict[str, List[float]]:
    metrics_path = find_metrics_csv(run_dir)

    rounds = []
    acc = []
    comm_bytes = []

    with metrics_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        rv = row.get("round")
        if rv in (None, ""):
            rv = row.get("round_id")
        if rv in (None, ""):
            continue

        r = int(float(rv))

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

        a = None
        for key in keys:
            value = row.get(key)
            if value not in (None, ""):
                a = float(value)
                break

        if a is None:
            continue

        if a <= 1.5:
            a *= 100.0

        b = row.get("cumulative_all_array_bytes")
        if b in (None, ""):
            continue

        rounds.append(r)
        acc.append(a)
        comm_bytes.append(float(b))

    if not rounds:
        raise RuntimeError(f"No usable rows found in {metrics_path}")

    return {
        "round": rounds,
        "accuracy": acc,
        "bytes": comm_bytes,
    }


def bytes_scale(unit: str) -> float:
    unit = unit.upper()
    if unit == "GB":
        return 1e9
    if unit == "MB":
        return 1e6
    raise ValueError(f"Unsupported unit: {unit}")


def sparse_run_name(profile: str, rule: str, keep: float) -> str:
    kt = ratio_tag(keep)

    if profile == "mnist_fixed":
        return (
            f"run_sparse_mnist_fixed1c_s10_fola_"
            f"{rule}_keep{kt}_"
            f"n100_seed0_e10_b32_lr001_lam0p01_"
            f"budget150dense"
        )

    fraction = "f1" if profile == "cifar_f1" else "f0p5"

    return (
        f"run_sparse_cifar_basiccnn_a0p01_{fraction}_fola_"
        f"{rule}_keep{kt}_cos400_"
        f"n20_seed0_e10_b32_lr002_lam0p01_"
        f"budget150dense"
    )


def select_marker_points(rounds: List[int]) -> List[int]:
    # Show marker at every 10 rounds plus the final round.
    idx = [i for i, r in enumerate(rounds) if r % 10 == 0]
    if (len(rounds) - 1) not in idx:
        idx.append(len(rounds) - 1)
    # Also ensure the first point is visible.
    if 0 not in idx:
        idx.insert(0, 0)
    return sorted(set(idx))


def plot_profile(profile: str, outputs_root: Path, outdir: Path, unit: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    scale = bytes_scale(unit)

    fig, ax = plt.subplots(figsize=(12.5, 8.0))

    # ------------------------------------------------------------
    # Dense FedAvg reference
    # ------------------------------------------------------------
    fedavg_name = DENSE_FEDAVG[profile]
    fedavg_dir = latest_run_dir(outputs_root, fedavg_name)
    fedavg_curve = read_curve(fedavg_dir, method="fedavg")
    fedavg_rounds = fedavg_curve["round"]
    fedavg_x = [b / scale for b in fedavg_curve["bytes"]]
    fedavg_y = fedavg_curve["accuracy"]

    ax.plot(
        fedavg_x,
        fedavg_y,
        color="gray",
        linestyle="-",
        linewidth=2.2,
        alpha=0.95,
        label=f"FedAvg dense (R={fedavg_rounds[-1]})",
    )

    fed_idx = select_marker_points(fedavg_rounds)
    ax.plot(
        [fedavg_x[i] for i in fed_idx],
        [fedavg_y[i] for i in fed_idx],
        linestyle="None",
        marker="o",
        color="gray",
        markersize=4.5,
        alpha=0.95,
    )

    # ------------------------------------------------------------
    # Dense FOLA reference (keep=1.0)
    # ------------------------------------------------------------
    fola_name = DENSE_FOLA[profile]
    fola_dir = latest_run_dir(outputs_root, fola_name)
    fola_curve = read_curve(fola_dir, method="fola")
    fola_rounds = fola_curve["round"]
    fola_x = [b / scale for b in fola_curve["bytes"]]
    fola_y = fola_curve["accuracy"]

    ax.plot(
        fola_x,
        fola_y,
        color=KEEP_COLOR[1.00],
        linestyle="-",
        linewidth=2.8,
        alpha=0.95,
        label=f"Dense FOLA 100% (R={fola_rounds[-1]})",
    )

    dense_idx = select_marker_points(fola_rounds)
    ax.plot(
        [fola_x[i] for i in dense_idx],
        [fola_y[i] for i in dense_idx],
        linestyle="None",
        marker=KEEP_MARKER[1.00],
        color=KEEP_COLOR[1.00],
        markersize=4.8,
        alpha=0.95,
    )

    # ------------------------------------------------------------
    # Sparse runs
    # ------------------------------------------------------------
    printed = []

    for keep in KEEP_RATIOS:
        for rule in RULES:
            run_name = sparse_run_name(profile, rule, keep)

            try:
                run_dir = latest_run_dir(outputs_root, run_name)
            except FileNotFoundError:
                print(f"[WARN] Missing run for plot: {run_name}")
                continue

            curve = read_curve(run_dir, method="fola")
            rounds = curve["round"]
            x = [b / scale for b in curve["bytes"]]
            y = curve["accuracy"]

            color = KEEP_COLOR[keep]
            marker = KEEP_MARKER[keep]
            linestyle = RULE_LINESTYLE[rule]

            label = (
                f"{RULE_LABEL[rule]} keep={keep:.2f} "
                f"(R={rounds[-1]})"
            )

            ax.plot(
                x,
                y,
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                alpha=0.95,
                label=label,
            )

            idx = select_marker_points(rounds)
            ax.plot(
                [x[i] for i in idx],
                [y[i] for i in idx],
                linestyle="None",
                marker=marker,
                color=color,
                markersize=4.8,
                alpha=0.95,
            )

            printed.append(
                (rule, keep, rounds[-1], x[-1], y[-1], run_dir)
            )

    # ------------------------------------------------------------
    # Axes / layout
    # ------------------------------------------------------------
    ax.set_title(
        f"Accuracy vs Communication Cost\n{PROFILE_TITLE[profile]}",
        fontsize=14,
    )
    ax.set_xlabel(f"Cumulative communication cost ({unit.upper()})", fontsize=12)
    ax.set_ylabel("Global test accuracy (%)", fontsize=12)
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.8)

    # Put legend outside because there are many curves.
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        frameon=True,
    )

    fig.tight_layout()

    stem = outdir / f"{profile}_accuracy_vs_communication_{unit.lower()}"
    fig.savefig(str(stem) + ".png", dpi=220, bbox_inches="tight")
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    plt.close(fig)

    print()
    print("=" * 96)
    print(f"PLOT COMPLETE: {profile}")
    print("=" * 96)
    print(f"Dense FedAvg : {fedavg_dir}")
    print(f"Dense FOLA   : {fola_dir}")
    print()
    print(f"{'RULE':20s} {'KEEP':>6s} {'R_FINAL':>8s} {'COMM':>12s} {'ACC':>10s}")
    print("-" * 96)

    printed.sort(key=lambda x: (x[1], x[0]))
    for rule, keep, rfin, xfin, yfin, _ in printed:
        print(
            f"{RULE_LABEL[rule]:20s} "
            f"{keep:6.2f} "
            f"{rfin:8d} "
            f"{xfin:11.4f} {unit.upper():2s} "
            f"{yfin:9.3f}%"
        )

    print()
    print(f"Saved: {stem}.png")
    print(f"Saved: {stem}.pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        required=True,
        choices=["cifar_f1", "cifar_f0p5", "mnist_fixed"],
    )
    parser.add_argument(
        "--unit",
        default="GB",
        choices=["GB", "MB", "gb", "mb"],
    )
    parser.add_argument(
        "--outputs-root",
        default="./outputs",
    )
    parser.add_argument(
        "--outdir",
        default="./outputs/plots/acc_comm",
    )

    args = parser.parse_args()

    plot_profile(
        profile=args.profile,
        outputs_root=Path(args.outputs_root).resolve(),
        outdir=Path(args.outdir).resolve(),
        unit=args.unit.upper(),
    )


if __name__ == "__main__":
    main()
