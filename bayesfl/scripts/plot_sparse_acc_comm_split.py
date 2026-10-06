from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


PROFILE_TITLE = {
    "cifar_f1":
        "CIFAR-10 | alpha=0.01 | local fraction=1.0",

    "cifar_f0p5":
        "CIFAR-10 | alpha=0.01 | local fraction=0.5",

    "mnist_fixed":
        "MNIST | fixed 10 samples/client | 1 class/client",
}


KEEP_RATIOS = [0.75, 0.50, 0.25, 0.10]

RULES = [
    "kl_global_local",
    "kl_local_global",
    "random",
]


RULE_LABEL = {
    "kl_global_local": "KL(global→local)",
    "kl_local_global": "KL(local→global)",
    "random": "Random",
}


RULE_STYLE = {
    "kl_global_local": "-",
    "kl_local_global": "-.",
    "random": "--",
}


# One ratio per plot, so method color is more readable
# than assigning color by keep ratio.
RULE_COLOR = {
    "kl_global_local": "tab:blue",
    "kl_local_global": "tab:orange",
    "random": "tab:green",
}


def ratio_tag(r):
    return format(r, ".8g").replace(".", "p")


def sparse_run_name(profile, rule, keep):
    kt = ratio_tag(keep)

    if profile == "mnist_fixed":
        return (
            f"run_sparse_mnist_fixed1c_s10_fola_"
            f"{rule}_keep{kt}_"
            f"n100_seed0_e10_b32_lr001_lam0p01_"
            f"budget150dense"
        )

    fraction = (
        "f1"
        if profile == "cifar_f1"
        else "f0p5"
    )

    return (
        f"run_sparse_cifar_basiccnn_a0p01_{fraction}_fola_"
        f"{rule}_keep{kt}_cos400_"
        f"n20_seed0_e10_b32_lr002_lam0p01_"
        f"budget150dense"
    )


def latest_run_dir(root, run_name):
    candidates = []

    exact = root / run_name

    if exact.is_dir():
        candidates.append(exact)

    candidates.extend(
        p
        for p in root.glob(run_name + "_*")
        if p.is_dir()
    )

    if not candidates:
        raise FileNotFoundError(
            f"No output found for {run_name}"
        )

    return max(
        candidates,
        key=lambda p: p.stat().st_mtime,
    )


def find_metrics(run_dir):
    preferred = (
        run_dir
        / "metrics"
        / "global_metrics.csv"
    )

    if preferred.exists():
        return preferred

    fallback = (
        run_dir
        / "global_metrics.csv"
    )

    if fallback.exists():
        return fallback

    found = list(
        run_dir.rglob(
            "global_metrics.csv"
        )
    )

    if len(found) == 1:
        return found[0]

    raise RuntimeError(
        f"Cannot uniquely find metrics under {run_dir}"
    )


def read_curve(run_dir, method):
    path = find_metrics(run_dir)

    rounds = []
    x_gb = []
    accuracy = []

    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    for row in rows:

        rv = row.get("round")

        if rv in (None, ""):
            rv = row.get("round_id")

        if rv in (None, ""):
            continue

        r = int(
            float(rv)
        )

        if r < 1:
            continue

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
            v = row.get(key)

            if v not in (
                None,
                "",
            ):
                a = float(v)
                break

        if a is None:
            continue

        if a <= 1.5:
            a *= 100.0

        b = row.get(
            "cumulative_all_array_bytes"
        )

        if b in (
            None,
            "",
        ):
            continue

        rounds.append(r)

        x_gb.append(
            float(b) / 1e9
        )

        accuracy.append(a)

    if not rounds:
        raise RuntimeError(
            f"No usable measurements in {path}"
        )

    order = np.argsort(
        rounds
    )

    return (
        np.asarray(
            rounds,
            dtype=int,
        )[order],
        np.asarray(
            x_gb,
            dtype=float,
        )[order],
        np.asarray(
            accuracy,
            dtype=float,
        )[order],
    )


def local_linear_gaussian(
    x,
    y,
    bandwidth=0.5,
):
    """
    Local-linear Gaussian kernel regression.

    Unlike a simple weighted moving average, the local
    linear fit greatly reduces boundary bias near the
    beginning/end of the training curve.
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    out = np.empty_like(
        y,
        dtype=float,
    )

    for i, x0 in enumerate(x):

        dx = x - x0

        w = np.exp(
            -0.5
            * (dx / bandwidth) ** 2
        )

        # Local model:
        #
        # y ~= beta0 + beta1 * (x-x0)
        #
        # Prediction at x0 is beta0.
        X = np.column_stack(
            [
                np.ones_like(dx),
                dx,
            ]
        )

        WX = (
            X
            * w[:, None]
        )

        A = X.T @ WX

        b = X.T @ (
            w * y
        )

        try:
            beta = np.linalg.solve(
                A,
                b,
            )

            out[i] = beta[0]

        except np.linalg.LinAlgError:

            sw = w.sum()

            out[i] = (
                np.dot(w, y) / sw
                if sw > 0
                else y[i]
            )

    return out


def marker_indices(rounds):
    """
    Show:
      - first round
      - every 10th training round
      - final round

    Small faint markers for ALL rounds are plotted separately.
    """

    idx = [
        i
        for i, r in enumerate(rounds)
        if r % 10 == 0
    ]

    idx.append(0)
    idx.append(
        len(rounds) - 1
    )

    return sorted(
        set(idx)
    )


def prepare_series(
    root,
    profile,
    keep,
):
    curves = []

    # ========================================================
    # Dense FedAvg
    # ========================================================
    fed_dir = latest_run_dir(
        root,
        DENSE_FEDAVG[
            profile
        ],
    )

    r, x, y = read_curve(
        fed_dir,
        "fedavg",
    )

    curves.append(
        {
            "key": "fedavg",
            "label":
                f"FedAvg dense (R={r[-1]})",
            "rounds": r,
            "x": x,
            "y": y,
            "color": "0.55",
            "style": "-",
            "linewidth": 2.0,
            "marker": "o",
        }
    )

    # ========================================================
    # Dense FOLA
    # ========================================================
    dense_dir = latest_run_dir(
        root,
        DENSE_FOLA[
            profile
        ],
    )

    r, x, y = read_curve(
        dense_dir,
        "fola",
    )

    curves.append(
        {
            "key": "dense_fola",
            "label":
                f"Dense FOLA, keep=1.00 (R={r[-1]})",
            "rounds": r,
            "x": x,
            "y": y,
            "color": "black",
            "style": "-",
            "linewidth": 2.5,
            "marker": "o",
        }
    )

    # ========================================================
    # Sparse methods
    # ========================================================
    for rule in RULES:

        run_name = sparse_run_name(
            profile,
            rule,
            keep,
        )

        run_dir = latest_run_dir(
            root,
            run_name,
        )

        r, x, y = read_curve(
            run_dir,
            "fola",
        )

        curves.append(
            {
                "key": rule,
                "label":
                    f"{RULE_LABEL[rule]}, "
                    f"keep={keep:.2f} "
                    f"(R={r[-1]})",
                "rounds": r,
                "x": x,
                "y": y,
                "color":
                    RULE_COLOR[rule],
                "style":
                    RULE_STYLE[rule],
                "linewidth": 2.2,
                "marker": "x",
            }
        )

    return curves


def plot_one(
    root,
    output_dir,
    profile,
    keep,
    bandwidth,
):

    curves = prepare_series(
        root,
        profile,
        keep,
    )

    fig, ax = plt.subplots(
        figsize=(10.5, 7.0)
    )

    all_y = []

    for c in curves:

        x = c["x"]
        y = c["y"]

        smooth = (
            local_linear_gaussian(
                x,
                y,
                bandwidth=bandwidth,
            )
        )

        all_y.extend(
            y.tolist()
        )

        # ----------------------------------------------------
        # Smoothed trend
        # ----------------------------------------------------
        ax.plot(
            x,
            smooth,
            color=c["color"],
            linestyle=c["style"],
            linewidth=c["linewidth"],
            label=c["label"],
            zorder=3,
        )

        # ----------------------------------------------------
        # Tiny raw observations at EVERY training round.
        #
        # This preserves visibility of the actual evaluated
        # data and shows the communication spacing per round.
        # ----------------------------------------------------
        ax.scatter(
            x,
            y,
            color=c["color"],
            s=7,
            alpha=0.18,
            linewidths=0,
            zorder=2,
        )

        # ----------------------------------------------------
        # Larger markers every 10 rounds + endpoint.
        # ----------------------------------------------------
        idx = marker_indices(
            c["rounds"]
        )

        if c["marker"] == "x":
            ax.scatter(
                x[idx],
                y[idx],
                color=c["color"],
                marker="x",
                s=27,
                linewidths=1.0,
                alpha=0.82,
                zorder=4,
            )

        else:
            ax.scatter(
                x[idx],
                y[idx],
                color=c["color"],
                marker="o",
                s=20,
                alpha=0.75,
                linewidths=0,
                zorder=4,
            )

        # Final endpoint is emphasized.
        ax.scatter(
            [x[-1]],
            [y[-1]],
            color=c["color"],
            marker=c["marker"],
            s=58,
            linewidths=1.5,
            zorder=6,
        )

    # ========================================================
    # Use similar y-axis scale across ratio plots.
    # ========================================================
    ymin = (
        np.floor(
            min(all_y) / 5.0
        ) * 5.0
        - 2.0
    )

    ymax = (
        np.ceil(
            max(all_y) / 5.0
        ) * 5.0
        + 2.0
    )

    ax.set_ylim(
        ymin,
        ymax,
    )

    ax.set_xlim(
        left=0.0
    )

    ax.set_xlabel(
        "Cumulative communication cost (GB)",
        fontsize=12,
    )

    ax.set_ylabel(
        "Global test accuracy (%)",
        fontsize=12,
    )

    ax.set_title(
        (
            f"{PROFILE_TITLE[profile]}\n"
            f"Keep ratio = {keep:.2f}"
        ),
        fontsize=14,
    )

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.75,
        alpha=0.7,
    )

    ax.legend(
        loc="lower right",
        fontsize=9.5,
        frameon=True,
    )

    fig.tight_layout()

    tag = ratio_tag(
        keep
    )

    stem = (
        output_dir
        / f"{profile}_keep{tag}_accuracy_vs_communication"
    )

    fig.savefig(
        str(stem) + ".png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        str(stem) + ".pdf",
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print("=" * 90)
    print(
        f"{profile} | keep={keep:.2f}"
    )
    print("=" * 90)

    print(
        f"Local-linear smoothing bandwidth: "
        f"{bandwidth:.2f} GB"
    )

    for c in curves:
        print(
            f"{c['label']:48s} | "
            f"final={c['y'][-1]:7.3f}% | "
            f"GB={c['x'][-1]:8.3f}"
        )

    print(
        f"PNG: {stem}.png"
    )

    print(
        f"PDF: {stem}.pdf"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--profile",
        required=True,
        choices=[
            "cifar_f1",
            "cifar_f0p5",
            "mnist_fixed",
        ],
    )

    parser.add_argument(
        "--bandwidth",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--outputs-root",
        default="./outputs",
    )

    parser.add_argument(
        "--outdir",
        default="./outputs/plots/acc_comm_split",
    )

    args = parser.parse_args()

    if args.bandwidth <= 0:
        raise ValueError(
            "--bandwidth must be positive"
        )

    root = Path(
        args.outputs_root
    ).resolve()

    output_dir = Path(
        args.outdir
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for keep in KEEP_RATIOS:
        plot_one(
            root=root,
            output_dir=output_dir,
            profile=args.profile,
            keep=keep,
            bandwidth=args.bandwidth,
        )


if __name__ == "__main__":
    main()
