from pathlib import Path
import csv
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "plots/stage15_accuracy_comm"
OUT.mkdir(parents=True, exist_ok=True)

GROUPS = [
    ("s10_lr0p10", 10, 0.10, 12),
    ("s5_lr0p10",   5, 0.10, 14),
    ("s5_lr0p03",   5, 0.03, 14),
]

SEEDS = range(5)
TARGET = {"fola": 150, "fedavg": 300}

COLORS = {
    "fola": "tab:blue",
    "fedavg": "tab:orange",
}

STYLES = {
    "fola": "-",
    "fedavg": "--",
}

LABELS = {
    "fola": "Dense FOLA",
    "fedavg": "FedAvg",
}

WINDOW = 11
PLOT_COUNT = 0


# ============================================================
# Expected experiment names
# ============================================================

def run_name(samples, lr, source_stage, seed, method):

    lr_tag = f"{lr:.2f}".replace(".", "p")
    lam = "_lam0p01" if method == "fola" else ""

    if seed == 0:
        rounds = 200 if method == "fola" else 400

        return (
            f"debug_mnist_stage{source_stage}_cnn16_32_"
            f"fixed1c_s{samples}_n20_{method}_"
            f"seed0_e5_b{samples}_lr{lr_tag}"
            f"{lam}_r{rounds}"
        )

    rounds = TARGET[method]

    return (
        f"run_mnist_stage15_cnn16_32_"
        f"fixed1c_s{samples}_n20_{method}_"
        f"seed{seed}_e5_b{samples}_lr{lr_tag}"
        f"{lam}_r{rounds}"
    )


# ============================================================
# Read accuracy and communication metrics
# ============================================================

def get_accuracy(row, method):

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
            value = float(value)
        except (TypeError, ValueError):
            continue

        if math.isfinite(value):

            if value <= 1.000001:
                value *= 100

            return value

    raise ValueError(
        f"No usable accuracy field for {method}"
    )


def load_run(name, method, minimum_rounds):

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

    for directory in candidates:

        path = directory / "metrics/global_metrics.csv"

        with path.open(newline="", encoding="utf-8") as f:
            rows = {
                int(float(row["round"])): row
                for row in csv.DictReader(f)
                if row.get("round")
            }

        # Require completed original training.
        if not all(
            r in rows for r in range(1, minimum_rounds + 1)
        ):
            continue

        # Seed 0 was trained longer; truncate to Stage-15
        # comparison horizon: FOLA 150, FedAvg 300.
        target = TARGET[method]

        rounds = np.arange(1, target + 1)

        accuracy = np.array([
            get_accuracy(rows[r], method)
            for r in rounds
        ])

        communication = np.array([
            float(rows[r]["cumulative_all_array_bytes"])
            for r in rounds
        ])

        assert np.all(np.isfinite(accuracy))
        assert np.all(np.isfinite(communication))
        assert np.all(np.diff(communication) > 0)

        return {
            "rounds": rounds,
            "acc": accuracy,
            "bytes": communication,
            "gb": communication / 1e9,
            "directory": directory,
        }

    raise RuntimeError(
        f"Missing/incomplete run:\n{name}"
    )


def smooth(y, window=WINDOW):

    half = window // 2

    return np.array([
        np.mean(
            y[max(0, i-half):min(len(y), i+half+1)]
        )
        for i in range(len(y))
    ])


def save_figure(fig, name):

    global PLOT_COUNT

    base = OUT / name

    fig.savefig(
        str(base) + ".png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        str(base) + ".pdf",
        bbox_inches="tight",
    )

    plt.close(fig)
    PLOT_COUNT += 1


# ============================================================
# Load all 30 trajectories
# ============================================================

DATA = {}

print("\nLOADING STAGE-15 EXPERIMENTS")

for tag, samples, lr, source_stage in GROUPS:

    for seed in SEEDS:

        for method in ("fola", "fedavg"):

            name = run_name(
                samples, lr, source_stage,
                seed, method
            )

            required = (
                (200 if method == "fola" else 400)
                if seed == 0
                else TARGET[method]
            )

            DATA[(tag, seed, method)] = load_run(
                name, method, required
            )

            print(
                f"PASS | {tag:10s} | "
                f"seed={seed} | {method}"
            )

        # Validate equal communication at every checkpoint,
        # not only at the final round.
        f = DATA[(tag, seed, "fola")]
        a = DATA[(tag, seed, "fedavg")]

        assert np.allclose(
            f["bytes"],
            a["bytes"][1::2],
            rtol=1e-9,
            atol=1.0,
        ), f"Communication mismatch: {tag}, seed={seed}"

    # Check that all seeds share the same communication grid.
    for method in ("fola", "fedavg"):

        reference = DATA[(tag, 0, method)]["bytes"]

        for seed in SEEDS:
            assert np.allclose(
                reference,
                DATA[(tag, seed, method)]["bytes"],
                rtol=1e-9,
                atol=1.0,
            )

print("\nPASS: 30 trajectories loaded and budgets verified")


# ============================================================
# Plot helpers
# ============================================================

def plot_pair(ax, f, a, show_raw=True):

    for method, d in (("fola", f), ("fedavg", a)):

        x = d["gb"]
        y = d["acc"]

        if show_raw:
            ax.plot(
                x, y,
                color=COLORS[method],
                linestyle=STYLES[method],
                alpha=0.22,
                linewidth=0.75,
            )

        ax.plot(
            x,
            smooth(y),
            color=COLORS[method],
            linestyle=STYLES[method],
            linewidth=2.0,
            label=LABELS[method],
        )

        ax.scatter(
            [x[-1]], [y[-1]],
            color=COLORS[method],
            s=32,
            zorder=5,
        )

    ax.axvline(
        f["gb"][119],
        color="gray",
        linestyle=":",
        alpha=0.6,
        linewidth=1,
    )

    ax.grid(alpha=0.20)

    ax.set_xlabel("Cumulative communication (GB)")
    ax.set_ylabel("MNIST test accuracy (%)")


def matched_gap(f, a, rf):

    return (
        f["acc"][rf-1]
        - a["acc"][2*rf-1]
    )


def tail_gap(f, a, rf):

    ra = 2 * rf

    ft = np.mean(f["acc"][rf-20:rf])
    at = np.mean(a["acc"][ra-40:ra])

    return ft - at


# ============================================================
# 1. Individual plots: 3 conditions x 5 seeds = 15
#
# Each figure has:
# - Full communication trajectory
# - Late-communication zoom
# ============================================================

print("\nGENERATING INDIVIDUAL-SEED PLOTS")

for tag, samples, lr, _ in GROUPS:

    for seed in SEEDS:

        f = DATA[(tag, seed, "fola")]
        a = DATA[(tag, seed, "fedavg")]

        budget = f["gb"][-1]

        fig, axes = plt.subplots(
            1, 2, figsize=(13.5, 4.8)
        )

        for ax in axes:
            plot_pair(ax, f, a)

        axes[0].set_title("Full communication trajectory")
        axes[0].set_xlim(0, budget * 1.04)

        axes[1].set_title("Late communication: 50%-100%")
        axes[1].set_xlim(budget * 0.50, budget * 1.04)

        # Zoom y-axis using only the late-stage values.
        late = np.concatenate([
            f["acc"][74:],
            a["acc"][149:],
        ])

        ymin = max(0, late.min() - 3)
        ymax = min(100, late.max() + 3)

        axes[1].set_ylim(ymin, ymax)

        gap120 = matched_gap(f, a, 120)
        gap150 = matched_gap(f, a, 150)
        tgap = tail_gap(f, a, 150)

        fig.suptitle(
            f"CNN 16-32 | {samples} samples/client | "
            f"LR={lr:.2f} | Seed {seed}\n"
            f"R120/240 gap={gap120:+.2f} pp | "
            f"R150/300 gap={gap150:+.2f} pp | "
            f"Final-tail gap={tgap:+.2f} pp",
            fontsize=11,
        )

        axes[0].legend(
            loc="lower right",
            fontsize=9,
        )

        fig.tight_layout()

        name = f"acc_comm_cnn_{tag}_seed{seed}"

        save_figure(fig, name)

        print("SAVED:", name + ".png")


# ============================================================
# 2. Five-seed comparison panels: 3 figures
# ============================================================

print("\nGENERATING FIVE-SEED COMPARISON PANELS")

for tag, samples, lr, _ in GROUPS:

    fig, axes = plt.subplots(
        1, 5,
        figsize=(19, 4.3),
        sharex=True,
        sharey=True,
    )

    budget = DATA[(tag, 0, "fola")]["gb"][-1]

    for seed, ax in zip(SEEDS, axes):

        f = DATA[(tag, seed, "fola")]
        a = DATA[(tag, seed, "fedavg")]

        plot_pair(ax, f, a, show_raw=False)

        gap = matched_gap(f, a, 150)

        ax.set_title(
            f"Seed {seed}\n"
            f"Gap={gap:+.2f} pp",
            fontsize=10,
        )

        ax.set_xlim(0, budget * 1.04)

        if seed != 0:
            ax.set_ylabel("")

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.04),
    )

    fig.suptitle(
        f"CNN 16-32 | {samples} samples/client | "
        f"LR={lr:.2f} | Five seeds",
        y=1.12,
        fontsize=12,
    )

    fig.tight_layout()

    name = f"acc_comm_cnn_{tag}_five_seed_panels"

    save_figure(fig, name)

    print("SAVED:", name + ".png")


# ============================================================
# 3. Five-seed mean +/- 1 SD plots
#
# SD here is BETWEEN SEEDS, not across rounds.
# Calculated from 11-round-smoothed trajectories.
# ============================================================

def plot_five_seed_mean(ax, tag):

    for method in ("fola", "fedavg"):

        trajectories = []

        for seed in SEEDS:

            d = DATA[(tag, seed, method)]

            trajectories.append(
                smooth(d["acc"])
            )

        matrix = np.stack(trajectories)

        mu = matrix.mean(axis=0)
        sd = matrix.std(axis=0, ddof=1)

        x = DATA[(tag, 0, method)]["gb"]

        ax.plot(
            x, mu,
            color=COLORS[method],
            linestyle=STYLES[method],
            linewidth=2.1,
            label=LABELS[method],
        )

        ax.fill_between(
            x,
            mu-sd,
            mu+sd,
            color=COLORS[method],
            alpha=0.16,
        )

    budget120 = DATA[(tag, 0, "fola")]["gb"][119]
    budget150 = DATA[(tag, 0, "fola")]["gb"][149]

    ax.axvline(
        budget120,
        linestyle=":",
        color="gray",
        alpha=0.65,
    )

    ax.set_xlim(0, budget150 * 1.04)

    ax.grid(alpha=0.20)

    ax.set_xlabel("Cumulative communication (GB)")
    ax.set_ylabel("MNIST test accuracy (%)")

    return budget120, budget150


print("\nGENERATING FIVE-SEED MEAN +/- SD PLOTS")

for tag, samples, lr, _ in GROUPS:

    fig, ax = plt.subplots(
        figsize=(8.2, 5.2)
    )

    plot_five_seed_mean(ax, tag)

    gaps120 = []
    gaps150 = []
    tails150 = []

    for seed in SEEDS:

        f = DATA[(tag, seed, "fola")]
        a = DATA[(tag, seed, "fedavg")]

        gaps120.append(
            matched_gap(f, a, 120)
        )

        gaps150.append(
            matched_gap(f, a, 150)
        )

        tails150.append(
            tail_gap(f, a, 150)
        )

    ax.set_title(
        f"CNN 16-32 | {samples} samples/client | "
        f"LR={lr:.2f}\n"
        "Five-seed mean accuracy +/- 1 seed SD",
        fontsize=11,
    )

    ax.legend(loc="lower right")

    fig.tight_layout()

    name = f"acc_comm_cnn_{tag}_five_seed_mean_sd"

    save_figure(fig, name)

    print("SAVED:", name + ".png")

    print(
        f"{tag}: "
        f"Mean R120 gap={np.mean(gaps120):+.2f} pp | "
        f"Mean R150 gap={np.mean(gaps150):+.2f} pp | "
        f"Mean tail gap={np.mean(tails150):+.2f} pp"
    )


# ============================================================
# 4. Combined overview of all three conditions
# ============================================================

print("\nGENERATING COMBINED OVERVIEW")

fig, axes = plt.subplots(
    1, 3,
    figsize=(17, 4.8),
    sharex=True,
)

for ax, (tag, samples, lr, _) in zip(axes, GROUPS):

    plot_five_seed_mean(ax, tag)

    ax.set_title(
        f"{samples} samples/client | LR={lr:.2f}",
        fontsize=11,
    )

    ax.legend(loc="lower right", fontsize=8)

    if ax is not axes[0]:
        ax.set_ylabel("")

fig.suptitle(
    "Stage-15 CNN 16-32 | Five-seed accuracy "
    "vs communication cost",
    fontsize=13,
    y=1.02,
)

fig.tight_layout()

save_figure(
    fig,
    "acc_comm_cnn_stage15_three_conditions_overview",
)

print("SAVED: combined overview")


# ============================================================
# 5. Export summary CSV + terminal statistics
# ============================================================

summary_rows = []

print()
print("=" * 105)
print("STAGE-15 ACCURACY-COMMUNICATION SUMMARY")
print("=" * 105)

print(
    "Setting     Seed | "
    "F120   A240   Gap120 | "
    "F150   A300   Gap150 | "
    "TailGap150 | Budget GB"
)
print("-" * 105)

for tag, samples, lr, _ in GROUPS:

    for seed in SEEDS:

        f = DATA[(tag, seed, "fola")]
        a = DATA[(tag, seed, "fedavg")]

        row = {
            "setting": tag,
            "samples_per_client": samples,
            "learning_rate": lr,
            "seed": seed,

            "fola_r120": f["acc"][119],
            "fedavg_r240": a["acc"][239],
            "gap_r120_r240": matched_gap(f, a, 120),

            "fola_r150": f["acc"][149],
            "fedavg_r300": a["acc"][299],
            "gap_r150_r300": matched_gap(f, a, 150),

            "tail_gap_r150_r300": tail_gap(f, a, 150),

            "budget_gb": f["gb"][-1],
        }

        summary_rows.append(row)

        print(
            f"{tag:11s} {seed:4d} | "
            f"{row['fola_r120']:5.2f} "
            f"{row['fedavg_r240']:6.2f} "
            f"{row['gap_r120_r240']:+7.2f} | "
            f"{row['fola_r150']:5.2f} "
            f"{row['fedavg_r300']:6.2f} "
            f"{row['gap_r150_r300']:+7.2f} | "
            f"{row['tail_gap_r150_r300']:+10.2f} | "
            f"{row['budget_gb']:.3f}"
        )

    print("-" * 105)


csv_path = OUT / "stage15_accuracy_comm_summary.csv"

with csv_path.open(
    "w", newline="", encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(summary_rows[0]),
    )

    writer.writeheader()
    writer.writerows(summary_rows)


# ============================================================
# 6. Five-seed summary
# ============================================================

print()
print("=" * 95)
print("FIVE-SEED PAIRED ACCURACY GAPS")
print("Mean +/- sample SD across five seeds")
print("=" * 95)

for tag, samples, lr, _ in GROUPS:

    rows = [
        x for x in summary_rows
        if x["setting"] == tag
    ]

    for field, label in [
        ("gap_r120_r240", "R120 / R240"),
        ("gap_r150_r300", "R150 / R300"),
        ("tail_gap_r150_r300", "Final-tail"),
    ]:

        vals = np.array([
            row[field] for row in rows
        ])

        print(
            f"{tag:11s} | "
            f"{label:12s} | "
            f"{vals.mean():+7.2f} +/- "
            f"{vals.std(ddof=1):5.2f} pp"
        )


print()
print("=" * 95)
print("ALL PLOTS GENERATED SUCCESSFULLY")
print("=" * 95)

print("Individual seed plots     : 15")
print("Five-seed comparison grids: 3")
print("Five-seed mean +/- SD     : 3")
print("Three-condition overview  : 1")
print("Total figures             :", PLOT_COUNT)
print("Formats                   : PNG + PDF")
print("Output directory          :", OUT)
print("Summary CSV               :", csv_path)
print("PASS")
