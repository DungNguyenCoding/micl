from pathlib import Path
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# Paths
# ============================================================

ROOT = Path("outputs").resolve()
OUT = Path("plots")
OUT.mkdir(parents=True, exist_ok=True)

COMM_PDF = OUT / "fig_accuracy_vs_communication_multicast.pdf"
COMM_PNG = OUT / "fig_accuracy_vs_communication_multicast.png"

KEEP_PDF = OUT / "fig_accuracy_vs_keep_ratio_multicast.pdf"
KEEP_PNG = OUT / "fig_accuracy_vs_keep_ratio_multicast.png"


# ============================================================
# Plot style
# ============================================================

RED = "#C62828"
GREY = "#6E6E6E"

# Distinct colors for KL sparse keep ratios
KEEP_COLORS = {
    0.75: "#1f77b4",  # blue
    0.50: "#ff7f0e",  # orange
    0.25: "#2ca02c",  # green
    0.10: "#9467bd",  # purple
}

BLUE = "#1565C0"
GREEN = "#2E7D32"
GRID = "#D5D5D5"

KEEP_ORDER = [
    0.75,
    0.50,
    0.25,
    0.10,
]

SMOOTH_WINDOW = 7  # mild smoothing for paper readability

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "DejaVu Serif",
    ],
    "mathtext.fontset": "stix",

    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0,

    "axes.linewidth": 0.75,
    "lines.linewidth": 1.45,

    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


# ============================================================
# Experiment definitions
# ============================================================

PROFILES = {
    "MNIST": {
        "short_title": "MNIST",
        "d": 545_810,
        "K": 100,
        "budget": 52_921_737_600,
        "dense_target": 120,
        "fedavg_target": 240,

        "dense": (
            "run_confirm_newsrc_"
            "mnist_fixed1c_s10_fola_n100_seed0_"
            "e10_b32_lr001_lam0p01_r150"
        ),
        "fedavg": (
            "run_budgetmatch_"
            "mnist_fixed1c_s10_fedavg_n100_seed0_"
            "e10_b32_lr001_budget150densefola"
        ),
        "sparse_prefix": "run_sparse_mnist_fixed1c_s10_fola_",
        "sparse_suffix": "_n100_seed0_e10_b32_lr001_lam0p01_budget150dense",

        "targets": {
            0.75: 156,
            0.50: 230,
            0.25: 439,
            0.10: 964,
        },
        "tags": {
            0.75: "0p75",
            0.50: "0p5",
            0.25: "0p25",
            0.10: "0p1",
        },
    },

    "CIFAR-10": {
        "short_title": "CIFAR-10",
        "d": 878_538,
        "K": 20,
        "budget": 22_139_157_600,
        "dense_target": 150,
        "fedavg_target": 300,

        "dense": (
            "run_confirm_newsrc_"
            "cifar_basiccnn_a0p01_fola_cos400_n20_f1_"
            "seed0_e10_b32_lr002_lam0p01_r150"
        ),
        "fedavg": (
            "run_budgetmatch_"
            "cifar_basiccnn_a0p01_fedavg_n20_f1_"
            "seed0_e10_b32_lr001_budget150densefola"
        ),
        "sparse_prefix": "run_sparse_cifar_basiccnn_a0p01_f1_fola_",
        "sparse_suffix": "_cos400_n20_seed0_e10_b32_lr002_lam0p01_budget150dense",

        "targets": {
            0.75: 193,
            0.50: 278,
            0.25: 499,
            0.10: 950,
        },
        "tags": {
            0.75: "0p75",
            0.50: "0p5",
            0.25: "0p25",
            0.10: "0p1",
        },
    },
}


# ============================================================
# Helpers
# ============================================================

def resolve(prefix: str) -> Path:
    exact = ROOT / prefix

    matches = []
    if exact.is_dir():
        matches.append(exact)

    matches.extend(
        p for p in ROOT.glob(prefix + "_*")
        if p.is_dir()
    )

    if not matches:
        raise RuntimeError(
            f"Could not find output directory for:\n{prefix}"
        )

    return max(set(matches), key=lambda p: p.stat().st_mtime)


def load_metrics(run_dir: Path):
    path = run_dir / "metrics" / "global_metrics.csv"

    if not path.exists():
        raise RuntimeError(f"Missing global_metrics.csv:\n{path}")

    with path.open(newline="", encoding="utf-8") as f:
        raw = list(csv.DictReader(f))

    by_round = {}
    for row in raw:
        try:
            rnd = int(float(row["round"]))
        except (KeyError, ValueError):
            continue

        if rnd >= 1:
            by_round[rnd] = row

    return [by_round[r] for r in sorted(by_round)]


def accuracy(row) -> float:
    preferred = [
        "global_test_accuracy",
        "global_accuracy",
        "centralized_accuracy",
        "centralized_acc",
        "mean_accuracy",
        "mean_acc",
        "fola_mean_accuracy",
        "accuracy",
    ]

    for key in preferred:
        value = row.get(key)
        if value not in (None, ""):
            x = float(value)
            return 100.0 * x if x <= 1.5 else x

    for key, value in row.items():
        low = key.lower()
        if (
            value not in (None, "")
            and "local" not in low
            and ("accuracy" in low or low.endswith("_acc"))
        ):
            try:
                x = float(value)
            except ValueError:
                continue
            return 100.0 * x if x <= 1.5 else x

    raise RuntimeError(f"Could not find accuracy column in {list(row.keys())}")


def communication_bytes(row) -> int:
    return int(float(row["cumulative_all_array_bytes"]))


def row_at(rows, target_round: int):
    for row in rows:
        if int(float(row["round"])) == target_round:
            return row
    final_round = max(int(float(x["round"])) for x in rows)
    raise RuntimeError(
        f"Required R{target_round} does not exist. "
        f"Trajectory currently ends at R{final_round}."
    )


def rows_through(rows, target_round: int):
    selected = [r for r in rows if int(float(r["round"])) <= target_round]
    row_at(selected, target_round)
    return selected


def sparse_run(profile, rule: str, keep: float):
    tag = profile["tags"][keep]
    name = (
        profile["sparse_prefix"]
        + rule
        + "_keep"
        + tag
        + profile["sparse_suffix"]
    )
    return resolve(name)


def validate_sparse_endpoint(profile, keep, row):
    d = profile["d"]
    K = profile["K"]
    budget = profile["budget"]
    target = profile["targets"][keep]

    m = math.ceil(keep * d)
    per_round = 8 * d + K * (8 * m + math.ceil(d / 8))
    expected = target * per_round
    actual = communication_bytes(row)

    if actual != expected:
        raise RuntimeError(
            f"Communication mismatch for keep={keep}: "
            f"actual={actual:,}, expected={expected:,}"
        )

    if expected > budget:
        raise RuntimeError(f"Target R{target} exceeds budget.")

    if (target + 1) * per_round <= budget:
        raise RuntimeError(
            f"R{target} is not the maximum round within budget."
        )


def moving_average(y, window=7):
    if len(y) <= 2 or window <= 1:
        return list(y)

    window = min(window, len(y))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return list(y)

    pad = window // 2
    yy = [y[0]] * pad + list(y) + [y[-1]] * pad
    out = []

    for i in range(len(y)):
        s = 0.0
        for j in range(window):
            s += yy[i + j]
        out.append(s / window)

    return out


# ============================================================
# Load and validate data
# ============================================================

DATA = {}

print("=" * 100)
print("PLOT DATA PREFLIGHT")
print("=" * 100)

for label, p in PROFILES.items():
    DATA[label] = {}

    dense_dir = resolve(p["dense"])
    dense_rows = load_metrics(dense_dir)
    dense_rows = rows_through(dense_rows, p["dense_target"])
    dense_final = row_at(dense_rows, p["dense_target"])

    dense_expected = p["dense_target"] * 8 * p["d"] * (p["K"] + 1)
    if communication_bytes(dense_final) != dense_expected:
        raise RuntimeError(f"{label}: dense FOLA communication mismatch.")
    if dense_expected != p["budget"]:
        raise RuntimeError(f"{label}: dense budget mismatch.")

    DATA[label]["dense"] = dense_rows

    fedavg_dir = resolve(p["fedavg"])
    fedavg_rows = load_metrics(fedavg_dir)
    fedavg_final = row_at(fedavg_rows, p["fedavg_target"])

    fedavg_expected = p["fedavg_target"] * 4 * p["d"] * (p["K"] + 1)
    if communication_bytes(fedavg_final) != fedavg_expected:
        raise RuntimeError(f"{label}: FedAvg communication mismatch.")
    if fedavg_expected != p["budget"]:
        raise RuntimeError(f"{label}: FedAvg budget mismatch.")

    DATA[label]["fedavg_final"] = fedavg_final
    DATA[label]["kl"] = {}
    DATA[label]["random_final"] = {}

    for keep in KEEP_ORDER:
        target = p["targets"][keep]

        kl_dir = sparse_run(p, "kl_global_local", keep)
        kl_rows = load_metrics(kl_dir)
        kl_rows = rows_through(kl_rows, target)
        kl_final = row_at(kl_rows, target)
        validate_sparse_endpoint(p, keep, kl_final)
        DATA[label]["kl"][keep] = kl_rows

        rand_dir = sparse_run(p, "random", keep)
        rand_rows = load_metrics(rand_dir)
        rand_final = row_at(rand_rows, target)
        validate_sparse_endpoint(p, keep, rand_final)
        DATA[label]["random_final"][keep] = rand_final

    print()
    print(label)
    print(f"  Dense FOLA R120: {accuracy(dense_final):.3f}%")
    print(f"  Dense FedAvg R240: {accuracy(fedavg_final):.3f}%")
    for keep in KEEP_ORDER:
        kl_final = row_at(DATA[label]["kl"][keep], p["targets"][keep])
        rand_final = DATA[label]["random_final"][keep]
        print(
            f"  keep={keep:.2f} R{p['targets'][keep]:4d} | "
            f"KL={accuracy(kl_final):7.3f}% | "
            f"Random={accuracy(rand_final):7.3f}% | "
            f"comm={communication_bytes(kl_final)/1e9:9.6f} GB"
        )

print()
print("PASS: all required checkpoints are present and valid.")


# ============================================================
# Figure 1: Accuracy vs communication cost (1x2)
# ============================================================

fig, axes = plt.subplots(
    1, 2,
    figsize=(7.2, 3.2),
    constrained_layout=False,
)

fig.subplots_adjust(
    left=0.085,
    right=0.995,
    top=0.90,
    bottom=0.28,
    wspace=0.20,
)

for ax, (label, p) in zip(axes, PROFILES.items()):
    max_comm_gb = p["budget"] / 1e9

    # Dense FedAvg final horizontal reference
    fedavg_acc = accuracy(DATA[label]["fedavg_final"])
    ax.axhline(
        fedavg_acc,
        color=GREY,
        linestyle=(0, (5, 3)),
        linewidth=1.25,
        zorder=1,
        label="Dense FedAvg (final)",
    )

    # Dense FOLA raw + smoothed
    dense_rows = DATA[label]["dense"]
    x_dense = [communication_bytes(r) / 1e9 for r in dense_rows]
    y_dense = [accuracy(r) for r in dense_rows]
    y_dense_sm = moving_average(y_dense, SMOOTH_WINDOW)

    ax.plot(
        x_dense, y_dense,
        color=RED,
        linewidth=0.8,
        alpha=0.20,
        zorder=2,
    )
    ax.plot(
        x_dense, y_dense_sm,
        color=RED,
        linewidth=1.75,
        label="Dense FOLA",
        zorder=5,
    )

    # KL sparse curves with different colors
    for keep in KEEP_ORDER:
        rows = DATA[label]["kl"][keep]
        x = [communication_bytes(r) / 1e9 for r in rows]
        y = [accuracy(r) for r in rows]
        y_sm = moving_average(y, SMOOTH_WINDOW)

        color = KEEP_COLORS[keep]
        marker = {
            0.75: "o",
            0.50: "s",
            0.25: "^",
            0.10: "D",
        }[keep]

        mark_every = max(1, len(rows) // 12)

        ax.plot(
            x, y,
            color=color,
            linewidth=0.7,
            alpha=0.18,
            zorder=2,
        )

        ax.plot(
            x, y_sm,
            color=color,
            linewidth=1.35,
            marker=marker,
            markersize=3.0,
            markerfacecolor="white",
            markeredgewidth=0.75,
            markevery=mark_every,
            label=f"KL sparse ({int(keep*100)}%)",
            zorder=4,
        )

    ax.set_xlim(0, max_comm_gb * 1.015)
    ax.set_title(p["short_title"], pad=4, fontweight="bold")
    ax.set_xlabel("Communication cost (GB)")
    ax.set_ylabel("Global test accuracy (%)")

    ax.grid(
        True,
        which="major",
        color=GRID,
        linewidth=0.45,
        linestyle=":",
        alpha=0.8,
    )
    ax.tick_params(top=True, right=True)

# shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.02),
    ncol=3,
    frameon=False,
    columnspacing=1.0,
    handlelength=2.8,
    handletextpad=0.5,
)

fig.savefig(COMM_PDF)
fig.savefig(COMM_PNG, dpi=600)
plt.close(fig)


# ============================================================
# Figure 2: Accuracy vs keep ratio (1x2)
# ============================================================

fig, axes = plt.subplots(
    1, 2,
    figsize=(7.2, 3.0),
    constrained_layout=False,
)

fig.subplots_adjust(
    left=0.085,
    right=0.995,
    top=0.90,
    bottom=0.26,
    wspace=0.20,
)

for ax, (label, p) in zip(axes, PROFILES.items()):
    dense_final = row_at(DATA[label]["dense"], p["dense_target"])
    dense_acc = accuracy(dense_final)

    ax.axhline(
        dense_acc,
        color=RED,
        linestyle=(0, (6, 3)),
        linewidth=1.5,
        label="Dense FOLA (100%)",
        zorder=2,
    )
    ax.plot(
        [100], [dense_acc],
        color=RED,
        marker="o",
        markersize=4.2,
        markerfacecolor="white",
        markeredgewidth=1.0,
        linestyle="None",
        zorder=5,
    )

    x_keep = [100 * k for k in KEEP_ORDER]
    kl_acc = []
    rand_acc = []

    for keep in KEEP_ORDER:
        kl_row = row_at(DATA[label]["kl"][keep], p["targets"][keep])
        rand_row = DATA[label]["random_final"][keep]
        kl_acc.append(accuracy(kl_row))
        rand_acc.append(accuracy(rand_row))

    ax.plot(
        x_keep,
        kl_acc,
        color=BLUE,
        linestyle="-",
        linewidth=1.65,
        marker="o",
        markersize=4.0,
        markerfacecolor="white",
        markeredgewidth=1.0,
        label="KL-score sparse",
        zorder=4,
    )

    ax.plot(
        x_keep,
        rand_acc,
        color=GREEN,
        linestyle="-.",
        linewidth=1.45,
        marker="s",
        markersize=3.8,
        markerfacecolor="white",
        markeredgewidth=1.0,
        label="Random sparse",
        zorder=3,
    )

    # Use only actual observed keep-ratio range:
    # 100, 75, 50, 25, 10
    ax.set_xlim(102, 8)
    ax.set_xticks([100, 75, 50, 25, 10])

    ax.set_title(p["short_title"], pad=4, fontweight="bold")
    ax.set_xlabel("Keep ratio (%)")
    ax.set_ylabel("Global test accuracy (%)")

    ax.grid(
        True,
        which="major",
        color=GRID,
        linewidth=0.45,
        linestyle=":",
        alpha=0.8,
    )
    ax.tick_params(top=True, right=True)

handles = [
    Line2D([0], [0], color=RED, linestyle=(0, (6, 3)), linewidth=1.5,
           label="Dense FOLA (100%)"),
    Line2D([0], [0], color=BLUE, linestyle="-", marker="o",
           markerfacecolor="white", linewidth=1.65,
           label="KL-score sparse"),
    Line2D([0], [0], color=GREEN, linestyle="-.", marker="s",
           markerfacecolor="white", linewidth=1.45,
           label="Random sparse"),
]

fig.legend(
    handles=handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.02),
    ncol=3,
    frameon=False,
    columnspacing=1.0,
    handlelength=2.8,
    handletextpad=0.5,
)

fig.savefig(KEEP_PDF)
fig.savefig(KEEP_PNG, dpi=600)
plt.close(fig)


# ============================================================
# Terminal summary
# ============================================================

print()
print("=" * 100)
print("FIGURES GENERATED")
print("=" * 100)
print(f"1) {COMM_PDF.resolve()}")
print(f"2) {COMM_PNG.resolve()}")
print(f"3) {KEEP_PDF.resolve()}")
print(f"4) {KEEP_PNG.resolve()}")
print()
print("Notes:")
print("- Both figures use 1x2 layout.")
print("- Titles are only MNIST and CIFAR-10.")
print("- Accuracy vs communication uses faint raw curves + smoothed curves.")
print("- keep=0.05 is ignored.")
print("- Keep-ratio x-axis uses 100, 75, 50, 25, 10.")
print("PASS")
