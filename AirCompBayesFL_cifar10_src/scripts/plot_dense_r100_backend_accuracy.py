from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

RESULTS = Path("results")
OUTPUT = Path("results/plot_dense_r100_backend_accuracy")
OUTPUT.mkdir(parents=True, exist_ok=True)

SEEDS = [12025, 12026, 12027]

PYRO_PATTERN = (
    "dense_priority_pyro_r100_seed{seed}"
)

BT_PATTERN = (
    "dense_priority_bayesian_torch_r100_seed{seed}"
)

# FedAvg seed12025 is known to exist.
# If 12026 and 12027 also exist, the script averages all three.
FEDAVG_PATTERN = (
    "dense_priority_fedavg_r100_seed{seed}"
)


# ============================================================
# Helpers
# ============================================================

def read_metric(
    directory_pattern,
    seeds,
    metric,
):
    frames = []

    for seed in seeds:
        path = (
            RESULTS
            / directory_pattern.format(seed=seed)
            / "metrics.csv"
        )

        if not path.exists():
            continue

        df = (
            pd.read_csv(path)
            .sort_values("round")
        )

        if metric not in df.columns:
            raise KeyError(
                f"{path} does not contain "
                f"{metric!r}"
            )

        temp = df[
            ["round", metric]
        ].copy()

        temp[metric] = pd.to_numeric(
            temp[metric],
            errors="coerce",
        )

        temp["seed"] = seed

        frames.append(temp)

    if not frames:
        raise FileNotFoundError(
            f"No runs found for pattern "
            f"{directory_pattern}"
        )

    data = pd.concat(
        frames,
        ignore_index=True,
    )

    summary = (
        data
        .groupby("round")[metric]
        .agg(
            mean="mean",
            std="std",
            count="count",
        )
        .reset_index()
    )

    # std is NaN when only one seed exists.
    summary["std"] = (
        summary["std"]
        .fillna(0.0)
    )

    return (
        data,
        summary,
        sorted(data["seed"].unique()),
    )


# ============================================================
# Load Bayesian methods
# ============================================================

pyro_raw, pyro, pyro_seeds = read_metric(
    PYRO_PATTERN,
    SEEDS,
    "posterior_mean_accuracy",
)

bt_raw, bt, bt_seeds = read_metric(
    BT_PATTERN,
    SEEDS,
    "posterior_mean_accuracy",
)


# ============================================================
# Load FedAvg
# ============================================================

fedavg_raw, fedavg, fedavg_seeds = read_metric(
    FEDAVG_PATTERN,
    SEEDS,
    "accuracy",
)


# ============================================================
# Validate Bayesian replication count
# ============================================================

if len(pyro_seeds) != 3:
    raise RuntimeError(
        "Expected 3 Pyro seeds, found "
        f"{pyro_seeds}"
    )

if len(bt_seeds) != 3:
    raise RuntimeError(
        "Expected 3 Bayesian-Torch seeds, found "
        f"{bt_seeds}"
    )


# ============================================================
# Build combined summary CSV
# ============================================================

def formatted_summary(
    summary,
    method,
    metric,
):
    result = summary.copy()

    result.insert(
        0,
        "method",
        method,
    )

    result.insert(
        1,
        "metric",
        metric,
    )

    return result


combined = pd.concat(
    [
        formatted_summary(
            fedavg,
            "FedAvg",
            "accuracy",
        ),
        formatted_summary(
            pyro,
            "Proposed/Pyro",
            "posterior_mean_accuracy",
        ),
        formatted_summary(
            bt,
            "Proposed/Bayesian-Torch",
            "posterior_mean_accuracy",
        ),
    ],
    ignore_index=True,
)

combined.to_csv(
    OUTPUT / "accuracy_round_summary.csv",
    index=False,
)


# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(
    figsize=(8.5, 5.5)
)


# ------------------------------------------------------------
# FedAvg
# ------------------------------------------------------------

fed_label = (
    f"FedAvg mean ({len(fedavg_seeds)} seeds)"
    if len(fedavg_seeds) > 1
    else f"FedAvg reference (seed {fedavg_seeds[0]})"
)

fed_line = ax.plot(
    fedavg["round"],
    100.0 * fedavg["mean"],
    linewidth=2.2,
    linestyle="--",
    label=fed_label,
)[0]

if len(fedavg_seeds) > 1:
    ax.fill_between(
        fedavg["round"],
        100.0 * (
            fedavg["mean"]
            - fedavg["std"]
        ),
        100.0 * (
            fedavg["mean"]
            + fedavg["std"]
        ),
        alpha=0.15,
        color=fed_line.get_color(),
    )


# ------------------------------------------------------------
# Pyro
# ------------------------------------------------------------

pyro_line = ax.plot(
    pyro["round"],
    100.0 * pyro["mean"],
    linewidth=2.2,
    label="Proposed/Pyro mean (3 seeds)",
)[0]

ax.fill_between(
    pyro["round"],
    100.0 * (
        pyro["mean"]
        - pyro["std"]
    ),
    100.0 * (
        pyro["mean"]
        + pyro["std"]
    ),
    alpha=0.15,
    color=pyro_line.get_color(),
)


# ------------------------------------------------------------
# Bayesian-Torch
# ------------------------------------------------------------

bt_line = ax.plot(
    bt["round"],
    100.0 * bt["mean"],
    linewidth=2.2,
    label="Proposed/Bayesian-Torch mean (3 seeds)",
)[0]

ax.fill_between(
    bt["round"],
    100.0 * (
        bt["mean"]
        - bt["std"]
    ),
    100.0 * (
        bt["mean"]
        + bt["std"]
    ),
    alpha=0.15,
    color=bt_line.get_color(),
)


# ============================================================
# Appearance
# ============================================================

ax.set_xlabel(
    "Logical Round",
    fontsize=12,
)

ax.set_ylabel(
    "Test Accuracy (%)",
    fontsize=12,
)

ax.set_title(
    "CIFAR-10 Dense Baseline: Accuracy vs. Logical Round",
    fontsize=13,
)

ax.set_xlim(
    0,
    100,
)

ax.grid(
    True,
    alpha=0.25,
)

ax.legend(
    fontsize=9,
    loc="lower right",
)

fig.tight_layout()


# ============================================================
# Save
# ============================================================

png_path = (
    OUTPUT
    / "dense_r100_accuracy_vs_round.png"
)

pdf_path = (
    OUTPUT
    / "dense_r100_accuracy_vs_round.pdf"
)

fig.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    pdf_path,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Numerical summary
# ============================================================

print()
print("=" * 85)
print("DENSE R100 ACCURACY-vs-ROUND SUMMARY")
print("=" * 85)

print()
print("Pyro seeds          :", pyro_seeds)
print("Bayesian-Torch seeds:", bt_seeds)
print("FedAvg seeds        :", fedavg_seeds)


def print_stats(
    name,
    summary,
):
    late = summary[
        (summary["round"] >= 80)
        & (summary["round"] <= 100)
    ]

    best_idx = (
        summary["mean"]
        .idxmax()
    )

    best = summary.loc[
        best_idx
    ]

    final = summary.iloc[-1]

    print()
    print(name)

    print(
        "  best mean accuracy : "
        f"{100 * best['mean']:.4f}% "
        f"(round {int(best['round'])})"
    )

    print(
        "  late mean R80-100  : "
        f"{100 * late['mean'].mean():.4f}%"
    )

    print(
        "  final mean accuracy: "
        f"{100 * final['mean']:.4f}%"
    )

    if int(final["count"]) > 1:
        print(
            "  final seed std     : "
            f"{100 * final['std']:.4f} pp"
        )


print_stats(
    "FedAvg",
    fedavg,
)

print_stats(
    "Proposed/Pyro",
    pyro,
)

print_stats(
    "Proposed/Bayesian-Torch",
    bt,
)

print()
print("Saved:")
print(" ", png_path)
print(" ", pdf_path)
print(
    " ",
    OUTPUT / "accuracy_round_summary.csv",
)
