from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

TARGET_ROUND = 160

SPARSE_METRICS = Path(
    r"results\sparse_proposed_rep3\metrics.csv"
)

DENSE_DIRS = [
    Path(r"results\fig2_proposed_rep1"),
    Path(r"results\sparse_dense_extra_12026_12027"),
]

OUTPUT_DIR = Path(
    r"results\sparse_proposed_rep3\plots"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PNG = OUTPUT_DIR / "final_accuracy_vs_keep_ratio_rep3.png"
OUTPUT_CSV = OUTPUT_DIR / "final_accuracy_vs_keep_ratio_rep3.csv"


# ============================================================
# Helpers
# ============================================================

def get_selection_column(df):
    for name in [
        "sparse_selection",
        "sparse_selection_method",
    ]:
        if name in df.columns:
            return name

    raise KeyError(
        "Cannot find sparse selection column."
    )


def get_keep_column(df):
    for name in [
        "sparse_keep_ratio",
        "sparse_ratio",
    ]:
        if name in df.columns:
            return name

    raise KeyError(
        "Cannot find sparse keep-ratio column."
    )


# ============================================================
# 1. Read sparse runs
# ============================================================

df = pd.read_csv(SPARSE_METRICS)

df["round"] = pd.to_numeric(
    df["round"],
    errors="coerce",
)

df["accuracy"] = pd.to_numeric(
    df["accuracy"],
    errors="coerce",
)

selection_col = get_selection_column(df)
keep_col = get_keep_column(df)

df[keep_col] = pd.to_numeric(
    df[keep_col],
    errors="coerce",
)

# We want final accuracy exactly at round 160.
final_sparse = df[
    (df["method"].astype(str).str.lower() == "proposed")
    & (df["round"] == TARGET_ROUND)
].copy()

final_sparse["selection"] = (
    final_sparse[selection_col]
    .astype(str)
    .str.lower()
)

final_sparse["keep_percent"] = (
    final_sparse[keep_col] * 100.0
)


# ============================================================
# 2. Average sparse accuracy over the 3 seeds
# ============================================================

summary = (
    final_sparse
    .groupby(
        ["selection", "keep_percent"],
        as_index=False,
    )
    .agg(
        n_seeds=("accuracy", "count"),
        final_accuracy_mean=("accuracy", "mean"),
        final_accuracy_std=("accuracy", "std"),
    )
)

summary = summary.sort_values(
    ["selection", "keep_percent"]
)


# ============================================================
# 3. Dense 100% baseline
#
# For every seed:
#   find the BEST accuracy over rounds <= 160.
#
# Then average those three per-seed best accuracies.
# ============================================================

dense_results = []

for dense_dir in DENSE_DIRS:

    dense_path = dense_dir / "metrics.csv"

    if not dense_path.exists():
        raise FileNotFoundError(
            f"Missing dense metrics: {dense_path}"
        )

    dense = pd.read_csv(dense_path)

    dense["round"] = pd.to_numeric(
        dense["round"],
        errors="coerce",
    )

    dense["accuracy"] = pd.to_numeric(
        dense["accuracy"],
        errors="coerce",
    )

    dense["seed"] = pd.to_numeric(
        dense["seed"],
        errors="coerce",
    )

    dense = dense[
        (dense["method"].astype(str).str.lower() == "proposed")
        & (dense["round"] <= TARGET_ROUND)
        & (dense["seed"].isin([12025, 12026, 12027]))
    ].copy()

    for run_id, group in dense.groupby("run_id"):

        best_row = group.loc[
            group["accuracy"].idxmax()
        ]

        dense_results.append(
            {
                "run_id": run_id,
                "seed": int(best_row["seed"]),
                "best_round": int(best_row["round"]),
                "best_accuracy": float(best_row["accuracy"]),
            }
        )


dense_df = pd.DataFrame(dense_results)

# Protect against duplicates if a run appears in both folders.
dense_df = (
    dense_df
    .sort_values(["seed", "best_accuracy"])
    .drop_duplicates(
        subset=["seed"],
        keep="last",
    )
    .sort_values("seed")
)

expected_seeds = {12025, 12026, 12027}
found_seeds = set(dense_df["seed"].tolist())

if found_seeds != expected_seeds:
    raise RuntimeError(
        f"Dense baseline seeds mismatch. "
        f"Expected {expected_seeds}, found {found_seeds}"
    )

dense_best_mean = dense_df["best_accuracy"].mean()
dense_best_std = dense_df["best_accuracy"].std(ddof=1)


# ============================================================
# 4. Print numerical results
# ============================================================

print()
print("=" * 72)
print("Sparse final accuracy: 3-seed average")
print("=" * 72)

display_summary = summary.copy()

display_summary["keep_percent"] = (
    display_summary["keep_percent"]
    .round(0)
    .astype(int)
)

display_summary["final_accuracy_mean"] = (
    display_summary["final_accuracy_mean"].round(4)
)

display_summary["final_accuracy_std"] = (
    display_summary["final_accuracy_std"].round(4)
)

print(
    display_summary.to_string(index=False)
)

print()
print("=" * 72)
print("Dense Proposed keep=100% baseline")
print("=" * 72)

print(
    dense_df.to_string(index=False)
)

print()
print(
    f"Dense 100% average BEST accuracy: "
    f"{dense_best_mean:.4f} +/- {dense_best_std:.4f}"
)


# ============================================================
# 5. Save combined numerical CSV
# ============================================================

csv_rows = []

for _, row in summary.iterrows():

    csv_rows.append(
        {
            "selection": row["selection"],
            "keep_percent": row["keep_percent"],
            "n_seeds": int(row["n_seeds"]),
            "accuracy_mean": row["final_accuracy_mean"],
            "accuracy_std": row["final_accuracy_std"],
            "metric": "final_accuracy",
        }
    )

# One shared dense row -- NOT duplicated as Bayesian and Random.
csv_rows.append(
    {
        "selection": "dense_shared",
        "keep_percent": 100.0,
        "n_seeds": 3,
        "accuracy_mean": dense_best_mean,
        "accuracy_std": dense_best_std,
        "metric": "best_accuracy",
    }
)

pd.DataFrame(csv_rows).to_csv(
    OUTPUT_CSV,
    index=False,
)


# ============================================================
# 6. Plot
# ============================================================

fig, ax = plt.subplots(
    figsize=(8.4, 5.4)
)

# Sparse methods
for selection in ["bayesian", "random"]:

    sub = summary[
        summary["selection"] == selection
    ].sort_values("keep_percent")

    if sub.empty:
        continue

    x = sub["keep_percent"].to_numpy()
    y = sub["final_accuracy_mean"].to_numpy()
    yerr = sub["final_accuracy_std"].to_numpy()

    label = (
        "Bayesian sparse"
        if selection == "bayesian"
        else "Random sparse"
    )

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        linewidth=2,
        capsize=4,
        label=label,
    )


# ------------------------------------------------------------
# Shared keep-100 dense baseline
#
# Draw ONE grey dashed horizontal line.
# ------------------------------------------------------------

ax.axhline(
    dense_best_mean,
    linestyle="--",
    linewidth=2,
    color="grey",
    label=(
        f"Dense Proposed 100% "
        f"(best={dense_best_mean:.4f})"
    ),
)

# Optional ±1 std region around dense baseline.
ax.axhspan(
    dense_best_mean - dense_best_std,
    dense_best_mean + dense_best_std,
    color="grey",
    alpha=0.10,
)


# ------------------------------------------------------------
# Formatting
# ------------------------------------------------------------

ax.set_xlabel(
    "Keep ratio (%)",
    fontsize=12,
)

ax.set_ylabel(
    "Test accuracy",
    fontsize=12,
)

ax.set_title(
    "Bayesian vs Random Sparse Posterior Communication\n"
    "3-seed average, 160 logical rounds",
    fontsize=13,
)

ax.set_xticks(
    [2, 5, 10, 25, 50, 75, 100]
)

ax.set_xlim(
    0,
    102,
)

ax.set_ylim(
    0.0,
    1.0,
)

ax.grid(
    True,
    alpha=0.25,
)

ax.legend(
    loc="lower right"
)

fig.tight_layout()

fig.savefig(
    OUTPUT_PNG,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


print()
print("=" * 72)
print("Saved")
print("=" * 72)
print(OUTPUT_PNG)
print(OUTPUT_CSV)