from pathlib import Path
import csv
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bayesfl.config import load_config
from bayesfl.data.datasets import partition_stem

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

PLOT_DIR = ROOT / "plots/cnn_selected_unclipped_comparison"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Seed numbers are ONLY used to locate the correct runs.
# They are NEVER displayed inside the generated figures.

CASES = {
    5: {
        "seed": 3,
        "fola": (
            "run_mnist_stage18_cnn16_32_fola_noclip_"
            "fixed1c_s5_n20_seed3_e5_b5_lr0p10_lam0p01_r200"
        ),
        "unclipped": (
            "run_stage16_paper00_fedavg_cnn16_32_"
            "s5_n20_seed3_e5_b5_lr0p10_r400"
        ),
        # Original R300 run, extended to R400 in Stage 19.
        "clipped": (
            "run_mnist_stage15_cnn16_32_fixed1c_s5_n20_"
            "fedavg_seed3_e5_b5_lr0p10_r300"
        ),
    },
    10: {
        "seed": 0,
        "fola": (
            "run_mnist_stage18_cnn16_32_fola_noclip_"
            "fixed1c_s10_n20_seed0_e5_b10_lr0p10_lam0p01_r200"
        ),
        "unclipped": (
            "run_stage16_paper00_fedavg_cnn16_32_"
            "s10_n20_seed0_e5_b10_lr0p10_r400"
        ),
        "clipped": (
            "debug_mnist_stage12_cnn16_32_fixed1c_s10_n20_"
            "fedavg_seed0_e5_b10_lr0p10_r400"
        ),
    },
}

MILESTONES = (120, 150, 200)

LABELS = {
    "fola": "Dense FOLA (unclipped)",
    "unclipped": "FedAvg (unclipped)",
    "clipped": "FedAvg (clipped at 10)",
}

COLORS = {
    "fola": "tab:blue",
    "unclipped": "tab:orange",
    "clipped": "tab:green",
}

STYLES = {
    "fola": "-",
    "unclipped": "--",
    "clipped": "-.",
}


# ============================================================
# 1. LOAD ACCURACY AND COMMUNICATION COST
# ============================================================

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
        if row.get(key) in (None, ""):
            continue

        value = float(row[key])

        if math.isfinite(value):
            return (
                100.0 * value
                if value <= 1.000001
                else value
            )

    raise ValueError(
        f"No valid accuracy field: {list(row)}"
    )


def load_run(name, key, samples, seed):
    target = 200 if key == "fola" else 400

    candidates = [
        OUTPUTS / name,
        *OUTPUTS.glob(name + "_*"),
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

    if not candidates:
        raise FileNotFoundError(
            f"No output directory for {name}"
        )

    for run in candidates:
        resolved = run / "resolved_config.yaml"

        if not resolved.is_file():
            continue

        cfg = load_config(resolved)

        expected_method = (
            "fola" if key == "fola" else "fedavg"
        )

        if (
            cfg.run_name != name
            or cfg.method != expected_method
        ):
            continue

        assert cfg.model.name == "mnist_cnn_16_32_128"
        assert cfg.runtime.seed == seed

        assert cfg.data.partition == {
            "type": "fixed_labels",
            "samples_per_client": samples,
            "labels_per_client": 1,
        }

        assert cfg.federation.num_clients == 20
        assert cfg.federation.clients_per_round == 20

        assert cfg.training.lr == 0.10
        assert cfg.training.local_epochs == 5
        assert cfg.training.batch_size == samples

        if key == "clipped":
            assert cfg.training.grad_clip_norm == 10.0
        else:
            assert cfg.training.grad_clip_norm is None

        if key == "unclipped":
            assert cfg.training.paper00_strict is True

        if key == "fola":
            assert cfg.fola.mode == "paper_reference"

        with (
            run / "metrics/global_metrics.csv"
        ).open(newline="", encoding="utf-8") as f:

            raw_rows = [
                row for row in csv.DictReader(f)
                if row.get("round")
            ]

        round_ids = [
            int(float(row["round"]))
            for row in raw_rows
        ]

        assert len(round_ids) == len(set(round_ids)), (
            f"Duplicate rounds in {run}"
        )

        rows = dict(zip(round_ids, raw_rows))

        if not all(r in rows for r in range(1, target + 1)):
            continue

        accuracy = np.asarray([
            parse_accuracy(
                rows[r],
                expected_method,
            )
            for r in range(1, target + 1)
        ])

        communication = np.asarray([
            float(
                rows[r]["cumulative_all_array_bytes"]
            )
            for r in range(1, target + 1)
        ])

        assert np.all(np.isfinite(accuracy))
        assert np.all(np.isfinite(communication))
        assert np.all(np.diff(communication) > 0)

        metadata = run / "partition_metadata.json"

        sha = None

        if metadata.is_file():
            sha = json.loads(
                metadata.read_text()
            ).get("sha256")

        print(
            f"LOADED | {samples} samples | "
            f"{LABELS[key]} | R{target}"
        )

        return {
            "acc": accuracy,
            "bytes": communication,
            "gb": communication / 1e9,
            "cfg": cfg,
            "sha": sha,
        }

    raise RuntimeError(
        f"Incomplete R{target} run: {name}\n"
        "For the clipped five-sample FedAvg run, "
        "verify that the R300→R400 resume has finished."
    )


# ============================================================
# 2. PLOT HELPERS
# ============================================================

def smooth(y, window):
    half = window // 2

    return np.asarray([
        np.mean(
            y[max(0, i-half):min(len(y), i+half+1)]
        )
        for i in range(len(y))
    ])


def draw(ax, entries, budget, zoom=False):
    for key, data in entries:

        x = data["gb"]
        y = data["acc"]

        # Communication-width-matched smoothing:
        # 11 FOLA rounds ~= 21 FedAvg rounds.
        window = 11 if key == "fola" else 21

        ax.plot(
            x, y,
            color=COLORS[key],
            linestyle=STYLES[key],
            linewidth=0.8,
            alpha=0.25,
        )

        ax.plot(
            x, smooth(y, window),
            color=COLORS[key],
            linestyle=STYLES[key],
            linewidth=2.0,
            label=LABELS[key],
        )

    fola = next(
        d for key, d in entries
        if key == "fola"
    )

    for rf in MILESTONES:
        ax.axvline(
            fola["gb"][rf-1],
            color="gray",
            linestyle=":",
            alpha=0.45,
        )

    if zoom:
        ax.set_xlim(
            budget * 0.50,
            budget * 1.035,
        )

        late = np.concatenate([
            d["acc"][d["gb"] >= budget * 0.50]
            for _, d in entries
        ])

        pad = max(
            0.65,
            0.14 * float(np.ptp(late)),
        )

        ax.set_ylim(
            max(0, late.min() - pad),
            min(100, late.max() + pad),
        )

        ax.set_title("Late-stage zoom")

    else:
        ax.set_xlim(0, budget * 1.035)
        ax.set_title("Complete trajectory")

    ax.set_xlabel(
        "Cumulative communication (GB)"
    )

    ax.set_ylabel(
        "MNIST test accuracy (%)"
    )

    ax.grid(alpha=0.20)


def save(fig, filename):
    base = PLOT_DIR / filename

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

    print("SAVED:", str(base) + ".png")


# ============================================================
# 3. PAIRWISE FIGURES WITH MILESTONE ACCURACY TABLE
# ============================================================

def plot_pair(samples, fola, fedavg, policy):

    budget = fola["gb"][-1]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(16, 5),
        gridspec_kw={
            "width_ratios": [1.55, 1.55, 1.55]
        },
    )

    ax_full, ax_zoom, ax_table = axes

    entries = [
        ("fola", fola),
        (policy, fedavg),
    ]

    draw(ax_full, entries, budget)
    draw(ax_zoom, entries, budget, zoom=True)

    ax_full.legend(
        loc="lower right",
        fontsize=9,
    )

    # ----------------------------------------
    # Numerical comparison table
    # ----------------------------------------

    ax_table.axis("off")

    ax_table.text(
        0.5, 0.91,
        "Equal-communication milestones",
        ha="center",
        va="center",
        weight="bold",
        fontsize=11,
        transform=ax_table.transAxes,
    )

    ax_table.text(
        0.5, 0.83,
        "Round pair = FOLA / FedAvg",
        ha="center",
        fontsize=9,
        transform=ax_table.transAxes,
    )

    cells = []

    for rf in MILESTONES:
        ra = 2 * rf

        fv = fola["acc"][rf-1]
        av = fedavg["acc"][ra-1]

        cells.append([
            f"{rf}/{ra}",
            f"{fola['gb'][rf-1]:.3f}",
            f"{fv:.2f}",
            f"{av:.2f}",
            f"{fv-av:+.2f}",
        ])

    table = ax_table.table(
        cellText=cells,
        colLabels=[
            "Rounds",
            "GB",
            "FOLA %",
            "FedAvg %",
            "Gap pp",
        ],
        cellLoc="center",
        colLoc="center",
        colWidths=[
            0.25, 0.16, 0.20, 0.21, 0.18
        ],
        bbox=[0.02, 0.29, 0.96, 0.43],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)

    ax_table.text(
        0.5, 0.19,
        "Table values use raw checkpoints,\n"
        "not the smoothed plot lines.",
        ha="center",
        fontsize=9,
        color="0.35",
        transform=ax_table.transAxes,
    )

    policy_name = (
        "unclipped"
        if policy == "unclipped"
        else "clipped"
    )

    fig.suptitle(
        f"CNN 16–32 | {samples} samples/client | "
        "one class/client | LR=0.10\n"
        "FOLA (unclipped) vs "
        f"FedAvg ({policy_name})",
        fontsize=12,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.90]
    )

    save(
        fig,
        f"cnn_s{samples}_fola_unclipped_"
        f"vs_fedavg_{policy}",
    )


# ============================================================
# 4. THREE-METHOD OVERVIEW
# ============================================================

def plot_overview(samples, fola, plain, clipped):

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12.2, 4.8),
    )

    entries = [
        ("fola", fola),
        ("unclipped", plain),
        ("clipped", clipped),
    ]

    budget = fola["gb"][-1]

    draw(
        axes[0],
        entries,
        budget,
    )

    draw(
        axes[1],
        entries,
        budget,
        zoom=True,
    )

    axes[0].legend(
        loc="lower right",
        fontsize=8,
    )

    fig.suptitle(
        f"CNN 16–32 | {samples} samples/client | "
        "one class/client | LR=0.10\n"
        "Unclipped FOLA versus both "
        "FedAvg clipping policies",
        fontsize=12,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.89]
    )

    save(
        fig,
        f"cnn_s{samples}_three_methods_"
        "accuracy_vs_communication",
    )


# ============================================================
# 5. LOAD, VERIFY, GENERATE
# ============================================================

summary = []

for samples, specs in CASES.items():

    seed = specs["seed"]

    data = {
        key: load_run(
            specs[key],
            key,
            samples,
            seed,
        )
        for key in (
            "fola",
            "unclipped",
            "clipped",
        )
    }

    fola = data["fola"]

    for policy in ("unclipped", "clipped"):

        fedavg = data[policy]

        # Confirm same training-data partition.
        assert (
            partition_stem(fola["cfg"])
            == partition_stem(fedavg["cfg"])
        )

        if fola["sha"] and fedavg["sha"]:
            assert fola["sha"] == fedavg["sha"]

        # Check actual recorded communication,
        # not only theoretical parameter counts.
        assert np.allclose(
            fola["bytes"],
            fedavg["bytes"][1::2],
            rtol=1e-9,
            atol=1.0,
        ), (
            samples,
            policy,
            "Communication cost mismatch",
        )

        plot_pair(
            samples,
            fola,
            fedavg,
            policy,
        )

        for rf in MILESTONES:

            ra = 2 * rf

            fv = fola["acc"][rf-1]
            av = fedavg["acc"][ra-1]

            summary.append({
                "samples_per_client": samples,
                "selected_seed": seed,
                "fedavg_policy": policy,
                "fola_round": rf,
                "fedavg_round": ra,
                "budget_gb": float(
                    fola["gb"][rf-1]
                ),
                "fola_accuracy_percent": float(fv),
                "fedavg_accuracy_percent": float(av),
                "fola_minus_fedavg_pp": float(
                    fv - av
                ),
            })

    plot_overview(
        samples,
        fola,
        data["unclipped"],
        data["clipped"],
    )


# ============================================================
# 6. EXPORT SUMMARY CSV
# ============================================================

csv_path = (
    PLOT_DIR
    / "selected_cases_milestone_comparison.csv"
)

with csv_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(summary[0]),
    )

    writer.writeheader()
    writer.writerows(summary)


# ============================================================
# 7. PRINT RESULTS IN TERMINAL
# ============================================================

print()
print("=" * 110)
print("EQUAL-COMMUNICATION ACCURACY COMPARISON")
print("FOLA: UNCLIPPED | FEDAVG: UNCLIPPED / CLIPPED")
print("=" * 110)

print(
    "Samples | Milestone | GB    | FOLA% | "
    "FedAvg unclipped% | Gap pp | "
    "FedAvg clipped% | Gap pp"
)

for samples in CASES:

    for rf in MILESTONES:

        plain = next(
            x for x in summary
            if x["samples_per_client"] == samples
            and x["fola_round"] == rf
            and x["fedavg_policy"] == "unclipped"
        )

        clipped = next(
            x for x in summary
            if x["samples_per_client"] == samples
            and x["fola_round"] == rf
            and x["fedavg_policy"] == "clipped"
        )

        print(
            f"{samples:7d} | "
            f"{rf:3d}/{2*rf:<3d}    | "
            f"{plain['budget_gb']:5.3f} | "
            f"{plain['fola_accuracy_percent']:5.2f} | "
            f"{plain['fedavg_accuracy_percent']:16.2f} | "
            f"{plain['fola_minus_fedavg_pp']:+6.2f} | "
            f"{clipped['fedavg_accuracy_percent']:15.2f} | "
            f"{clipped['fola_minus_fedavg_pp']:+6.2f}"
        )

print()
print("PASS: All plots generated")
print("Figures:", PLOT_DIR)
print("Summary CSV:", csv_path)
