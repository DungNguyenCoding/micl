from pathlib import Path
import csv
import json
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bayesfl.config import load_config
from bayesfl.data.datasets import partition_stem

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

SAVE_DIR = ROOT / "plots/cnn_selected_fola_fedavg"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

MILESTONES = (120, 150, 200)

# Used only for finding the selected experiments.
# These identifiers are never printed on the figures.
CASES = {
    5: {
        "seed": 3,
        "fola": (
            "run_mnist_stage18_cnn16_32_fola_noclip_"
            "fixed1c_s5_n20_seed3_e5_b5_lr0p10_lam0p01_r200"
        ),
        "fedavg": (
            "run_mnist_stage15_cnn16_32_"
            "fixed1c_s5_n20_fedavg_"
            "seed3_e5_b5_lr0p10_r300"
        ),
    },
    10: {
        "seed": 0,
        "fola": (
            "run_mnist_stage18_cnn16_32_fola_noclip_"
            "fixed1c_s10_n20_seed0_e5_b10_lr0p10_lam0p01_r200"
        ),
        "fedavg": (
            "debug_mnist_stage12_cnn16_32_"
            "fixed1c_s10_n20_fedavg_"
            "seed0_e5_b10_lr0p10_r400"
        ),
    },
}


# ============================================================
# Read simulation results
# ============================================================

def parse_accuracy(row, method):
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
            return value * 100 if value <= 1.000001 else value

    raise ValueError("No valid accuracy column")


def load_run(name, method, samples, seed, target):
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

    for directory in candidates:
        resolved = directory / "resolved_config.yaml"

        if not resolved.is_file():
            continue

        cfg = load_config(resolved)

        if cfg.run_name != name or cfg.method != method:
            continue

        assert cfg.model.name == "mnist_cnn_16_32_128"
        assert cfg.runtime.seed == seed
        assert cfg.training.lr == 0.10
        assert cfg.training.local_epochs == 5
        assert cfg.training.batch_size == samples

        assert cfg.federation.num_clients == 20
        assert cfg.federation.clients_per_round == 20

        assert cfg.data.partition == {
            "type": "fixed_labels",
            "samples_per_client": samples,
            "labels_per_client": 1,
        }

        # Verify the intended optimizer settings.
        if method == "fola":
            assert cfg.training.grad_clip_norm is None
            assert cfg.fola.mode == "paper_reference"
        else:
            assert cfg.training.grad_clip_norm == 10.0

        with (
            directory / "metrics/global_metrics.csv"
        ).open(newline="", encoding="utf-8") as f:
            raw = [
                row for row in csv.DictReader(f)
                if row.get("round")
            ]

        ids = [int(float(row["round"])) for row in raw]

        assert len(ids) == len(set(ids)), (
            f"Duplicate round IDs: {directory}"
        )

        rows = dict(zip(ids, raw))

        if not all(r in rows for r in range(1, target + 1)):
            continue

        accuracy = np.array([
            parse_accuracy(rows[r], method)
            for r in range(1, target + 1)
        ])

        communication = np.array([
            float(rows[r]["cumulative_all_array_bytes"])
            for r in range(1, target + 1)
        ])

        assert np.all(np.isfinite(accuracy))
        assert np.all(np.isfinite(communication))
        assert np.all(np.diff(communication) > 0)

        metadata_path = directory / "partition_metadata.json"

        partition_sha = None

        if metadata_path.is_file():
            partition_sha = json.loads(
                metadata_path.read_text()
            ).get("sha256")

        return {
            "accuracy": accuracy,
            "bytes": communication,
            "gb": communication / 1e9,
            "cfg": cfg,
            "sha": partition_sha,
        }

    raise RuntimeError(
        f"Missing or incomplete R{target} result:\n"
        f"{name}\n"
        "Check that the R300-to-R400 continuation has finished."
    )


def moving_average(values, window=11):
    half = window // 2

    return np.array([
        np.mean(
            values[max(0, i-half):min(len(values), i+half+1)]
        )
        for i in range(len(values))
    ])


# ============================================================
# Plot styling
# ============================================================

FOLA_COLOR = "tab:blue"
FEDAVG_COLOR = "tab:orange"


def draw_accuracy_plot(ax, fola, fedavg, x_mode):

    if x_mode == "communication":
        f_x = fola["gb"]
        a_x = fedavg["gb"]

        # Similar smoothing widths in communication bytes.
        f_window = 11
        a_window = 21

        ax.set_title("Accuracy vs. communication cost")
        ax.set_xlabel("Cumulative communication (GB)")

    else:
        # Compare precisely the same number of rounds.
        f_x = np.arange(1, 201)
        a_x = np.arange(1, 201)

        f_window = 11
        a_window = 11

        ax.set_title("Accuracy vs. communication rounds")
        ax.set_xlabel("Communication rounds")

    series = [
        (
            "Dense FOLA",
            f_x,
            fola["accuracy"][:len(f_x)],
            FOLA_COLOR,
            "-",
            f_window,
        ),
        (
            "FedAvg",
            a_x,
            fedavg["accuracy"][:len(a_x)],
            FEDAVG_COLOR,
            "--",
            a_window,
        ),
    ]

    for label, x, y, color, style, window in series:

        # Raw accuracy curve.
        ax.plot(
            x, y,
            color=color,
            linestyle=style,
            alpha=0.25,
            linewidth=0.8,
        )

        # Smoothed accuracy curve.
        ax.plot(
            x,
            moving_average(y, window),
            color=color,
            linestyle=style,
            linewidth=2.0,
            label=label,
        )

    # Mark all three FOLA milestones.
    for rf in MILESTONES:

        if x_mode == "communication":
            x_mark = fola["gb"][rf-1]
            a_index = 2 * rf - 1

            f_point_x = x_mark
            a_point_x = fedavg["gb"][a_index]

        else:
            x_mark = rf
            a_index = rf - 1

            f_point_x = rf
            a_point_x = rf

        ax.axvline(
            x_mark,
            linestyle=":",
            color="gray",
            alpha=0.40,
            linewidth=0.9,
        )

        ax.scatter(
            [f_point_x],
            [fola["accuracy"][rf-1]],
            color=FOLA_COLOR,
            s=35,
            zorder=5,
            edgecolors="white",
            linewidths=0.6,
        )

        ax.scatter(
            [a_point_x],
            [fedavg["accuracy"][a_index]],
            color=FEDAVG_COLOR,
            s=35,
            marker="s",
            zorder=5,
            edgecolors="white",
            linewidths=0.6,
        )

    if x_mode == "communication":
        ax.set_xlim(
            0,
            fola["gb"][-1] * 1.04,
        )
    else:
        ax.set_xlim(0, 207)

    ax.set_ylabel("MNIST test accuracy (%)")
    ax.grid(alpha=0.20)
    ax.legend(loc="lower right", fontsize=9)


# ============================================================
# Numerical milestone tables
# ============================================================

def draw_table(ax, rows, headers, title, widths):

    ax.axis("off")

    ax.set_title(
        title,
        fontsize=11,
        weight="bold",
        pad=9,
    )

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        colWidths=widths,
        bbox=[0.01, 0.08, 0.98, 0.78],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eeeeee")

    return table


def build_tables(fola, fedavg):

    # Table A: Equal communication budget.
    equal_cost = []

    # Table B: Equal communication rounds.
    equal_rounds = []

    report_rows = []

    for rf in MILESTONES:

        ra = 2 * rf

        fv = float(fola["accuracy"][rf-1])

        # Equal-communication FedAvg accuracy.
        av_cost = float(fedavg["accuracy"][ra-1])

        # Same-round FedAvg accuracy.
        av_round = float(fedavg["accuracy"][rf-1])

        f_gb = float(fola["gb"][rf-1])

        a_gb_cost = float(fedavg["gb"][ra-1])
        a_gb_round = float(fedavg["gb"][rf-1])

        assert math.isclose(
            f_gb, a_gb_cost,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )

        equal_cost.append([
            f"{rf}/{ra}",
            f"{f_gb:.3f}",
            f"{fv:.2f}",
            f"{av_cost:.2f}",
            f"{fv-av_cost:+.2f}",
        ])

        equal_rounds.append([
            f"{rf}",
            f"{fv:.2f}",
            f"{av_round:.2f}",
            f"{fv-av_round:+.2f}",
            f"{f_gb:.3f}",
            f"{a_gb_round:.3f}",
        ])

        report_rows.extend([
            {
                "comparison": "equal_communication",
                "fola_round": rf,
                "fedavg_round": ra,
                "fola_accuracy": fv,
                "fedavg_accuracy": av_cost,
                "gap_pp": fv-av_cost,
                "fola_gb": f_gb,
                "fedavg_gb": a_gb_cost,
            },
            {
                "comparison": "same_rounds",
                "fola_round": rf,
                "fedavg_round": rf,
                "fola_accuracy": fv,
                "fedavg_accuracy": av_round,
                "gap_pp": fv-av_round,
                "fola_gb": f_gb,
                "fedavg_gb": a_gb_round,
            },
        ])

    return equal_cost, equal_rounds, report_rows


# ============================================================
# Generate one four-panel figure for each sample setting
# ============================================================

all_csv_rows = []

for samples, case in CASES.items():

    fola = load_run(
        case["fola"],
        "fola",
        samples,
        case["seed"],
        200,
    )

    fedavg = load_run(
        case["fedavg"],
        "fedavg",
        samples,
        case["seed"],
        400,
    )

    # Ensure matching partitions.
    assert (
        partition_stem(fola["cfg"])
        == partition_stem(fedavg["cfg"])
    )

    if fola["sha"] and fedavg["sha"]:
        assert fola["sha"] == fedavg["sha"]

    # Check every communication-matched round.
    assert np.allclose(
        fola["bytes"],
        fedavg["bytes"][1::2],
        rtol=1e-9,
        atol=1.0,
    )

    equal_cost, equal_rounds, report = build_tables(
        fola, fedavg
    )

    fig = plt.figure(figsize=(16.5, 8.7))

    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[3.1, 1.45],
        hspace=0.37,
        wspace=0.22,
    )

    ax_cost = fig.add_subplot(gs[0, 0])
    ax_rounds = fig.add_subplot(gs[0, 1])

    ax_table_cost = fig.add_subplot(gs[1, 0])
    ax_table_rounds = fig.add_subplot(gs[1, 1])

    # Upper-left: communication efficiency.
    draw_accuracy_plot(
        ax_cost,
        fola,
        fedavg,
        "communication",
    )

    # Upper-right: same-round convergence behavior.
    draw_accuracy_plot(
        ax_rounds,
        fola,
        fedavg,
        "rounds",
    )

    # Lower-left: matched cost numerical table.
    draw_table(
        ax_table_cost,
        equal_cost,
        [
            "Rounds\nF/A",
            "Budget\nGB",
            "FOLA\n%",
            "FedAvg\n%",
            "Gap\npp",
        ],
        "Equal-communication comparison",
        [0.18, 0.20, 0.20, 0.21, 0.21],
    )

    # Lower-right: same-round numerical table.
    draw_table(
        ax_table_rounds,
        equal_rounds,
        [
            "Round",
            "FOLA\n%",
            "FedAvg\n%",
            "Gap\npp",
            "FOLA\nGB",
            "FedAvg\nGB",
        ],
        "Same-round comparison",
        [0.12, 0.17, 0.17, 0.16, 0.19, 0.19],
    )

    fig.suptitle(
        f"CNN 16–32 | MNIST | "
        f"{samples} samples/client | 1 class/client\n"
        "N=20, LR=0.10, E=5",
        fontsize=14,
        y=0.98,
    )

    base = SAVE_DIR / (
        f"cnn_s{samples}_accuracy_cost_and_rounds"
    )

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

    print()
    print("=" * 75)
    print(f"CNN — {samples} SAMPLES/CLIENT")
    print("=" * 75)

    print("\nEQUAL COMMUNICATION")
    print(
        "FOLA/FedAvg rounds | GB | FOLA% | FedAvg% | Gap"
    )

    for row in equal_cost:
        print(" | ".join(row))

    print("\nSAME ROUNDS")
    print(
        "Round | FOLA% | FedAvg% | Gap | "
        "FOLA GB | FedAvg GB"
    )

    for row in equal_rounds:
        print(" | ".join(row))

    print("\nSAVED:", str(base) + ".png")
    print("SAVED:", str(base) + ".pdf")

    for record in report:
        record["samples_per_client"] = samples
        all_csv_rows.append(record)


# ============================================================
# Export numerical comparisons
# ============================================================

csv_file = SAVE_DIR / "milestone_accuracy_comparisons.csv"

with csv_file.open(
    "w",
    newline="",
    encoding="utf-8",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "samples_per_client",
            "comparison",
            "fola_round",
            "fedavg_round",
            "fola_accuracy",
            "fedavg_accuracy",
            "gap_pp",
            "fola_gb",
            "fedavg_gb",
        ],
    )

    writer.writeheader()
    writer.writerows(all_csv_rows)


print()
print("=" * 75)
print("PASS: ALL FIGURES AND MILESTONE TABLES GENERATED")
print("=" * 75)

print("Output directory:", SAVE_DIR)
print("Summary CSV:", csv_file)
