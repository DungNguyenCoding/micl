from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "plots"
OUT.mkdir(parents=True, exist_ok=True)

RUNS = {
    "fola": (
        "run_mnist_stage15_cnn16_32_"
        "fixed1c_s5_n20_fola_"
        "seed4_e5_b5_lr0p03_lam0p01_r150"
    ),
    "fedavg": (
        "run_mnist_stage15_cnn16_32_"
        "fixed1c_s5_n20_fedavg_"
        "seed4_e5_b5_lr0p03_r300"
    ),
}

TARGETS = {
    "fola": 150,
    "fedavg": 300,
}


# ============================================================
# Load completed results
# ============================================================

def load_run(method):

    name = RUNS[method]
    target = TARGETS[method]

    candidates = sorted(
        [
            p for p in (ROOT / "outputs").glob(name + "_*")
            if p.is_dir()
            and (p / "metrics/global_metrics.csv").is_file()
        ],
        key=lambda p: (
            p / "metrics/global_metrics.csv"
        ).stat().st_mtime,
        reverse=True,
    )

    for run in candidates:

        path = run / "metrics/global_metrics.csv"

        with path.open(newline="", encoding="utf-8") as f:
            rows = {
                int(float(row["round"])): row
                for row in csv.DictReader(f)
                if row.get("round")
            }

        if not all(r in rows for r in range(1, target + 1)):
            continue

        accuracy = []
        communication = []

        for r in range(1, target + 1):

            row = rows[r]

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

            value = None

            for key in keys:
                if row.get(key) not in (None, ""):
                    v = float(row[key])
                    if np.isfinite(v):
                        value = v
                        break

            if value is None:
                raise ValueError(
                    f"Missing accuracy at R{r}: {run}"
                )

            if value <= 1.000001:
                value *= 100

            accuracy.append(value)

            communication.append(
                float(row["cumulative_all_array_bytes"])
            )

        accuracy = np.asarray(accuracy)
        communication = np.asarray(communication)

        assert np.all(np.isfinite(accuracy))
        assert np.all(np.diff(communication) > 0)

        print(f"{method.upper()} RUN:")
        print(" ", run)

        return {
            "rounds": np.arange(1, target + 1),
            "accuracy": accuracy,
            "bytes": communication,
            "gb": communication / 1e9,
        }

    raise FileNotFoundError(
        f"Completed R{target} run not found: {name}"
    )


def smooth(y, window=11):

    half = window // 2

    return np.asarray([
        np.mean(
            y[max(0, i-half):min(len(y), i+half+1)]
        )
        for i in range(len(y))
    ])


f = load_run("fola")
a = load_run("fedavg")

# Check exact modeled communication matching.

for rf, ra in ((120, 240), (150, 300)):

    assert np.isclose(
        f["bytes"][rf-1],
        a["bytes"][ra-1],
        rtol=1e-9,
        atol=1,
    ), (
        f"Communication mismatch: "
        f"FOLA R{rf}, FedAvg R{ra}"
    )


# ============================================================
# Plot: test accuracy versus cumulative communication
# ============================================================

fig, ax = plt.subplots(figsize=(8.5, 5.2))

styles = (
    (f, "Dense FOLA", "-"),
    (a, "FedAvg", "--"),
)

for data, label, linestyle in styles:

    x = data["gb"]
    y = data["accuracy"]

    # Smoothed main curve.
    line, = ax.plot(
        x,
        smooth(y, 11),
        linestyle=linestyle,
        linewidth=2.1,
        label=f"{label} (11-round average)",
    )

    color = line.get_color()

    # Raw per-round accuracy in the background.
    ax.plot(
        x,
        y,
        linestyle=linestyle,
        color=color,
        linewidth=0.85,
        alpha=0.30,
    )

    # Mark the actual endpoint.
    ax.scatter(
        [x[-1]],
        [y[-1]],
        color=color,
        s=58,
        zorder=6,
        edgecolors="white",
        linewidths=0.8,
    )


# Communication-matched milestones.

budget120 = f["gb"][119]
budget150 = f["gb"][149]

ax.axvline(
    budget120,
    color="gray",
    linestyle=":",
    linewidth=1.0,
    alpha=0.65,
)

ax.axvline(
    budget150,
    color="gray",
    linestyle=":",
    linewidth=1.0,
    alpha=0.65,
)

# Mark R120 / R240.

ax.scatter(
    [f["gb"][119]],
    [f["accuracy"][119]],
    marker="o",
    s=30,
    zorder=5,
)

ax.scatter(
    [a["gb"][239]],
    [a["accuracy"][239]],
    marker="s",
    s=30,
    zorder=5,
)


# Endpoint annotations.

ax.annotate(
    f"FOLA R150: {f['accuracy'][149]:.2f}%",
    xy=(budget150, f["accuracy"][149]),
    xytext=(-115, 24),
    textcoords="offset points",
    fontsize=9,
    arrowprops=dict(
        arrowstyle="-",
        linewidth=0.8,
    ),
)

ax.annotate(
    f"FedAvg R300: {a['accuracy'][299]:.2f}%",
    xy=(budget150, a["accuracy"][299]),
    xytext=(-115, -29),
    textcoords="offset points",
    fontsize=9,
    arrowprops=dict(
        arrowstyle="-",
        linewidth=0.8,
    ),
)


ax.set_title(
    "CNN 16–32 | MNIST | 5 Samples/Client | Seed 4\n"
    "N=20, 1 Class/Client, LR=0.03, E=5",
    fontsize=11,
)

ax.set_xlabel(
    "Total Cumulative Communication Cost (GB)"
)

ax.set_ylabel(
    "MNIST Test Accuracy (%)"
)

ax.set_xlim(0, budget150 * 1.07)

all_accuracy = np.concatenate([
    f["accuracy"], a["accuracy"]
])

ax.set_ylim(
    max(0, np.min(all_accuracy) - 5),
    min(100, np.max(all_accuracy) + 7),
)

ax.grid(True, alpha=0.22)

ax.legend(
    loc="lower right",
    fontsize=9,
)

fig.tight_layout()

base = OUT / (
    "fig_stage15_cnn_s5_lr003_seed4_"
    "accuracy_vs_communication"
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


# ============================================================
# Print accuracy and communication comparisons
# ============================================================

print()
print("=" * 90)
print("EQUAL-COMMUNICATION ENDPOINT COMPARISON")
print("=" * 90)

for rf, ra in ((120, 240), (150, 300)):

    fa = f["accuracy"][rf-1]
    aa = a["accuracy"][ra-1]
    cost = f["gb"][rf-1]

    print(
        f"FOLA R{rf} vs FedAvg R{ra} | "
        f"FOLA={fa:.2f}% | "
        f"FedAvg={aa:.2f}% | "
        f"Gap={fa-aa:+.2f} pp | "
        f"Cost={cost:.3f} GB"
    )


# ============================================================
# Final-tail comparison, matched communication width
# ============================================================

print()
print("=" * 90)
print("COMMUNICATION-MATCHED TAIL COMPARISON")
print("=" * 90)

for rf, ra in ((120, 240), (150, 300)):

    fy = f["accuracy"]
    ay = a["accuracy"]

    f_tail = fy[rf-20:rf]
    a_tail = ay[ra-40:ra]

    f_previous = fy[rf-40:rf-20]
    a_previous = ay[ra-80:ra-40]

    fmean = f_tail.mean()
    amean = a_tail.mean()

    print(
        f"\nFOLA R{rf-19}-{rf} vs "
        f"FedAvg R{ra-39}-{ra}"
    )

    print(
        f"  Tail FOLA       : {fmean:.2f}%"
    )
    print(
        f"  Tail FedAvg     : {amean:.2f}%"
    )
    print(
        f"  Tail gap        : {fmean-amean:+.2f} pp"
    )
    print(
        f"  Within-run SD   : "
        f"FOLA={f_tail.std():.2f}, "
        f"FedAvg={a_tail.std():.2f} pp"
    )
    print(
        f"  Late improvement: "
        f"FOLA={fmean-f_previous.mean():+.2f}, "
        f"FedAvg={amean-a_previous.mean():+.2f} pp"
    )


# ============================================================
# Peak and endpoint diagnostics
# ============================================================

print()
print("=" * 90)
print("BEST ACCURACY AND FINAL ACCURACY")
print("=" * 90)

for name, data in (
    ("FOLA", f),
    ("FedAvg", a),
):
    y = data["accuracy"]
    best_i = int(np.argmax(y))

    print(
        f"{name:7s} | "
        f"best={y[best_i]:.2f}% @R{best_i+1} | "
        f"final={y[-1]:.2f}% | "
        f"drop={y[best_i]-y[-1]:.2f} pp"
    )


print()
print("SAVED FIGURES:")
print(str(base) + ".png")
print(str(base) + ".pdf")
print()
print("PASS: Seed-4 accuracy-communication plot generated")
