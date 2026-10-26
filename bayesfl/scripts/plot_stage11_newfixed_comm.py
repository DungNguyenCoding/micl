from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "plots"
OUT.mkdir(exist_ok=True)

def load_run(method, target):
    name = (
        f"debug_mnist_stage11_newfixed_{method}_"
        f"n20_seed0_e5_b128_lr0p01_"
        + ("lam0p01_" if method == "fola" else "")
        + f"r{target}"
    )

    candidates = sorted(
        [p for p in (ROOT / "outputs").glob(name + "*")
         if p.is_dir() and
         (p / "metrics/global_metrics.csv").is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    for run in candidates:
        with (run / "metrics/global_metrics.csv").open(
            newline="", encoding="utf-8"
        ) as f:
            rows = {
                int(float(row["round"])): row
                for row in csv.DictReader(f)
                if row.get("round")
            }

        if not all(r in rows for r in range(1, target + 1)):
            continue

        def acc(row):
            keys = (
                ("fola_mean_accuracy", "accuracy",
                 "global_accuracy", "test_accuracy")
                if method == "fola" else
                ("accuracy", "global_accuracy", "test_accuracy")
            )
            for key in keys:
                if row.get(key) not in (None, ""):
                    v = float(row[key])
                    if np.isfinite(v):
                        return v * 100 if v <= 1.000001 else v
            raise ValueError("Accuracy field missing")

        rounds = np.arange(1, target + 1)
        bytes_used = np.array([
            float(rows[r]["cumulative_all_array_bytes"])
            for r in rounds
        ])
        accuracy = np.array([acc(rows[r]) for r in rounds])

        assert np.all(np.diff(bytes_used) > 0)
        assert np.all(np.isfinite(accuracy))

        print(f"{method.upper()} RUN: {run.name}")
        return bytes_used, accuracy

    raise FileNotFoundError(
        f"No completed R{target} run found for {name}"
    )


def smooth(y, window=11):
    h = window // 2
    return np.array([
        np.mean(y[max(0, i-h):min(len(y), i+h+1)])
        for i in range(len(y))
    ])


f_bytes, f_acc = load_run("fola", 200)
a_bytes, a_acc = load_run("fedavg", 400)

# Exact equality of the recorded total communication budget.
assert np.isclose(
    f_bytes[-1], a_bytes[-1], rtol=1e-9
), "Final FOLA/FedAvg communication budgets differ"

f_gb = f_bytes / 1e9
a_gb = a_bytes / 1e9

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.4, 4.5))

for x, y, color, style, label in [
    (f_gb, f_acc, "tab:blue", "-", "Dense FOLA"),
    (a_gb, a_acc, "tab:orange", "--", "FedAvg"),
]:
    # Faint: raw per-round trajectory.
    ax.plot(
        x, y, color=color, linestyle=style,
        linewidth=0.85, alpha=0.30
    )

    # Strong: centered 11-round moving average.
    ax.plot(
        x, smooth(y), color=color, linestyle=style,
        linewidth=2.0, label=label
    )

    # Actual endpoint, not the smoothed value.
    ax.scatter(
        [x[-1]], [y[-1]], s=46,
        color=color, edgecolor="white", zorder=5
    )

budget = f_gb[-1]

ax.axvline(
    budget, color="gray", linestyle=":",
    linewidth=1, alpha=0.7
)

ax.annotate(
    f"FOLA R200: {f_acc[-1]:.2f}%",
    (f_gb[-1], f_acc[-1]),
    xytext=(-112, 22), textcoords="offset points",
    fontsize=9, color="tab:blue",
    arrowprops=dict(arrowstyle="-", color="tab:blue")
)

ax.annotate(
    f"FedAvg R400: {a_acc[-1]:.2f}%",
    (a_gb[-1], a_acc[-1]),
    xytext=(-112, -25), textcoords="offset points",
    fontsize=9, color="tab:orange",
    arrowprops=dict(arrowstyle="-", color="tab:orange")
)

ax.set_xlabel(
    "Total Communication Volume (GB)",
    fontsize=10
)
ax.set_ylabel(
    "MNIST Test Accuracy (%)",
    fontsize=10
)

ax.set_xlim(0, budget * 1.05)
ax.set_ylim(
    max(0, min(f_acc.min(), a_acc.min()) - 6),
    min(100, max(f_acc.max(), a_acc.max()) + 9)
)

ax.grid(True, alpha=0.22)
ax.legend(loc="upper left", frameon=True, fontsize=9)
ax.tick_params(labelsize=9)

ax.set_title(
    "New 6-Layer MLP | MNIST | 1 Class/Client | N=20",
    fontsize=10
)

fig.tight_layout()

base = OUT / "fig_stage11_newmlp_fixed_accuracy_vs_comm"

fig.savefig(str(base) + ".png", dpi=300)
fig.savefig(str(base) + ".pdf")
plt.close(fig)

# ------------------------------------------------------------
# Communication-matched statistics
# ------------------------------------------------------------

print("\nCOMMUNICATION-MATCHED CHECKPOINTS")
print("FOLA round | FedAvg round | FOLA% | FedAvg% | gap(pp)")

for r in (50, 100, 150, 200):
    ar = 2 * r

    assert np.isclose(
        f_bytes[r-1], a_bytes[ar-1], rtol=1e-9
    )

    print(
        f"R{r:<9d} | R{ar:<12d} | "
        f"{f_acc[r-1]:6.2f} | "
        f"{a_acc[ar-1]:7.2f} | "
        f"{f_acc[r-1]-a_acc[ar-1]:+7.2f}"
    )

print("\nFINAL RESULTS")
print(f"Total communication : {budget:.3f} GB")
print(f"FOLA R200           : {f_acc[-1]:.2f}%")
print(f"FedAvg R400         : {a_acc[-1]:.2f}%")
print(f"Final gap           : {f_acc[-1]-a_acc[-1]:+.2f} pp")

# Equal-width communication window:
# FOLA R181-200 consumes the same bytes as FedAvg R361-400.
f_tail = np.mean(f_acc[180:200])
a_tail = np.mean(a_acc[360:400])

print("\nCOMMUNICATION-MATCHED FINAL TAIL")
print(f"FOLA R181-200       : {f_tail:.2f}%")
print(f"FedAvg R361-400     : {a_tail:.2f}%")
print(f"Tail gap            : {f_tail-a_tail:+.2f} pp")

print("\nSAVED")
print(str(base) + ".png")
print(str(base) + ".pdf")
print("PASS")
