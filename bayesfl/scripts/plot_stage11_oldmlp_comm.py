from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "plots"
OUT.mkdir(exist_ok=True)

CASES = {
    "olddir": {
        "title": "Old MLP | MNIST Dirichlet α=0.01 | N=20",
        "fola": (
            "debug_p02_fig5repro_mnist_oldmlp_"
            "a0p01_n20_fola_b128_e5_lr0p01_r200_seed0"
        ),
        "fedavg": (
            "debug_p02_fig5repro_mnist_oldmlp_"
            "a0p01_n20_fedavg_b128_e5_lr0p01_r200_seed0"
        ),
        "filename": "fig_stage11_oldmlp_dirichlet_accuracy_vs_comm",
    },
    "oldfixed": {
        "title": "Old MLP | MNIST 1 Class/Client | N=20",
        "fola": (
            "debug_mnist_stage11_oldfixed_fola_"
            "n20_seed0_e5_b128_lr0p01_lam0p01_r200"
        ),
        "fedavg": (
            "debug_mnist_stage11_oldfixed_fedavg_"
            "n20_seed0_e5_b128_lr0p01_r400"
        ),
        "filename": "fig_stage11_oldmlp_fixed_accuracy_vs_comm",
    },
}


def load_run(prefix, method, target):
    candidates = sorted(
        [
            p for p in (ROOT / "outputs").glob(prefix + "*")
            if p.is_dir()
            and (p / "metrics/global_metrics.csv").is_file()
        ],
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

        def get_acc(row):
            keys = (
                ("fola_mean_accuracy", "accuracy",
                 "global_accuracy", "test_accuracy")
                if method == "fola"
                else
                ("accuracy", "global_accuracy", "test_accuracy")
            )

            for key in keys:
                if row.get(key) not in (None, ""):
                    v = float(row[key])
                    if np.isfinite(v):
                        return v * 100 if v <= 1.000001 else v

            raise ValueError("No accuracy metric found")

        rounds = np.arange(1, target + 1)

        comm = np.array([
            float(rows[r]["cumulative_all_array_bytes"])
            for r in rounds
        ]) / 1e9

        acc = np.array([
            get_acc(rows[r]) for r in rounds
        ])

        assert np.all(np.diff(comm) > 0)
        assert np.all(np.isfinite(acc))

        print(f"{method.upper()} R{target}: {run.name}")

        return comm, acc

    raise FileNotFoundError(
        f"No completed R{target} output for {prefix}"
    )


def smooth(y, window=11):
    h = window // 2
    return np.array([
        np.mean(y[max(0, i-h):min(len(y), i+h+1)])
        for i in range(len(y))
    ])


def draw(ax, fcomm, facc, acomm, aacc, title):
    for x, y, color, style, name in [
        (fcomm, facc, "tab:blue", "-", "Dense FOLA"),
        (acomm, aacc, "tab:orange", "--", "FedAvg"),
    ]:
        # Raw accuracy
        ax.plot(
            x, y, color=color, linestyle=style,
            linewidth=0.8, alpha=0.28
        )

        # Smoothed accuracy
        ax.plot(
            x, smooth(y, 11),
            color=color, linestyle=style,
            linewidth=2.0, label=name
        )

        # Actual endpoint
        ax.scatter(
            [x[-1]], [y[-1]],
            color=color, s=45,
            edgecolor="white", zorder=5
        )

    budget = fcomm[-1]

    ax.axvline(
        budget, color="gray",
        linestyle=":", linewidth=1
    )

    ax.annotate(
        f"FOLA R200: {facc[-1]:.2f}%",
        (fcomm[-1], facc[-1]),
        xytext=(-8, 16),
        textcoords="offset points",
        ha="right", color="tab:blue",
        fontsize=8.5
    )

    ax.annotate(
        f"FedAvg R400: {aacc[-1]:.2f}%",
        (acomm[-1], aacc[-1]),
        xytext=(-8, -23),
        textcoords="offset points",
        ha="right", color="tab:orange",
        fontsize=8.5
    )

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Total Communication Volume (GB)")
    ax.set_ylabel("MNIST Test Accuracy (%)")
    ax.set_xlim(0, budget * 1.06)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left", fontsize=9)


results = {}

for key, case in CASES.items():
    fcomm, facc = load_run(case["fola"], "fola", 200)
    acomm, aacc = load_run(case["fedavg"], "fedavg", 400)

    assert np.isclose(
        fcomm[-1], acomm[-1], rtol=1e-9
    ), f"Communication budget mismatch: {key}"

    for r in (50, 100, 150, 200):
        assert np.isclose(
            fcomm[r-1], acomm[2*r-1], rtol=1e-9
        )

    results[key] = (fcomm, facc, acomm, aacc)

# Same y-scale across both cases for fair visualization.
all_acc = np.concatenate([
    arr
    for fcomm, facc, acomm, aacc in results.values()
    for arr in (facc, aacc)
])

ymin = max(0, 5 * np.floor((all_acc.min() - 5) / 5))
ymax = min(100, 5 * np.ceil((all_acc.max() + 5) / 5))

# ------------------------------------------------------------
# Two individual figures
# ------------------------------------------------------------

for key, case in CASES.items():
    fig, ax = plt.subplots(figsize=(7.4, 4.5))

    draw(ax, *results[key], case["title"])
    ax.set_ylim(ymin, ymax)

    fig.tight_layout()

    base = OUT / case["filename"]
    fig.savefig(str(base) + ".png", dpi=300)
    fig.savefig(str(base) + ".pdf")
    plt.close(fig)

    print("SAVED:", str(base) + ".png")

# ------------------------------------------------------------
# Combined two-panel figure
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1, 2, figsize=(13.5, 4.5),
    sharex=True, sharey=True
)

for ax, key in zip(axes, ("olddir", "oldfixed")):
    draw(ax, *results[key], CASES[key]["title"])
    ax.set_ylim(ymin, ymax)

fig.tight_layout()

base = OUT / "fig_stage11_oldmlp_2cases_accuracy_vs_comm"

fig.savefig(str(base) + ".png", dpi=300)
fig.savefig(str(base) + ".pdf")
plt.close(fig)

print("SAVED:", str(base) + ".png")

# ------------------------------------------------------------
# Terminal summary
# ------------------------------------------------------------

print("\nCOMMUNICATION-MATCHED RESULTS")
print(
    "Case       | Budget GB | FOLA R200 | FedAvg R400 | "
    "Final gap | Equal-comm tail gap"
)
print("-" * 92)

for key in ("olddir", "oldfixed"):
    fc, fa, ac, aa = results[key]

    # Equal communication tail window:
    # 20 FOLA rounds = 40 FedAvg rounds.
    f_tail = fa[180:200].mean()
    a_tail = aa[360:400].mean()

    print(
        f"{key:10s} | "
        f"{fc[-1]:9.3f} | "
        f"{fa[-1]:9.2f}% | "
        f"{aa[-1]:10.2f}% | "
        f"{fa[-1]-aa[-1]:+8.2f} pp | "
        f"{f_tail-a_tail:+8.2f} pp"
    )

print("\nPASS: all plots generated")
