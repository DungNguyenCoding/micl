from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =====================================================================
# OUTPUT
# =====================================================================

OUTPUT_DIR = Path(
    "outputs/plots/paper_main"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================================================
# PROFILE REGISTRY
# =====================================================================

PROFILES = {
    "mnist_fixed": {
        "panel": "(a) MNIST",
        "subtitle": "10 samples/client, 1 class/client",

        "dense_fola": (
            "run_confirm_newsrc_"
            "mnist_fixed1c_s10_fola_n100_seed0_"
            "e10_b32_lr001_lam0p01_r150"
        ),

        "budget_fedavg": (
            "run_budgetmatch_"
            "mnist_fixed1c_s10_fedavg_n100_seed0_"
            "e10_b32_lr001_budget150densefola"
        ),
    },

    "cifar_f1": {
        "panel": "(b) CIFAR-10",
        "subtitle": r"$\alpha$=0.01, fraction=1.0",

        "dense_fola": (
            "run_confirm_newsrc_"
            "cifar_basiccnn_a0p01_fola_cos400_"
            "n20_f1_seed0_e10_b32_lr002_"
            "lam0p01_r150"
        ),

        "budget_fedavg": (
            "run_budgetmatch_"
            "cifar_basiccnn_a0p01_fedavg_"
            "n20_f1_seed0_e10_b32_lr001_"
            "budget150densefola"
        ),
    },

    "cifar_f0p5": {
        "panel": "(c) CIFAR-10",
        "subtitle": r"$\alpha$=0.01, fraction=0.5",

        "dense_fola": (
            "run_confirm_newsrc_"
            "cifar_basiccnn_a0p01_fola_cos400_"
            "n20_f0p5_seed0_e10_b32_lr002_"
            "lam0p01_r150"
        ),

        "budget_fedavg": (
            "run_budgetmatch_"
            "cifar_basiccnn_a0p01_fedavg_"
            "n20_f0p5_seed0_e10_b32_lr001_"
            "budget150densefola"
        ),
    },
}


PROFILE_ORDER = [
    "mnist_fixed",
    "cifar_f1",
    "cifar_f0p5",
]


# =====================================================================
# SPARSE SETTINGS
# =====================================================================

KEEP_RATIOS = [
    0.75,
    0.50,
    0.25,
    0.10,
]


RULES = [
    "kl_global_local",
    "kl_local_global",
    "random",
]


RULE_LABEL = {
    "kl_global_local": "KL(G→L)",
    "kl_local_global": "KL(L→G)",
    "random": "Random",
}


# =====================================================================
# PAPER STYLE
# =====================================================================

STYLE = {
    "dense_fola": {
        "label": "Dense FOLA",
        "color": "black",
        "linestyle": "-",
        "linewidth": 1.65,
        "marker": "o",
    },

    "fedavg": {
        "label": "Dense FedAvg",
        "color": "0.55",
        "linestyle": "--",
        "linewidth": 1.45,
        "marker": "^",
    },

    "kl_global_local": {
        "label": "KL(G→L)",
        "color": "tab:blue",
        "linestyle": "-",
        "linewidth": 1.55,
        "marker": "o",
    },

    "kl_local_global": {
        "label": "KL(L→G)",
        "color": "tab:orange",
        "linestyle": "-.",
        "linewidth": 1.55,
        "marker": "s",
    },

    "random": {
        "label": "Random",
        "color": "tab:green",
        "linestyle": "--",
        "linewidth": 1.45,
        "marker": "^",
    },
}


# =====================================================================
# REQUIREMENT VALIDATION VALUES
# =====================================================================
#
# These values are NOT used to construct the plots.
# They are only used to make sure the run data we resolve matches
# the requirements supplied for the paper figures.
#
# tolerance is intentionally small because these come from the same
# completed experiment set.
# =====================================================================

VALIDATION_FIG1 = {
    "mnist_fixed": {
        "dense_r150": 81.240,
        "fed_final_round": 300,
        "fed_final_acc": 82.410,
        "gl_r150": 81.250,
        "gl_final": 83.360,
        "lg_r150": 81.250,
        "lg_final": 83.350,
        "random_r150": 77.010,
        "random_final": 78.500,
    },

    "cifar_f1": {
        "dense_r150": 52.750,
        "fed_final_round": 300,
        "fed_final_acc": 47.280,
        "gl_r150": 52.780,
        "gl_final": 54.460,
        "lg_r150": 52.750,
        "lg_final": 54.410,
        "random_r150": 39.850,
        "random_final": 39.670,
    },

    "cifar_f0p5": {
        "dense_r150": 47.660,
        "fed_final_round": 300,
        "fed_final_acc": 46.320,
        "gl_r150": 50.130,
        "gl_final": 50.730,
        "lg_r150": 48.910,
        "lg_final": 51.420,
        "random_r150": 34.080,
        "random_final": 36.300,
    },
}


VALIDATION_FIG2 = {
    "mnist_fixed": {
        0.75: [+0.030, +0.030, -1.010],
        0.50: [+0.160, +0.160, -2.220],
        0.25: [+0.010, +0.010, -4.230],
        0.10: [-0.160, -0.150, -7.460],
    },

    "cifar_f1": {
        0.75: [-1.080, -0.100, -2.180],
        0.50: [-1.330, -1.020, -7.110],
        0.25: [+0.030, +0.000, -12.900],
        0.10: [-9.050, -10.040, -23.510],
    },

    "cifar_f0p5": {
        0.75: [+0.860, +0.310, -1.270],
        0.50: [+1.240, +0.860, -6.470],
        0.25: [+2.470, +1.250, -13.580],
        0.10: [-4.010, -5.640, -18.980],
    },
}


# =====================================================================
# RUN-NAME HELPERS
# =====================================================================

def ratio_tag(r: float) -> str:
    return format(
        r,
        ".8g",
    ).replace(
        ".",
        "p",
    )


def sparse_run_name(
    profile: str,
    rule: str,
    keep: float,
) -> str:

    kt = ratio_tag(
        keep
    )

    if profile == "mnist_fixed":

        return (
            "run_sparse_"
            "mnist_fixed1c_s10_fola_"
            f"{rule}_keep{kt}_"
            "n100_seed0_e10_b32_lr001_"
            "lam0p01_budget150dense"
        )

    fraction = (
        "f1"
        if profile == "cifar_f1"
        else "f0p5"
    )

    return (
        "run_sparse_"
        f"cifar_basiccnn_a0p01_{fraction}_fola_"
        f"{rule}_keep{kt}_cos400_"
        "n20_seed0_e10_b32_lr002_"
        "lam0p01_budget150dense"
    )


# =====================================================================
# OUTPUT DISCOVERY
# =====================================================================

def latest_run_dir(
    run_name: str,
) -> Path:

    root = Path(
        "./outputs"
    ).resolve()

    candidates = []

    exact = (
        root
        / run_name
    )

    if exact.is_dir():
        candidates.append(
            exact
        )

    candidates.extend(
        p
        for p in root.glob(
            run_name + "_*"
        )
        if p.is_dir()
    )

    if not candidates:

        raise FileNotFoundError(
            "\nCannot find completed run:\n"
            f"  {run_name}\n"
        )

    return max(
        candidates,
        key=lambda p:
            p.stat().st_mtime,
    )


def find_metrics(
    run_dir: Path,
) -> Path:

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

    matches = list(
        run_dir.rglob(
            "global_metrics.csv"
        )
    )

    if len(matches) == 1:
        return matches[0]

    raise RuntimeError(
        "Cannot uniquely find "
        f"global_metrics.csv under {run_dir}"
    )


# =====================================================================
# METRIC READING
# =====================================================================

def get_accuracy(
    row: dict,
    method: str,
) -> float:

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

    for key in keys:

        value = row.get(
            key
        )

        if value not in (
            None,
            "",
        ):

            x = float(
                value
            )

            if x <= 1.5:
                x *= 100.0

            return x

    raise RuntimeError(
        "No usable global-test-accuracy column."
    )


def load_curve(
    run_name: str,
    method: str,
) -> dict:

    run_dir = latest_run_dir(
        run_name
    )

    path = find_metrics(
        run_dir
    )

    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    data = {}

    for row in rows:

        value = row.get(
            "round"
        )

        if value in (
            None,
            "",
        ):
            value = row.get(
                "round_id"
            )

        if value in (
            None,
            "",
        ):
            continue

        r = int(
            float(value)
        )

        if r < 1:
            continue

        comm = row.get(
            "cumulative_all_array_bytes"
        )

        if comm in (
            None,
            "",
        ):
            continue

        data[r] = {
            "accuracy":
                get_accuracy(
                    row,
                    method,
                ),

            "bytes":
                int(
                    float(comm)
                ),
        }

    if not data:

        raise RuntimeError(
            f"No usable metrics: {path}"
        )

    rounds = np.asarray(
        sorted(data),
        dtype=int,
    )

    accuracy = np.asarray(
        [
            data[r][
                "accuracy"
            ]
            for r in rounds
        ],
        dtype=float,
    )

    communication = np.asarray(
        [
            data[r][
                "bytes"
            ]
            for r in rounds
        ],
        dtype=np.int64,
    )

    return {
        "run_name":
            run_name,

        "run_dir":
            run_dir,

        "rounds":
            rounds,

        "accuracy":
            accuracy,

        "bytes":
            communication,

        "by_round":
            data,

        "final_round":
            int(
                rounds[-1]
            ),

        "final_accuracy":
            float(
                accuracy[-1]
            ),

        "final_bytes":
            int(
                communication[-1]
            ),
    }


# =====================================================================
# LOAD REQUIRED DATA
# =====================================================================

DATA = {}


def load_all_data():

    for profile in PROFILE_ORDER:

        spec = (
            PROFILES[
                profile
            ]
        )

        dense = load_curve(
            spec[
                "dense_fola"
            ],
            "fola",
        )

        fedavg = load_curve(
            spec[
                "budget_fedavg"
            ],
            "fedavg",
        )

        sparse = {}

        for keep in KEEP_RATIOS:

            sparse[
                keep
            ] = {}

            for rule in RULES:

                sparse[
                    keep
                ][
                    rule
                ] = load_curve(
                    sparse_run_name(
                        profile,
                        rule,
                        keep,
                    ),
                    "fola",
                )

        DATA[
            profile
        ] = {
            "dense":
                dense,

            "fedavg":
                fedavg,

            "sparse":
                sparse,
        }


# =====================================================================
# VALIDATION
# =====================================================================

def close_enough(
    actual,
    expected,
    tolerance=0.021,
):
    return (
        abs(
            actual
            - expected
        )
        <= tolerance
    )


def validate_against_requirements():

    print()
    print("=" * 78)
    print(
        "VALIDATION AGAINST PLOT-REQUIREMENT VALUES"
    )
    print("=" * 78)

    all_ok = True

    for profile in PROFILE_ORDER:

        data = (
            DATA[
                profile
            ]
        )

        expected = (
            VALIDATION_FIG1[
                profile
            ]
        )

        dense150 = (
            data[
                "dense"
            ][
                "by_round"
            ][150][
                "accuracy"
            ]
        )

        checks = [
            (
                "Dense FOLA R150",
                dense150,
                expected[
                    "dense_r150"
                ],
            ),

            (
                "FedAvg final acc",
                data[
                    "fedavg"
                ][
                    "final_accuracy"
                ],
                expected[
                    "fed_final_acc"
                ],
            ),

            (
                "KL(G->L) .25 R150",
                data[
                    "sparse"
                ][0.25][
                    "kl_global_local"
                ][
                    "by_round"
                ][150][
                    "accuracy"
                ],
                expected[
                    "gl_r150"
                ],
            ),

            (
                "KL(G->L) .25 final",
                data[
                    "sparse"
                ][0.25][
                    "kl_global_local"
                ][
                    "final_accuracy"
                ],
                expected[
                    "gl_final"
                ],
            ),

            (
                "KL(L->G) .25 R150",
                data[
                    "sparse"
                ][0.25][
                    "kl_local_global"
                ][
                    "by_round"
                ][150][
                    "accuracy"
                ],
                expected[
                    "lg_r150"
                ],
            ),

            (
                "KL(L->G) .25 final",
                data[
                    "sparse"
                ][0.25][
                    "kl_local_global"
                ][
                    "final_accuracy"
                ],
                expected[
                    "lg_final"
                ],
            ),

            (
                "Random .25 R150",
                data[
                    "sparse"
                ][0.25][
                    "random"
                ][
                    "by_round"
                ][150][
                    "accuracy"
                ],
                expected[
                    "random_r150"
                ],
            ),

            (
                "Random .25 final",
                data[
                    "sparse"
                ][0.25][
                    "random"
                ][
                    "final_accuracy"
                ],
                expected[
                    "random_final"
                ],
            ),
        ]

        print()
        print(
            PROFILES[
                profile
            ][
                "panel"
            ]
        )

        for label, actual, exp in checks:

            ok = close_enough(
                actual,
                exp,
            )

            all_ok = (
                all_ok
                and ok
            )

            print(
                f"  {'PASS' if ok else 'WARN'} | "
                f"{label:24s} | "
                f"actual={actual:8.3f} | "
                f"required={exp:8.3f}"
            )

        actual_round = (
            data[
                "fedavg"
            ][
                "final_round"
            ]
        )

        expected_round = (
            expected[
                "fed_final_round"
            ]
        )

        round_ok = (
            actual_round
            == expected_round
        )

        all_ok = (
            all_ok
            and round_ok
        )

        print(
            f"  {'PASS' if round_ok else 'WARN'} | "
            f"{'FedAvg final round':24s} | "
            f"actual=R{actual_round} | "
            f"required=R{expected_round}"
        )

        # Figure 2 deltas
        dense_acc = dense150

        for keep in KEEP_RATIOS:

            actual_deltas = [
                (
                    data[
                        "sparse"
                    ][keep][rule][
                        "by_round"
                    ][150][
                        "accuracy"
                    ]
                    - dense_acc
                )
                for rule in RULES
            ]

            expected_deltas = (
                VALIDATION_FIG2[
                    profile
                ][keep]
            )

            for rule, actual, exp in zip(
                RULES,
                actual_deltas,
                expected_deltas,
            ):

                ok = close_enough(
                    actual,
                    exp,
                )

                all_ok = (
                    all_ok
                    and ok
                )

                if not ok:

                    print(
                        f"  WARN | Fig2 "
                        f"keep={keep:.2f} "
                        f"{RULE_LABEL[rule]} | "
                        f"actual={actual:+.3f} "
                        f"required={exp:+.3f}"
                    )

    print()

    if all_ok:

        print(
            "PASS: resolved experiment outputs "
            "match the supplied requirement values."
        )

    else:

        print(
            "WARNING: at least one value differs "
            "from the supplied requirement file."
        )

        print(
            "Plots will still use ACTUAL run metrics."
        )


# =====================================================================
# GENERAL MATPLOTLIB CONFIGURATION
# =====================================================================

def configure_matplotlib():

    plt.rcParams.update({
        "font.size":
            7.5,

        "axes.labelsize":
            8.0,

        "axes.titlesize":
            8.0,

        "legend.fontsize":
            7.0,

        "xtick.labelsize":
            7.0,

        "ytick.labelsize":
            7.0,

        "axes.linewidth":
            0.7,

        "grid.linewidth":
            0.45,

        "lines.solid_capstyle":
            "round",

        "lines.dash_capstyle":
            "round",

        "pdf.fonttype":
            42,

        "ps.fonttype":
            42,
    })


# =====================================================================
# FIGURE 1
# =====================================================================

def normalized_x(
    run,
    budget_bytes,
):

    return (
        100.0
        * run[
            "bytes"
        ].astype(float)
        / float(
            budget_bytes
        )
    )


def plot_accuracy_vs_communication():

    # IEEE double-column width ≈ 7.16 in.
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            7.16,
            2.82,
        ),
        sharex=True,
    )

    legend_handles = None
    legend_labels = None

    for ax, profile in zip(
        axes,
        PROFILE_ORDER,
    ):

        spec = (
            PROFILES[
                profile
            ]
        )

        data = (
            DATA[
                profile
            ]
        )

        dense = (
            data[
                "dense"
            ]
        )

        fedavg = (
            data[
                "fedavg"
            ]
        )

        gl = (
            data[
                "sparse"
            ][0.25][
                "kl_global_local"
            ]
        )

        lg = (
            data[
                "sparse"
            ][0.25][
                "kl_local_global"
            ]
        )

        rnd = (
            data[
                "sparse"
            ][0.25][
                "random"
            ]
        )

        budget_bytes = (
            dense[
                "by_round"
            ][150][
                "bytes"
            ]
        )

        series = [
            (
                "dense_fola",
                dense,
            ),

            (
                "fedavg",
                fedavg,
            ),

            (
                "kl_global_local",
                gl,
            ),

            (
                "kl_local_global",
                lg,
            ),

            (
                "random",
                rnd,
            ),
        ]

        # -------------------------------------------------------------
        # Plot complete observed trajectories only.
        # No smoothing and no interpolation.
        # -------------------------------------------------------------
        for key, run in series:

            style = (
                STYLE[
                    key
                ]
            )

            x = normalized_x(
                run,
                budget_bytes,
            )

            # All intended budget-limited trajectories should stay
            # at or just below 100%; numerical guard included.
            mask = (
                x
                <= 100.000001
            )

            ax.plot(
                x[mask],
                run[
                    "accuracy"
                ][mask],
                label=style[
                    "label"
                ],
                color=style[
                    "color"
                ],
                linestyle=style[
                    "linestyle"
                ],
                linewidth=style[
                    "linewidth"
                ],
                zorder=3,
            )

        # -------------------------------------------------------------
        # Requirement: mark Dense FOLA R150.
        # -------------------------------------------------------------
        dense_r150 = (
            dense[
                "by_round"
            ][150]
        )

        dense_x150 = (
            100.0
            * dense_r150[
                "bytes"
            ]
            / budget_bytes
        )

        dense_y150 = (
            dense_r150[
                "accuracy"
            ]
        )

        ax.scatter(
            [dense_x150],
            [dense_y150],
            color=STYLE[
                "dense_fola"
            ][
                "color"
            ],
            marker="o",
            s=20,
            linewidths=0.7,
            edgecolors="white",
            zorder=7,
        )

        # -------------------------------------------------------------
        # Requirement: mark KL(G->L), keep=.25, R150.
        # -------------------------------------------------------------
        sparse_r150 = (
            gl[
                "by_round"
            ][150]
        )

        sparse_x150 = (
            100.0
            * sparse_r150[
                "bytes"
            ]
            / budget_bytes
        )

        sparse_y150 = (
            sparse_r150[
                "accuracy"
            ]
        )

        ax.scatter(
            [sparse_x150],
            [sparse_y150],
            color=STYLE[
                "kl_global_local"
            ][
                "color"
            ],
            marker="o",
            s=22,
            linewidths=0.7,
            edgecolors="white",
            zorder=7,
        )

        communication_saving = (
            100.0
            * (
                1.0
                - sparse_r150[
                    "bytes"
                ]
                / dense_r150[
                    "bytes"
                ]
            )
        )

        # -------------------------------------------------------------
        # Communication difference annotation.
        #
        # Drawn in x-direction only; it does NOT imply identical
        # accuracy at the two R150 points.
        # -------------------------------------------------------------
        ymin, ymax = (
            ax.get_ylim()
        )

        yrange = (
            ymax
            - ymin
        )

        yarrow = (
            ymax
            - 0.11
            * yrange
        )

        ax.annotate(
            "",
            xy=(
                dense_x150,
                yarrow,
            ),
            xytext=(
                sparse_x150,
                yarrow,
            ),
            arrowprops={
                "arrowstyle":
                    "<->",

                "linewidth":
                    0.65,

                "color":
                    "0.25",
            },
            annotation_clip=True,
        )

        ax.text(
            (
                dense_x150
                + sparse_x150
            )
            / 2.0,
            yarrow
            + 0.025
            * yrange,
            (
                f"{communication_saving:.2f}% less\n"
                "communication at R150"
            ),
            ha="center",
            va="bottom",
            fontsize=6.0,
            color="0.25",
        )

        # -------------------------------------------------------------
        # Panel appearance
        # -------------------------------------------------------------
        ax.set_title(
            spec[
                "panel"
            ]
            + "\n"
            + spec[
                "subtitle"
            ],
            pad=4.0,
        )

        ax.set_xlim(
            0,
            103,
        )

        ax.set_xticks(
            [
                0,
                25,
                50,
                75,
                100,
            ]
        )

        ax.grid(
            True,
            linestyle=":",
            alpha=0.45,
        )

        ax.set_axisbelow(
            True
        )

        ax.spines[
            "top"
        ].set_visible(
            False
        )

        ax.spines[
            "right"
        ].set_visible(
            False
        )

        if legend_handles is None:

            (
                legend_handles,
                legend_labels,
            ) = ax.get_legend_handles_labels()

    # -------------------------------------------------------------
    # Shared labels and legend
    # -------------------------------------------------------------
    fig.supxlabel(
        "Communication budget used (%)",
        y=0.015,
        fontsize=8.2,
    )

    fig.supylabel(
        "Global test accuracy (%)",
        x=0.012,
        fontsize=8.2,
    )

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            1.035,
        ),
        ncol=5,
        frameon=False,
        columnspacing=1.15,
        handlelength=2.2,
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.992,
        bottom=0.205,
        top=0.755,
        wspace=0.22,
    )

    pdf = (
        OUTPUT_DIR
        / "figure_01_accuracy_vs_communication.pdf"
    )

    png = (
        OUTPUT_DIR
        / "figure_01_accuracy_vs_communication.png"
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    fig.savefig(
        png,
        dpi=400,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print()
    print(
        f"Saved Figure 1 PDF: {pdf}"
    )

    print(
        f"Saved Figure 1 PNG: {png}"
    )


# =====================================================================
# FIGURE 2
# =====================================================================

def plot_fixed_round_tradeoff():

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            7.16,
            2.76,
        ),
        sharex=True,
    )

    legend_handles = None
    legend_labels = None

    all_panel_values = {}

    for profile in PROFILE_ORDER:

        data = (
            DATA[
                profile
            ]
        )

        dense150 = (
            data[
                "dense"
            ][
                "by_round"
            ][150]
        )

        dense_acc = (
            dense150[
                "accuracy"
            ]
        )

        dense_bytes = (
            dense150[
                "bytes"
            ]
        )

        panel = {}

        for rule in RULES:

            xs = []
            ys = []
            keeps = []

            for keep in KEEP_RATIOS:

                run = (
                    data[
                        "sparse"
                    ][keep][rule]
                )

                sparse150 = (
                    run[
                        "by_round"
                    ][150]
                )

                saving = (
                    100.0
                    * (
                        1.0
                        - sparse150[
                            "bytes"
                        ]
                        / dense_bytes
                    )
                )

                delta = (
                    sparse150[
                        "accuracy"
                    ]
                    - dense_acc
                )

                xs.append(
                    saving
                )

                ys.append(
                    delta
                )

                keeps.append(
                    keep
                )

            panel[
                rule
            ] = {
                "x":
                    np.asarray(
                        xs,
                        dtype=float,
                    ),

                "y":
                    np.asarray(
                        ys,
                        dtype=float,
                    ),

                "keep":
                    keeps,
            }

        all_panel_values[
            profile
        ] = panel

    # Global y-limits so cross-panel vertical differences
    # remain visually comparable.
    all_deltas = [
        y
        for profile in PROFILE_ORDER
        for rule in RULES
        for y in all_panel_values[
            profile
        ][rule][
            "y"
        ]
    ]

    ymin = (
        min(
            all_deltas
        )
        - 1.5
    )

    ymax = (
        max(
            all_deltas
        )
        + 1.5
    )

    for ax, profile in zip(
        axes,
        PROFILE_ORDER,
    ):

        spec = (
            PROFILES[
                profile
            ]
        )

        panel = (
            all_panel_values[
                profile
            ]
        )

        # -------------------------------------------------------------
        # Dense FOLA reference
        # -------------------------------------------------------------
        ax.axhline(
            0.0,
            color="0.25",
            linestyle="--",
            linewidth=0.9,
            zorder=1,
        )

        ax.scatter(
            [0.0],
            [0.0],
            color="black",
            marker="o",
            s=21,
            zorder=5,
        )

        # -------------------------------------------------------------
        # Three sparse-selection rules
        # -------------------------------------------------------------
        for rule in RULES:

            style = (
                STYLE[
                    rule
                ]
            )

            x = (
                panel[
                    rule
                ][
                    "x"
                ]
            )

            y = (
                panel[
                    rule
                ][
                    "y"
                ]
            )

            # Points are already naturally ordered by increasing
            # communication saving for keep .75 -> .10.
            ax.plot(
                x,
                y,
                label=RULE_LABEL[
                    rule
                ],
                color=style[
                    "color"
                ],
                linestyle=style[
                    "linestyle"
                ],
                linewidth=1.45,
                marker=style[
                    "marker"
                ],
                markersize=4.0,
                markerfacecolor="white",
                markeredgewidth=0.9,
                zorder=4,
            )

        ax.set_title(
            spec[
                "panel"
            ]
            + "\n"
            + spec[
                "subtitle"
            ],
            pad=4.0,
        )

        ax.set_xlim(
            -2,
            48,
        )

        ax.set_ylim(
            ymin,
            ymax,
        )

        ax.set_xticks(
            [
                0,
                10,
                20,
                30,
                40,
            ]
        )

        ax.grid(
            True,
            linestyle=":",
            alpha=0.45,
        )

        ax.set_axisbelow(
            True
        )

        ax.spines[
            "top"
        ].set_visible(
            False
        )

        ax.spines[
            "right"
        ].set_visible(
            False
        )

        if legend_handles is None:

            (
                legend_handles,
                legend_labels,
            ) = ax.get_legend_handles_labels()

    fig.supxlabel(
        "Total communication saved at R150 (%)",
        y=0.015,
        fontsize=8.2,
    )

    fig.supylabel(
        "Accuracy change vs Dense FOLA at R150 (pp)",
        x=0.012,
        fontsize=8.2,
    )

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            1.025,
        ),
        ncol=3,
        frameon=False,
        columnspacing=1.6,
        handlelength=2.4,
    )

    fig.subplots_adjust(
        left=0.078,
        right=0.992,
        bottom=0.205,
        top=0.765,
        wspace=0.22,
    )

    pdf = (
        OUTPUT_DIR
        / "figure_02_fixed_round_tradeoff.pdf"
    )

    png = (
        OUTPUT_DIR
        / "figure_02_fixed_round_tradeoff.png"
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    fig.savefig(
        png,
        dpi=400,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print()
    print(
        f"Saved Figure 2 PDF: {pdf}"
    )

    print(
        f"Saved Figure 2 PNG: {png}"
    )


# =====================================================================
# TERMINAL DATA SUMMARY
# =====================================================================

def print_plot_data():

    print()
    print("=" * 78)
    print(
        "FIGURE 1 — KEY ENDPOINTS"
    )
    print("=" * 78)

    for profile in PROFILE_ORDER:

        data = (
            DATA[
                profile
            ]
        )

        dense = (
            data[
                "dense"
            ]
        )

        budget = (
            dense[
                "by_round"
            ][150][
                "bytes"
            ]
        )

        print()
        print(
            PROFILES[
                profile
            ][
                "panel"
            ],
            "|",
            PROFILES[
                profile
            ][
                "subtitle"
            ],
        )

        methods = [
            (
                "Dense FOLA",
                dense,
            ),

            (
                "Dense FedAvg",
                data[
                    "fedavg"
                ],
            ),

            (
                "KL(G->L) .25",
                data[
                    "sparse"
                ][0.25][
                    "kl_global_local"
                ],
            ),

            (
                "KL(L->G) .25",
                data[
                    "sparse"
                ][0.25][
                    "kl_local_global"
                ],
            ),

            (
                "Random .25",
                data[
                    "sparse"
                ][0.25][
                    "random"
                ],
            ),
        ]

        for label, run in methods:

            pct = (
                100.0
                * run[
                    "final_bytes"
                ]
                / budget
            )

            print(
                f"  {label:16s} | "
                f"R{run['final_round']:3d} | "
                f"{run['final_accuracy']:7.3f}% | "
                f"{pct:7.3f}% budget"
            )

    print()
    print("=" * 78)
    print(
        "FIGURE 2 — R150 TRADE-OFF VALUES"
    )
    print("=" * 78)

    for profile in PROFILE_ORDER:

        data = (
            DATA[
                profile
            ]
        )

        dense150 = (
            data[
                "dense"
            ][
                "by_round"
            ][150]
        )

        print()
        print(
            PROFILES[
                profile
            ][
                "panel"
            ]
        )

        print(
            f"  Dense FOLA R150 = "
            f"{dense150['accuracy']:.3f}%"
        )

        for keep in KEEP_RATIOS:

            sample = (
                data[
                    "sparse"
                ][keep][
                    "kl_global_local"
                ][
                    "by_round"
                ][150]
            )

            saving = (
                100.0
                * (
                    1.0
                    - sample[
                        "bytes"
                    ]
                    / dense150[
                        "bytes"
                    ]
                )
            )

            print(
                f"  keep={keep:.2f} | "
                f"saving={saving:6.2f}% | ",
                end="",
            )

            values = []

            for rule in RULES:

                run150 = (
                    data[
                        "sparse"
                    ][keep][rule][
                        "by_round"
                    ][150]
                )

                delta = (
                    run150[
                        "accuracy"
                    ]
                    - dense150[
                        "accuracy"
                    ]
                )

                values.append(
                    f"{RULE_LABEL[rule]}={delta:+.3f}pp"
                )

            print(
                " | ".join(
                    values
                )
            )


# =====================================================================
# MAIN
# =====================================================================

def main():

    configure_matplotlib()

    load_all_data()

    validate_against_requirements()

    print_plot_data()

    plot_accuracy_vs_communication()

    plot_fixed_round_tradeoff()

    print()
    print("=" * 78)
    print("PLOTTING COMPLETE")
    print("=" * 78)

    print(
        "Vector PDFs are the intended LaTeX/IEEE outputs."
    )

    print(
        "PNG files are high-resolution previews."
    )

    print(
        f"Output directory: "
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
