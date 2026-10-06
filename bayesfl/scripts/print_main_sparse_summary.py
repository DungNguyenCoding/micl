from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev


# ======================================================================
# EXPERIMENT REGISTRY
# ======================================================================

PROFILES = {
    "mnist_fixed": {
        "label": (
            "MNIST | fixed distribution | "
            "10 samples/client | 1 class/client"
        ),

        "dense_fola": (
            "run_confirm_newsrc_"
            "mnist_fixed1c_s10_fola_n100_seed0_"
            "e10_b32_lr001_lam0p01_r150"
        ),

        "dense_fedavg_r150": (
            "run_confirm_newsrc_"
            "mnist_fixed1c_s10_fedavg_n100_seed0_"
            "e10_b32_lr001_r150"
        ),

        "budget_fedavg": (
            "run_budgetmatch_"
            "mnist_fixed1c_s10_fedavg_n100_seed0_"
            "e10_b32_lr001_budget150densefola"
        ),
    },

    "cifar_f1": {
        "label": (
            "CIFAR-10 | alpha=0.01 | "
            "local_data_fraction=1.0"
        ),

        "dense_fola": (
            "run_confirm_newsrc_"
            "cifar_basiccnn_a0p01_fola_cos400_"
            "n20_f1_seed0_e10_b32_lr002_"
            "lam0p01_r150"
        ),

        "dense_fedavg_r150": (
            "run_confirm_newsrc_"
            "cifar_basiccnn_a0p01_fedavg_"
            "n20_f1_seed0_e10_b32_lr001_r150"
        ),

        "budget_fedavg": (
            "run_budgetmatch_"
            "cifar_basiccnn_a0p01_fedavg_"
            "n20_f1_seed0_e10_b32_lr001_"
            "budget150densefola"
        ),
    },

    "cifar_f0p5": {
        "label": (
            "CIFAR-10 | alpha=0.01 | "
            "local_data_fraction=0.5"
        ),

        "dense_fola": (
            "run_confirm_newsrc_"
            "cifar_basiccnn_a0p01_fola_cos400_"
            "n20_f0p5_seed0_e10_b32_lr002_"
            "lam0p01_r150"
        ),

        "dense_fedavg_r150": (
            "run_confirm_newsrc_"
            "cifar_basiccnn_a0p01_fedavg_"
            "n20_f0p5_seed0_e10_b32_lr001_r150"
        ),

        "budget_fedavg": (
            "run_budgetmatch_"
            "cifar_basiccnn_a0p01_fedavg_"
            "n20_f0p5_seed0_e10_b32_lr001_"
            "budget150densefola"
        ),
    },
}


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
    "kl_global_local": "KL(G->L)",
    "kl_local_global": "KL(L->G)",
    "random": "Random",
}


# ======================================================================
# REPORT WRITER
# ======================================================================

REPORT_LINES = []


def out(text=""):
    text = str(text)

    print(text)

    REPORT_LINES.append(
        text
    )


# ======================================================================
# RUN NAME CONSTRUCTION
# ======================================================================

def ratio_tag(r):
    return format(
        r,
        ".8g",
    ).replace(
        ".",
        "p",
    )


def sparse_run_name(
    profile,
    rule,
    keep,
):
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


# ======================================================================
# OUTPUT DISCOVERY
# ======================================================================

def latest_run_dir(
    run_name,
):

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

        raise RuntimeError(
            "\nNO OUTPUT FOUND:\n"
            f"  {run_name}\n"
        )

    return max(
        candidates,
        key=lambda p:
            p.stat().st_mtime,
    )


def find_metrics(
    run_dir,
):

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
        "Cannot uniquely locate "
        "global_metrics.csv under:\n"
        f"{run_dir}"
    )


# ======================================================================
# READ METRICS
# ======================================================================

def get_round(
    row,
):

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
        raise ValueError(
            "Missing round field"
        )

    return int(
        float(value)
    )


def get_accuracy(
    row,
    method,
):

    if method == "fola":

        candidates = [
            "fola_mean_accuracy",
            "accuracy",
            "global_accuracy",
            "test_accuracy",
        ]

    else:

        candidates = [
            "accuracy",
            "global_accuracy",
            "test_accuracy",
        ]

    for key in candidates:

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
        "No usable accuracy column.\n"
        f"Columns = {sorted(row)}"
    )


def get_bytes(
    row,
):

    value = row.get(
        "cumulative_all_array_bytes"
    )

    if value in (
        None,
        "",
    ):

        raise RuntimeError(
            "Missing cumulative_all_array_bytes"
        )

    return int(
        float(value)
    )


def load_run(
    run_name,
    method,
):

    run_dir = latest_run_dir(
        run_name
    )

    metric_path = find_metrics(
        run_dir
    )

    with metric_path.open(
        newline="",
        encoding="utf-8",
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    by_round = {}

    for row in rows:

        try:
            r = get_round(
                row
            )
        except Exception:
            continue

        if r < 1:
            continue

        by_round[
            r
        ] = row

    if not by_round:

        raise RuntimeError(
            "No evaluated rounds:\n"
            f"{metric_path}"
        )

    rounds = sorted(
        by_round
    )

    accuracy = {}

    communication = {}

    for r in rounds:

        accuracy[
            r
        ] = get_accuracy(
            by_round[r],
            method,
        )

        communication[
            r
        ] = get_bytes(
            by_round[r]
        )

    final_round = (
        rounds[-1]
    )

    final_accuracy = (
        accuracy[
            final_round
        ]
    )

    final_bytes = (
        communication[
            final_round
        ]
    )

    # ----------------------------------------------------------
    # Communication per completed round
    # ----------------------------------------------------------
    round_costs = []

    previous = 0

    for r in rounds:

        current = (
            communication[r]
        )

        delta = (
            current
            - previous
        )

        if delta > 0:
            round_costs.append(
                delta
            )

        previous = current

    median_round_bytes = (
        int(
            median(
                round_costs
            )
        )
        if round_costs
        else 0
    )

    # ----------------------------------------------------------
    # Stability statistics
    # ----------------------------------------------------------
    final10_rounds = (
        rounds[-10:]
    )

    final10_values = [
        accuracy[r]
        for r in final10_rounds
    ]

    last10_mean = mean(
        final10_values
    )

    last10_std = (
        pstdev(
            final10_values
        )
        if len(final10_values) > 1
        else 0.0
    )

    best_round = max(
        rounds,
        key=lambda r:
            accuracy[r],
    )

    best_accuracy = (
        accuracy[
            best_round
        ]
    )

    best_bytes = (
        communication[
            best_round
        ]
    )

    peak_to_final = (
        best_accuracy
        - final_accuracy
    )

    # ----------------------------------------------------------
    # Stop reason
    # ----------------------------------------------------------
    summary_path = (
        run_dir
        / "run_summary.json"
    )

    if summary_path.exists():

        summary = json.loads(
            summary_path.read_text(
                encoding="utf-8"
            )
        )

        stop_reason = (
            summary.get(
                "stop_reason",
                "?",
            )
        )

    else:
        stop_reason = "?"

    return {
        "run_name":
            run_name,

        "run_dir":
            run_dir,

        "metric_path":
            metric_path,

        "rounds":
            rounds,

        "accuracy":
            accuracy,

        "bytes":
            communication,

        "final_round":
            final_round,

        "final_accuracy":
            final_accuracy,

        "final_bytes":
            final_bytes,

        "median_round_bytes":
            median_round_bytes,

        "last10_mean":
            last10_mean,

        "last10_std":
            last10_std,

        "best_round":
            best_round,

        "best_accuracy":
            best_accuracy,

        "best_bytes":
            best_bytes,

        "peak_to_final":
            peak_to_final,

        "stop_reason":
            stop_reason,
    }


# ======================================================================
# ANALYSIS HELPERS
# ======================================================================

def GB(
    value,
):

    return (
        value
        / 1_000_000_000
    )


def MB(
    value,
):

    return (
        value
        / 1_000_000
    )


def first_reach(
    run,
    target_accuracy,
):

    for r in run[
        "rounds"
    ]:

        if (
            run[
                "accuracy"
            ][r]
            >= target_accuracy
        ):

            return {
                "round":
                    r,

                "bytes":
                    run[
                        "bytes"
                    ][r],

                "accuracy":
                    run[
                        "accuracy"
                    ][r],
            }

    return None


def at_or_under_budget(
    run,
    target_bytes,
):

    valid = [
        r
        for r in run[
            "rounds"
        ]
        if (
            run[
                "bytes"
            ][r]
            <= target_bytes
        )
    ]

    if not valid:
        return None

    r = max(
        valid
    )

    return {
        "round":
            r,

        "bytes":
            run[
                "bytes"
            ][r],

        "accuracy":
            run[
                "accuracy"
            ][r],
    }


def fmt_reach(
    x,
):

    if x is None:
        return "NOT REACHED"

    return (
        f"R{x['round']} | "
        f"{GB(x['bytes']):.3f} GB | "
        f"{x['accuracy']:.3f}%"
    )


def fmt_budget_cell(
    x,
):

    if x is None:
        return "N/A"

    return (
        f"{x['accuracy']:.2f}%"
        f"(R{x['round']})"
    )


def get_git_head():

    try:

        return subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    except Exception:
        return "UNKNOWN"


# ======================================================================
# LOAD ALL REQUIRED RUNS BEFORE PRINTING REPORT
# ======================================================================

DATA = {}


for profile, spec in (
    PROFILES.items()
):

    dense_fola = load_run(
        spec[
            "dense_fola"
        ],
        "fola",
    )

    dense_fedavg = load_run(
        spec[
            "dense_fedavg_r150"
        ],
        "fedavg",
    )

    budget_fedavg = load_run(
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
            ] = load_run(
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
        "dense_fola":
            dense_fola,

        "dense_fedavg":
            dense_fedavg,

        "budget_fedavg":
            budget_fedavg,

        "sparse":
            sparse,
    }


# ======================================================================
# REPORT HEADER
# ======================================================================

out()
out("=" * 132)
out("BAYESFL SPARSE-COMMUNICATION EXPERIMENT REPORT")
out("=" * 132)

out(
    "Generated : "
    + datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
)

out(
    "Repository: "
    + str(
        Path.cwd().resolve()
    )
)

out(
    "Git HEAD  : "
    + get_git_head()
)

out()

out(
    "Included completed experiments:"
)

out(
    "  - 3 confirmed Dense FOLA R150 baselines"
)

out(
    "  - 3 original Dense FedAvg R150 baselines"
)

out(
    "  - 3 communication-budget-matched Dense FedAvg runs"
)

out(
    "  - 36 Sparse FOLA runs"
)

out(
    "      3 data profiles x "
    "4 keep ratios x "
    "3 coordinate-selection rules"
)


# ======================================================================
# EXPERIMENT EXPLANATION
# ======================================================================

out()
out("=" * 132)
out("1. EXPERIMENT DESIGN")
out("=" * 132)

out("""
Dense FOLA
----------
Primary Bayesian federated-learning baseline.

keep_ratio = 1.00.

Clients communicate the complete FOLA posterior representation:
posterior mean + precision.

The communication consumed by Dense FOLA after R150 defines the
COMMON COMMUNICATION BUDGET for that data profile.


Dense FedAvg
------------
Conventional dense federated averaging baseline.

FedAvg communicates dense model parameters but does not transmit the
additional FOLA posterior precision vector.

Two FedAvg results are retained:

  1. Original R150 run:
     used for SAME-ROUND comparison.

  2. Communication-budget-matched run:
     allowed to continue beyond R150 until approximately the same
     communication budget as Dense FOLA R150 is consumed.


Sparse FOLA
-----------
Local FOLA training itself remains FULL / DENSE.

Sparsification is applied only to the client upload after local training.

Keep ratios:

    0.75
    0.50
    0.25
    0.10

The server-to-client downlink remains dense.

The sparse client upload contains selected posterior-mean coordinates,
selected precision coordinates, and a packed bitmap describing the mask.

Every sparse experiment is constrained by the communication budget of
the corresponding Dense FOLA R150 experiment.


Coordinate-selection methods
----------------------------

KL(G->L)

    Coordinate importance is ranked using the per-coordinate divergence:

        KL(global posterior before local training
           ||
           local posterior after local training)


KL(L->G)

    Reverse-KL ablation:

        KL(local posterior after local training
           ||
           global posterior before local training)


Random

    Uniform random exact-cardinality coordinate selection.

    Random uses the same keep ratio and therefore the same logical
    communication payload as the two KL selectors.


Communication metric
--------------------

    cumulative_all_array_bytes

This is the logical array payload accounted by the experiment framework.

Units reported below:

    1 MB = 1,000,000 bytes
    1 GB = 1,000,000,000 bytes
""")


# ======================================================================
# IMPORTANT INTERPRETATION NOTES
# ======================================================================

out()
out("=" * 132)
out("2. IMPORTANT INTERPRETATION / FAIRNESS NOTES")
out("=" * 132)

out("""
1. SAME ROUND and SAME COMMUNICATION answer different questions.

   R150 comparison:

       All methods have completed 150 federated rounds.

   Equal-budget comparison:

       Methods may complete different numbers of rounds, but have consumed
       approximately the same total communication.


2. Equal communication does NOT imply equal computation.

   Sparse FOLA still performs full local training before communication
   sparsification. KL scoring also requires an additional coordinate-
   importance calculation.

   Therefore, these experiments evaluate COMMUNICATION efficiency,
   not wall-clock time, FLOPs, or energy efficiency.


3. The communication metric is logical model-array payload.

   It should not be interpreted as exact Ethernet/Wi-Fi/network-wire
   traffic. Protocol headers, serialization metadata, TCP/IP overhead,
   framework control messages, etc. are not the quantity being optimized.


4. Downlink remains dense.

   Therefore, decreasing the upload keep ratio cannot reduce total
   communication proportionally to the keep ratio.

   At very small keep ratios, communication savings have diminishing
   returns because dense downlink and bitmap overhead remain.


5. Final-point accuracy can be noisy.

   For that reason, the report includes:

       final accuracy
       last-10-round mean
       last-10-round standard deviation
       best observed accuracy
       best round
       peak-to-final drop


6. All current experiments use one seed (seed 0).

   Differences reported here are descriptive for this experiment set.

   They do NOT provide an estimate of across-seed variance or statistical
   significance. Multi-seed confirmation would be needed before making
   strong statistical claims.


7. Highest-observed accuracy should not be treated as the main metric.

   Selecting the best point from a noisy trajectory can be optimistic.

   The main comparisons should remain:

       R150 accuracy
       equal-communication final accuracy
       last-10-round behavior
       complete accuracy-vs-communication trajectory
""")


# ======================================================================
# PROFILE REPORTS
# ======================================================================

for profile in (
    "mnist_fixed",
    "cifar_f1",
    "cifar_f0p5",
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

    fola = (
        data[
            "dense_fola"
        ]
    )

    fed150 = (
        data[
            "dense_fedavg"
        ]
    )

    fedbudget = (
        data[
            "budget_fedavg"
        ]
    )

    if 150 not in fola["rounds"]:
        raise RuntimeError(
            f"Dense FOLA missing R150: {profile}"
        )

    if 150 not in fed150["rounds"]:
        raise RuntimeError(
            f"FedAvg reference missing R150: {profile}"
        )

    if 150 not in fedbudget["rounds"]:
        raise RuntimeError(
            f"Budget FedAvg missing R150: {profile}"
        )

    budget_bytes = (
        fola[
            "bytes"
        ][150]
    )

    budget_gb = GB(
        budget_bytes
    )

    dense_acc = (
        fola[
            "accuracy"
        ][150]
    )

    out()
    out()
    out("#" * 132)
    out(spec["label"])
    out("#" * 132)

    out(
        f"Common communication budget = "
        f"Dense FOLA R150 = "
        f"{budget_gb:.6f} GB"
    )


    # ==================================================================
    # A. EXPERIMENT-INTEGRITY CHECKS
    # ==================================================================

    out()
    out("=" * 132)
    out("A. EXPERIMENT-INTEGRITY / FAIRNESS CHECKS")
    out("=" * 132)

    out(
        f"Dense FOLA reference : "
        f"R150 | "
        f"{dense_acc:.3f}% | "
        f"{budget_gb:.6f} GB"
    )

    old_fed_r150_acc = (
        fed150[
            "accuracy"
        ][150]
    )

    new_fed_r150_acc = (
        fedbudget[
            "accuracy"
        ][150]
    )

    old_fed_r150_bytes = (
        fed150[
            "bytes"
        ][150]
    )

    new_fed_r150_bytes = (
        fedbudget[
            "bytes"
        ][150]
    )

    out()
    out(
        "Original FedAvg R150 vs budget-matched FedAvg at R150:"
    )

    out(
        f"  old R150 accuracy       = "
        f"{old_fed_r150_acc:.6f}%"
    )

    out(
        f"  budget-run R150 accuracy= "
        f"{new_fed_r150_acc:.6f}%"
    )

    out(
        f"  accuracy difference     = "
        f"{new_fed_r150_acc-old_fed_r150_acc:+.6f} pp"
    )

    out(
        f"  old R150 bytes          = "
        f"{GB(old_fed_r150_bytes):.6f} GB"
    )

    out(
        f"  budget-run R150 bytes   = "
        f"{GB(new_fed_r150_bytes):.6f} GB"
    )

    if (
        old_fed_r150_bytes
        == new_fed_r150_bytes
    ):
        out(
            "  communication consistency: PASS"
        )
    else:
        out(
            "  communication consistency: WARNING"
        )

    out()
    out(
        "Sparse same-ratio communication equality:"
    )

    for keep in KEEP_RATIOS:

        runs = [
            data[
                "sparse"
            ][keep][rule]
            for rule in RULES
        ]

        final_rounds = {
            x[
                "final_round"
            ]
            for x in runs
        }

        final_bytes_set = {
            x[
                "final_bytes"
            ]
            for x in runs
        }

        have_r150 = all(
            150 in x[
                "rounds"
            ]
            for x in runs
        )

        stop_ok = all(
            x[
                "stop_reason"
            ]
            == "communication_budget"
            for x in runs
        )

        within_budget = all(
            x[
                "final_bytes"
            ]
            <= budget_bytes
            for x in runs
        )

        same_payload = (
            len(
                final_rounds
            )
            == 1
            and len(
                final_bytes_set
            )
            == 1
        )

        status = (
            "PASS"
            if (
                have_r150
                and stop_ok
                and within_budget
                and same_payload
            )
            else "WARNING"
        )

        out(
            f"  keep={keep:.2f} | "
            f"R150={'YES' if have_r150 else 'NO':3s} | "
            f"same bytes/rounds="
            f"{'YES' if same_payload else 'NO':3s} | "
            f"budget stop="
            f"{'YES' if stop_ok else 'NO':3s} | "
            f"within budget="
            f"{'YES' if within_budget else 'NO':3s} | "
            f"{status}"
        )

    out()
    out(
        f"Budget-matched FedAvg stop reason: "
        f"{fedbudget['stop_reason']}"
    )

    out(
        f"Budget-matched FedAvg used: "
        f"{GB(fedbudget['final_bytes']):.6f} GB "
        f"/ {budget_gb:.6f} GB "
        f"({100*fedbudget['final_bytes']/budget_bytes:.3f}%)"
    )


    # ==================================================================
    # B. ALL RUNS
    # ==================================================================

    out()
    out("=" * 132)
    out("B. ALL RUNS — ACCURACY AND COMMUNICATION COST")
    out("=" * 132)

    out(
        f"{'TYPE':18s} "
        f"{'RULE':10s} "
        f"{'KEEP':>6s} "
        f"{'FINAL_R':>8s} "
        f"{'R150_ACC':>10s} "
        f"{'FINAL_ACC':>10s} "
        f"{'R150_GB':>10s} "
        f"{'FINAL_GB':>10s} "
        f"{'MB/RND':>9s} "
        f"{'USED%':>8s}"
    )

    out(
        "-" * 132
    )

    def print_inventory_row(
        type_name,
        rule_name,
        keep,
        run,
    ):

        r150acc = (
            run[
                "accuracy"
            ][150]
            if 150 in run[
                "accuracy"
            ]
            else float("nan")
        )

        r150bytes = (
            run[
                "bytes"
            ][150]
            if 150 in run[
                "bytes"
            ]
            else 0
        )

        out(
            f"{type_name:18s} "
            f"{rule_name:10s} "
            f"{keep:6.2f} "
            f"{run['final_round']:8d} "
            f"{r150acc:9.3f}% "
            f"{run['final_accuracy']:9.3f}% "
            f"{GB(r150bytes):10.4f} "
            f"{GB(run['final_bytes']):10.4f} "
            f"{MB(run['median_round_bytes']):9.3f} "
            f"{100*run['final_bytes']/budget_bytes:7.3f}%"
        )

    print_inventory_row(
        "FedAvg R150",
        "Dense",
        1.00,
        fed150,
    )

    print_inventory_row(
        "FedAvg budget",
        "Dense",
        1.00,
        fedbudget,
    )

    print_inventory_row(
        "Dense FOLA",
        "Dense",
        1.00,
        fola,
    )

    for keep in KEEP_RATIOS:

        for rule in RULES:

            print_inventory_row(
                "Sparse FOLA",
                RULE_LABEL[
                    rule
                ],
                keep,
                data[
                    "sparse"
                ][keep][rule],
            )


    # ==================================================================
    # C. FOLA VS FEDAVG CONVERGENCE
    # ==================================================================

    out()
    out("=" * 132)
    out(
        "C. DENSE FOLA VS COMMUNICATION-BUDGET-MATCHED FEDAVG CONVERGENCE"
    )
    out("=" * 132)

    out()
    out(
        "C1. Same training-round milestones"
    )

    out(
        f"{'ROUND':>7s} | "
        f"{'FOLA ACC':>10s} "
        f"{'FOLA GB':>10s} | "
        f"{'FEDAVG ACC':>11s} "
        f"{'FEDAVG GB':>10s} | "
        f"{'FOLA-FED':>10s}"
    )

    out("-" * 78)

    for r in (
        20,
        50,
        100,
        150,
    ):

        if (
            r in fola[
                "accuracy"
            ]
            and r in fedbudget[
                "accuracy"
            ]
        ):

            fa = (
                fola[
                    "accuracy"
                ][r]
            )

            ga = (
                fedbudget[
                    "accuracy"
                ][r]
            )

            out(
                f"R{r:<6d} | "
                f"{fa:9.3f}% "
                f"{GB(fola['bytes'][r]):10.4f} | "
                f"{ga:10.3f}% "
                f"{GB(fedbudget['bytes'][r]):10.4f} | "
                f"{fa-ga:+9.3f}pp"
            )

    out()
    out(
        "C2. Same communication-budget milestones"
    )

    out(
        "Each value uses the LAST COMPLETED full round "
        "whose cumulative communication does not exceed the target."
    )

    out()

    out(
        f"{'BUDGET':>8s} | "
        f"{'TARGET GB':>10s} | "
        f"{'FOLA':>21s} | "
        f"{'FEDAVG':>21s} | "
        f"{'ACC GAP':>10s}"
    )

    out("-" * 85)

    for fraction in (
        0.25,
        0.50,
        0.75,
        1.00,
    ):

        target = int(
            budget_bytes
            * fraction
        )

        fr = at_or_under_budget(
            fola,
            target,
        )

        gr = at_or_under_budget(
            fedbudget,
            target,
        )

        gap = (
            fr["accuracy"]
            - gr["accuracy"]
            if (
                fr is not None
                and gr is not None
            )
            else None
        )

        gap_text = (
            f"{gap:+.3f}pp"
            if gap is not None
            else "N/A"
        )

        out(
            f"{100*fraction:7.0f}% | "
            f"{GB(target):10.4f} | "
            f"{fmt_budget_cell(fr):>21s} | "
            f"{fmt_budget_cell(gr):>21s} | "
            f"{gap_text:>10s}"
        )

    out()
    out(
        "C3. First point reaching Dense-FOLA milestone accuracies"
    )

    out(
        f"{'TARGET':12s} "
        f"{'ACC':>10s} | "
        f"{'DENSE FOLA FIRST REACH':30s} | "
        f"{'FEDAVG FIRST REACH':30s}"
    )

    out("-" * 94)

    for source_round in (
        50,
        100,
        150,
    ):

        target_acc = (
            fola[
                "accuracy"
            ][
                source_round
            ]
        )

        fr = first_reach(
            fola,
            target_acc,
        )

        gr = first_reach(
            fedbudget,
            target_acc,
        )

        out(
            f"{'FOLA@R'+str(source_round):12s} "
            f"{target_acc:9.3f}% | "
            f"{fmt_reach(fr):30s} | "
            f"{fmt_reach(gr):30s}"
        )

    out()
    out(
        "C4. Final equal-budget comparison"
    )

    out(
        f"  Dense FOLA : "
        f"R{fola['final_round']} | "
        f"{fola['final_accuracy']:.3f}% | "
        f"{GB(fola['final_bytes']):.6f} GB"
    )

    out(
        f"  FedAvg     : "
        f"R{fedbudget['final_round']} | "
        f"{fedbudget['final_accuracy']:.3f}% | "
        f"{GB(fedbudget['final_bytes']):.6f} GB"
    )

    out(
        f"  FedAvg - FOLA final accuracy gap: "
        f"{fedbudget['final_accuracy']-fola['final_accuracy']:+.3f} pp"
    )


    # ==================================================================
    # D. SPARSE @ R150
    # ==================================================================

    out()
    out("=" * 132)
    out(
        "D. SPARSE FOLA AT R150 — SAME NUMBER OF TRAINING ROUNDS"
    )
    out("=" * 132)

    out(
        f"Dense FOLA R150 reference: "
        f"{dense_acc:.3f}% @ "
        f"{budget_gb:.6f} GB"
    )

    out()

    out(
        f"{'KEEP':>6s} | "
        f"{'KL(G->L)':>10s} "
        f"{'DELTA':>9s} | "
        f"{'KL(L->G)':>10s} "
        f"{'DELTA':>9s} | "
        f"{'RANDOM':>10s} "
        f"{'DELTA':>9s} | "
        f"{'R150 GB':>10s} "
        f"{'COMM SAVE':>10s}"
    )

    out("-" * 122)

    for keep in KEEP_RATIOS:

        gl = (
            data[
                "sparse"
            ][keep][
                "kl_global_local"
            ]
        )

        lg = (
            data[
                "sparse"
            ][keep][
                "kl_local_global"
            ]
        )

        rnd = (
            data[
                "sparse"
            ][keep][
                "random"
            ]
        )

        gl_acc = (
            gl[
                "accuracy"
            ][150]
        )

        lg_acc = (
            lg[
                "accuracy"
            ][150]
        )

        rnd_acc = (
            rnd[
                "accuracy"
            ][150]
        )

        sparse_bytes = (
            gl[
                "bytes"
            ][150]
        )

        comm_saving = (
            100.0
            * (
                1.0
                - sparse_bytes
                / budget_bytes
            )
        )

        out(
            f"{keep:6.2f} | "
            f"{gl_acc:9.3f}% "
            f"{gl_acc-dense_acc:+8.3f} | "
            f"{lg_acc:9.3f}% "
            f"{lg_acc-dense_acc:+8.3f} | "
            f"{rnd_acc:9.3f}% "
            f"{rnd_acc-dense_acc:+8.3f} | "
            f"{GB(sparse_bytes):10.4f} "
            f"{comm_saving:9.2f}%"
        )


    # ==================================================================
    # E. SPARSE @ FINAL BUDGET
    # ==================================================================

    out()
    out("=" * 132)
    out(
        "E. SPARSE FOLA AT FINAL COMMON COMMUNICATION BUDGET"
    )
    out("=" * 132)

    out(
        f"Dense FOLA reference: "
        f"R150 | "
        f"{dense_acc:.3f}% | "
        f"{budget_gb:.6f} GB"
    )

    out()

    out(
        f"{'KEEP':>6s} "
        f"{'FINAL_R':>8s} | "
        f"{'KL(G->L)':>10s} "
        f"{'DELTA':>9s} | "
        f"{'KL(L->G)':>10s} "
        f"{'DELTA':>9s} | "
        f"{'RANDOM':>10s} "
        f"{'DELTA':>9s} | "
        f"{'USED GB':>10s} "
        f"{'USED%':>8s}"
    )

    out("-" * 129)

    for keep in KEEP_RATIOS:

        gl = (
            data[
                "sparse"
            ][keep][
                "kl_global_local"
            ]
        )

        lg = (
            data[
                "sparse"
            ][keep][
                "kl_local_global"
            ]
        )

        rnd = (
            data[
                "sparse"
            ][keep][
                "random"
            ]
        )

        if not (
            gl[
                "final_round"
            ]
            == lg[
                "final_round"
            ]
            == rnd[
                "final_round"
            ]
        ):
            raise RuntimeError(
                f"Final round mismatch at "
                f"{profile}, keep={keep}"
            )

        if not (
            gl[
                "final_bytes"
            ]
            == lg[
                "final_bytes"
            ]
            == rnd[
                "final_bytes"
            ]
        ):
            raise RuntimeError(
                f"Final-byte mismatch at "
                f"{profile}, keep={keep}"
            )

        final_r = (
            gl[
                "final_round"
            ]
        )

        used = (
            gl[
                "final_bytes"
            ]
        )

        gl_acc = (
            gl[
                "final_accuracy"
            ]
        )

        lg_acc = (
            lg[
                "final_accuracy"
            ]
        )

        rnd_acc = (
            rnd[
                "final_accuracy"
            ]
        )

        out(
            f"{keep:6.2f} "
            f"{final_r:8d} | "
            f"{gl_acc:9.3f}% "
            f"{gl_acc-dense_acc:+8.3f} | "
            f"{lg_acc:9.3f}% "
            f"{lg_acc-dense_acc:+8.3f} | "
            f"{rnd_acc:9.3f}% "
            f"{rnd_acc-dense_acc:+8.3f} | "
            f"{GB(used):10.4f} "
            f"{100*used/budget_bytes:7.3f}%"
        )


    # ==================================================================
    # F. FIXED COMMUNICATION MILESTONES FOR ALL METHODS
    # ==================================================================

    out()
    out("=" * 132)
    out(
        "F. ACCURACY AT FIXED COMMUNICATION MILESTONES"
    )
    out("=" * 132)

    out(
        "This section helps characterize the complete "
        "accuracy-vs-communication convergence behavior."
    )

    out(
        "Each cell is Accuracy%(Round), using the last completed "
        "round at or below the target communication."
    )

    out()

    out(
        f"{'METHOD':13s} "
        f"{'KEEP':>6s} | "
        f"{'25% budget':>15s} "
        f"{'50% budget':>15s} "
        f"{'75% budget':>15s} "
        f"{'100% budget':>15s}"
    )

    out("-" * 91)

    milestone_fractions = (
        0.25,
        0.50,
        0.75,
        1.00,
    )

    def print_comm_milestone_row(
        method_name,
        keep,
        run,
    ):

        cells = []

        for fraction in (
            milestone_fractions
        ):

            target = int(
                budget_bytes
                * fraction
            )

            x = at_or_under_budget(
                run,
                target,
            )

            cells.append(
                fmt_budget_cell(
                    x
                )
            )

        out(
            f"{method_name:13s} "
            f"{keep:6.2f} | "
            f"{cells[0]:>15s} "
            f"{cells[1]:>15s} "
            f"{cells[2]:>15s} "
            f"{cells[3]:>15s}"
        )

    print_comm_milestone_row(
        "Dense FOLA",
        1.00,
        fola,
    )

    print_comm_milestone_row(
        "FedAvg",
        1.00,
        fedbudget,
    )

    for rule in RULES:

        out()

        for keep in KEEP_RATIOS:

            print_comm_milestone_row(
                RULE_LABEL[
                    rule
                ],
                keep,
                data[
                    "sparse"
                ][keep][rule],
            )


    # ==================================================================
    # G. STABILITY / PEAK-TO-FINAL
    # ==================================================================

    out()
    out("=" * 132)
    out(
        "G. CONVERGENCE STABILITY / PEAK-TO-FINAL BEHAVIOR"
    )
    out("=" * 132)

    out(
        "A large BEST-FINAL drop can indicate that the final point "
        "understates an earlier peak or that late training is unstable."
    )

    out()

    out(
        f"{'METHOD':14s} "
        f"{'KEEP':>6s} "
        f"{'FINAL':>9s} "
        f"{'LAST10':>9s} "
        f"{'STD10':>8s} "
        f"{'BEST':>9s} "
        f"{'BEST_R':>7s} "
        f"{'BEST-FINAL':>11s}"
    )

    out("-" * 87)

    def print_stability(
        name,
        keep,
        run,
    ):

        out(
            f"{name:14s} "
            f"{keep:6.2f} "
            f"{run['final_accuracy']:8.3f}% "
            f"{run['last10_mean']:8.3f}% "
            f"{run['last10_std']:8.3f} "
            f"{run['best_accuracy']:8.3f}% "
            f"{run['best_round']:7d} "
            f"{run['peak_to_final']:10.3f}pp"
        )

    print_stability(
        "Dense FOLA",
        1.00,
        fola,
    )

    print_stability(
        "FedAvg",
        1.00,
        fedbudget,
    )

    for keep in KEEP_RATIOS:

        for rule in RULES:

            print_stability(
                RULE_LABEL[
                    rule
                ],
                keep,
                data[
                    "sparse"
                ][keep][rule],
            )


    # ==================================================================
    # H. KL DIRECTION AND RANDOM ABLATION
    # ==================================================================

    out()
    out("=" * 132)
    out(
        "H. COORDINATE-SELECTION ABLATION"
    )
    out("=" * 132)

    out(
        "Positive KL-Random means the KL-selected run achieved "
        "higher accuracy than random selection at the same payload."
    )

    out(
        "Positive G->L minus L->G means the proposed KL direction "
        "was higher for that specific measurement."
    )

    out()

    out(
        f"{'KEEP':>6s} | "
        f"{'G-L vs L-G R150':>17s} "
        f"{'G-L vs L-G BUD':>17s} | "
        f"{'G-L vs RAND R150':>18s} "
        f"{'G-L vs RAND BUD':>18s} | "
        f"{'L-G vs RAND R150':>18s} "
        f"{'L-G vs RAND BUD':>18s}"
    )

    out("-" * 126)

    for keep in KEEP_RATIOS:

        gl = (
            data[
                "sparse"
            ][keep][
                "kl_global_local"
            ]
        )

        lg = (
            data[
                "sparse"
            ][keep][
                "kl_local_global"
            ]
        )

        rnd = (
            data[
                "sparse"
            ][keep][
                "random"
            ]
        )

        gl150 = (
            gl[
                "accuracy"
            ][150]
        )

        lg150 = (
            lg[
                "accuracy"
            ][150]
        )

        r150 = (
            rnd[
                "accuracy"
            ][150]
        )

        glb = (
            gl[
                "final_accuracy"
            ]
        )

        lgb = (
            lg[
                "final_accuracy"
            ]
        )

        rb = (
            rnd[
                "final_accuracy"
            ]
        )

        out(
            f"{keep:6.2f} | "
            f"{gl150-lg150:+16.3f} "
            f"{glb-lgb:+16.3f} | "
            f"{gl150-r150:+17.3f} "
            f"{glb-rb:+17.3f} | "
            f"{lg150-r150:+17.3f} "
            f"{lgb-rb:+17.3f}"
        )


    # ==================================================================
    # I. DENSE-ACCURACY RETENTION
    # ==================================================================

    out()
    out("=" * 132)
    out(
        "I. WHICH SPARSE RATIOS MATCH / EXCEED DENSE FOLA "
        "AT THE FINAL COMMUNICATION BUDGET?"
    )
    out("=" * 132)

    out(
        f"Dense FOLA target accuracy = "
        f"{dense_acc:.3f}%"
    )

    out()

    for rule in RULES:

        qualifying = []

        for keep in KEEP_RATIOS:

            run = (
                data[
                    "sparse"
                ][keep][rule]
            )

            if (
                run[
                    "final_accuracy"
                ]
                >= dense_acc
            ):
                qualifying.append(
                    keep
                )

        if qualifying:

            most_aggressive = min(
                qualifying
            )

            text = (
                ", ".join(
                    f"{x:.2f}"
                    for x in qualifying
                )
            )

            out(
                f"{RULE_LABEL[rule]:10s}: "
                f"matching/exceeding ratios = "
                f"[{text}] | "
                f"lowest qualifying keep = "
                f"{most_aggressive:.2f}"
            )

        else:

            out(
                f"{RULE_LABEL[rule]:10s}: "
                "no tested keep ratio reached "
                "Dense-FOLA final-budget accuracy"
            )


# ======================================================================
# CROSS-PROFILE SUMMARY
# ======================================================================

out()
out()
out("=" * 132)
out("3. CROSS-PROFILE SUMMARY")
out("=" * 132)

out(
    "The 'highest observed' sparse values below are descriptive "
    "single-seed maxima, not statistical rankings."
)

out()

out(
    f"{'PROFILE':14s} "
    f"{'DENSE FOLA':>11s} "
    f"{'FEDAVG BUD':>11s} "
    f"{'FED-FOLA':>10s} | "
    f"{'G->L HIGH':>10s} "
    f"{'KEEP':>6s} | "
    f"{'L->G HIGH':>10s} "
    f"{'KEEP':>6s} | "
    f"{'RAND HIGH':>10s} "
    f"{'KEEP':>6s}"
)

out("-" * 117)

for profile in (
    "mnist_fixed",
    "cifar_f1",
    "cifar_f0p5",
):

    data = (
        DATA[
            profile
        ]
    )

    fola = (
        data[
            "dense_fola"
        ]
    )

    fed = (
        data[
            "budget_fedavg"
        ]
    )

    maxima = {}

    for rule in RULES:

        values = [
            (
                data[
                    "sparse"
                ][keep][rule][
                    "final_accuracy"
                ],
                keep,
            )
            for keep in KEEP_RATIOS
        ]

        maxima[
            rule
        ] = max(
            values,
            key=lambda x:
                x[0],
        )

    profile_short = {
        "mnist_fixed":
            "MNIST fixed",

        "cifar_f1":
            "CIFAR f1",

        "cifar_f0p5":
            "CIFAR f0.5",
    }[
        profile
    ]

    out(
        f"{profile_short:14s} "
        f"{fola['final_accuracy']:10.3f}% "
        f"{fed['final_accuracy']:10.3f}% "
        f"{fed['final_accuracy']-fola['final_accuracy']:+9.3f} | "
        f"{maxima['kl_global_local'][0]:9.3f}% "
        f"{maxima['kl_global_local'][1]:6.2f} | "
        f"{maxima['kl_local_global'][0]:9.3f}% "
        f"{maxima['kl_local_global'][1]:6.2f} | "
        f"{maxima['random'][0]:9.3f}% "
        f"{maxima['random'][1]:6.2f}"
    )


# ======================================================================
# FINAL CHECKLIST
# ======================================================================

out()
out()
out("=" * 132)
out("4. WHAT TO PAY ATTENTION TO WHEN INTERPRETING THESE RESULTS")
out("=" * 132)

out("""
A. Do KL selectors separate clearly from Random?

   If KL remains accurate while Random deteriorates at the exact same
   communication payload, that is evidence that coordinate selection
   matters rather than the gain coming only from performing more rounds.


B. Does one KL direction consistently outperform the other?

   Compare KL(G->L) vs KL(L->G) at:

       R150
       final communication budget
       intermediate communication milestones

   A small or inconsistent difference suggests that the value may come
   mainly from KL-based importance ranking rather than the KL direction.


C. Where does aggressive sparsification begin to fail?

   Observe whether accuracy first improves or remains stable as keep ratio
   decreases, then deteriorates below some ratio.

   This threshold may differ between MNIST, CIFAR full data, and CIFAR
   fraction=0.5.


D. Separate same-round effects from extra-round effects.

   A sparse method can be worse at R150 but recover by the final budget
   because its cheaper communication allows additional rounds.

   Therefore both R150 and equal-budget comparisons are necessary.


E. Check late-training stability.

   Large BEST-FINAL values or large last-10 standard deviations indicate
   that a single final checkpoint may not represent the trajectory well.


F. Compare communication saving at R150.

   This answers how much communication is removed when the same amount
   of training has been performed.


G. Compare accuracy at fixed communication milestones.

   This gives a stronger picture of communication convergence than using
   only the final budget point.


H. Remember the current statistical limitation.

   All current results are seed 0 only.

   Before treating small differences such as 0.1-0.5 percentage points as
   meaningful, repeated seeds should be considered.


I. Do not interpret communication savings as compute savings.

   Local FOLA training remains dense. The present study isolates the
   communication side of the problem.
""")


# ======================================================================
# RESOLVED RUN DIRECTORIES
# ======================================================================

out()
out("=" * 132)
out("5. RESOLVED OUTPUT DIRECTORIES")
out("=" * 132)

for profile in (
    "mnist_fixed",
    "cifar_f1",
    "cifar_f0p5",
):

    out()
    out(
        PROFILES[
            profile
        ][
            "label"
        ]
    )

    data = (
        DATA[
            profile
        ]
    )

    out(
        "  Dense FOLA    : "
        + str(
            data[
                "dense_fola"
            ][
                "run_dir"
            ]
        )
    )

    out(
        "  FedAvg R150   : "
        + str(
            data[
                "dense_fedavg"
            ][
                "run_dir"
            ]
        )
    )

    out(
        "  FedAvg budget : "
        + str(
            data[
                "budget_fedavg"
            ][
                "run_dir"
            ]
        )
    )

    for keep in KEEP_RATIOS:

        for rule in RULES:

            out(
                f"  Sparse {RULE_LABEL[rule]:10s} "
                f"keep={keep:.2f}: "
                + str(
                    data[
                        "sparse"
                    ][keep][rule][
                        "run_dir"
                    ]
                )
            )


# ======================================================================
# SAVE REPORT
# ======================================================================

out()
out("=" * 132)
out("REPORT COMPLETE")
out("=" * 132)

out(
    "All numerical results above were extracted from the actual "
    "completed global_metrics.csv files."
)

out(
    "The report does not replace the original experiment outputs."
)


def main_save():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        default=(
            "outputs/reports/"
            "main_sparse_summary.txt"
        ),
    )

    args = parser.parse_args()

    path = Path(
        args.output
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "\n".join(
            REPORT_LINES
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        f"TXT report saved to: "
        f"{path.resolve()}"
    )


if __name__ == "__main__":
    main_save()
