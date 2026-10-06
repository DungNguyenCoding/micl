from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


DENSE_FEDAVG = {
    "cifar_f1":
        "run_confirm_newsrc_cifar_basiccnn_a0p01_fedavg_n20_f1_seed0_e10_b32_lr001_r150",

    "cifar_f0p5":
        "run_confirm_newsrc_cifar_basiccnn_a0p01_fedavg_n20_f0p5_seed0_e10_b32_lr001_r150",

    "mnist_fixed":
        "run_confirm_newsrc_mnist_fixed1c_s10_fedavg_n100_seed0_e10_b32_lr001_r150",
}


DENSE_FOLA = {
    "cifar_f1":
        "run_confirm_newsrc_cifar_basiccnn_a0p01_fola_cos400_n20_f1_seed0_e10_b32_lr002_lam0p01_r150",

    "cifar_f0p5":
        "run_confirm_newsrc_cifar_basiccnn_a0p01_fola_cos400_n20_f0p5_seed0_e10_b32_lr002_lam0p01_r150",

    "mnist_fixed":
        "run_confirm_newsrc_mnist_fixed1c_s10_fola_n100_seed0_e10_b32_lr001_lam0p01_r150",
}


PROFILE_TITLE = {
    "cifar_f1":
        "CIFAR-10 | alpha=.01 | local fraction=1.0",

    "cifar_f0p5":
        "CIFAR-10 | alpha=.01 | local fraction=.5",

    "mnist_fixed":
        "MNIST | fixed 10 samples/client | 1 class/client",
}


KEEP_RATIOS = [0.75, 0.50, 0.25, 0.10]

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


def ratio_tag(r):
    return format(r, ".8g").replace(".", "p")


def sparse_run_name(profile, rule, keep):
    kt = ratio_tag(keep)

    if profile == "mnist_fixed":
        return (
            f"run_sparse_mnist_fixed1c_s10_fola_"
            f"{rule}_keep{kt}_"
            f"n100_seed0_e10_b32_lr001_lam0p01_"
            f"budget150dense"
        )

    fraction = (
        "f1"
        if profile == "cifar_f1"
        else "f0p5"
    )

    return (
        f"run_sparse_cifar_basiccnn_a0p01_{fraction}_fola_"
        f"{rule}_keep{kt}_cos400_"
        f"n20_seed0_e10_b32_lr002_lam0p01_"
        f"budget150dense"
    )


def latest_run_dir(root, run_name):
    candidates = []

    exact = root / run_name

    if exact.is_dir():
        candidates.append(exact)

    candidates.extend(
        p
        for p in root.glob(run_name + "_*")
        if p.is_dir()
    )

    if not candidates:
        raise FileNotFoundError(
            f"No output found for {run_name}"
        )

    return max(
        candidates,
        key=lambda p: p.stat().st_mtime,
    )


def find_metrics(run_dir):
    candidates = [
        run_dir / "metrics" / "global_metrics.csv",
        run_dir / "global_metrics.csv",
    ]

    for p in candidates:
        if p.exists():
            return p

    found = list(
        run_dir.rglob("global_metrics.csv")
    )

    if len(found) == 1:
        return found[0]

    raise RuntimeError(
        f"Cannot uniquely locate global_metrics.csv in {run_dir}"
    )


def read_curve(run_dir, method):
    path = find_metrics(run_dir)

    rounds = []
    acc = []
    comm = []

    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    for row in rows:

        rv = row.get("round")

        if rv in (None, ""):
            rv = row.get("round_id")

        if rv in (None, ""):
            continue

        r = int(float(rv))

        # Ignore round 0 initialization point.
        if r < 1:
            continue

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

        a = None

        for key in keys:

            value = row.get(key)

            if value not in (
                None,
                "",
            ):
                a = float(value)
                break

        if a is None:
            continue

        if a <= 1.5:
            a *= 100.0

        b = row.get(
            "cumulative_all_array_bytes"
        )

        if b in (None, ""):
            continue

        rounds.append(r)
        acc.append(a)
        comm.append(
            float(b) / 1e9
        )

    if not rounds:
        raise RuntimeError(
            f"No usable rows in {path}"
        )

    # Protect against accidental duplicated round rows.
    order = np.argsort(rounds)

    rounds = np.asarray(
        rounds,
        dtype=int,
    )[order]

    acc = np.asarray(
        acc,
        dtype=float,
    )[order]

    comm = np.asarray(
        comm,
        dtype=float,
    )[order]

    if len(np.unique(rounds)) != len(rounds):
        raise RuntimeError(
            f"Duplicate evaluated rounds in {path}"
        )

    return rounds, comm, acc


def gaussian_smooth_x(
    x,
    y,
    bandwidth,
):
    """
    Gaussian kernel smoother defined in communication (GB),
    not in number of rounds.

    Result is evaluated at each original x location.
    """
    result = np.empty_like(
        y,
        dtype=float,
    )

    for i, xi in enumerate(x):

        z = (
            (x - xi)
            / bandwidth
        )

        weights = np.exp(
            -0.5 * z * z
        )

        sw = weights.sum()

        if sw <= 0:
            result[i] = y[i]
        else:
            result[i] = np.dot(
                weights,
                y,
            ) / sw

    return result


def roughness(y):
    """
    Median absolute point-to-point change.
    Lower means visually smoother.
    """
    if len(y) < 2:
        return 0.0

    return float(
        np.median(
            np.abs(
                np.diff(y)
            )
        )
    )


def checkpoints(rounds, acc):
    wanted = [
        1,
        5,
        10,
        20,
        30,
        50,
        75,
        100,
        125,
        150,
        175,
        200,
        225,
        250,
    ]

    mapping = {
        int(r): float(a)
        for r, a in zip(
            rounds,
            acc,
        )
    }

    out = []

    for r in wanted:
        if r in mapping:
            out.append(
                (r, mapping[r])
            )

    final_r = int(
        rounds[-1]
    )

    if final_r not in {
        r for r, _ in out
    }:
        out.append(
            (
                final_r,
                mapping[final_r],
            )
        )

    return out


def describe_series(
    name,
    run_dir,
    rounds,
    comm,
    acc,
):
    dacc = np.diff(acc)
    absd = np.abs(dacc)

    dcomm = np.diff(comm)

    median_step = (
        float(np.median(absd))
        if len(absd)
        else 0.0
    )

    mean_step = (
        float(np.mean(absd))
        if len(absd)
        else 0.0
    )

    p90_step = (
        float(np.percentile(absd, 90))
        if len(absd)
        else 0.0
    )

    max_step = (
        float(np.max(absd))
        if len(absd)
        else 0.0
    )

    # A "large" jump here is data-driven:
    # > max(1 percentage point, 3 * median step).
    large_threshold = max(
        1.0,
        3.0 * median_step,
    )

    large_count = int(
        np.sum(
            absd > large_threshold
        )
    )

    gb_round = (
        float(np.median(dcomm))
        if len(dcomm)
        else float("nan")
    )

    print()
    print("=" * 104)
    print(name)
    print("=" * 104)

    print(
        f"run directory          : {run_dir}"
    )

    print(
        f"rounds                 : "
        f"R{rounds[0]} -> R{rounds[-1]} "
        f"({len(rounds)} evaluated training rounds)"
    )

    print(
        f"communication          : "
        f"{comm[0]:.6f} -> {comm[-1]:.6f} GB"
    )

    print(
        f"median GB / round      : "
        f"{gb_round:.6f} GB"
    )

    print(
        f"accuracy               : "
        f"{acc[0]:.3f}% -> {acc[-1]:.3f}%"
    )

    print()
    print("ROUND-TO-ROUND ACCURACY FLUCTUATION")

    print(
        f"  median |Δacc|        : "
        f"{median_step:.4f} pp"
    )

    print(
        f"  mean   |Δacc|        : "
        f"{mean_step:.4f} pp"
    )

    print(
        f"  p90    |Δacc|        : "
        f"{p90_step:.4f} pp"
    )

    print(
        f"  maximum|Δacc|        : "
        f"{max_step:.4f} pp"
    )

    print(
        f"  large-jump threshold : "
        f"{large_threshold:.4f} pp"
    )

    print(
        f"  large jumps          : "
        f"{large_count}/{len(absd)}"
    )

    if len(absd):

        largest = np.argsort(
            absd
        )[-5:][::-1]

        print()
        print("  Largest 5 round-to-round jumps:")

        for idx in largest:

            r0 = int(
                rounds[idx]
            )

            r1 = int(
                rounds[idx + 1]
            )

            a0 = acc[idx]
            a1 = acc[idx + 1]

            print(
                f"    R{r0:3d} -> R{r1:3d}: "
                f"{a0:7.3f}% -> {a1:7.3f}% "
                f"({a1-a0:+7.3f} pp), "
                f"x={comm[idx]:.3f}->{comm[idx+1]:.3f} GB"
            )

    print()
    print("RAW CHECKPOINTS")

    for r, a in checkpoints(
        rounds,
        acc,
    ):
        idx = np.where(
            rounds == r
        )[0][0]

        print(
            f"  R{r:3d} | "
            f"{comm[idx]:9.4f} GB | "
            f"{a:7.3f}%"
        )

    return {
        "name": name,
        "x": comm,
        "y": acc,
        "raw_roughness": roughness(acc),
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--profile",
        required=True,
        choices=[
            "cifar_f1",
            "cifar_f0p5",
            "mnist_fixed",
        ],
    )

    parser.add_argument(
        "--outputs-root",
        default="./outputs",
    )

    args = parser.parse_args()

    root = Path(
        args.outputs_root
    ).resolve()

    profile = args.profile

    print()
    print("#" * 104)
    print("ACCURACY FLUCTUATION DIAGNOSTIC")
    print(PROFILE_TITLE[profile])
    print("#" * 104)

    series = []

    # ------------------------------------------------------------
    # Dense FedAvg
    # ------------------------------------------------------------
    name = DENSE_FEDAVG[
        profile
    ]

    run_dir = latest_run_dir(
        root,
        name,
    )

    rounds, comm, acc = read_curve(
        run_dir,
        "fedavg",
    )

    series.append(
        describe_series(
            "FedAvg dense",
            run_dir,
            rounds,
            comm,
            acc,
        )
    )

    # ------------------------------------------------------------
    # Dense FOLA
    # ------------------------------------------------------------
    name = DENSE_FOLA[
        profile
    ]

    run_dir = latest_run_dir(
        root,
        name,
    )

    rounds, comm, acc = read_curve(
        run_dir,
        "fola",
    )

    series.append(
        describe_series(
            "Dense FOLA keep=1.00",
            run_dir,
            rounds,
            comm,
            acc,
        )
    )

    # ------------------------------------------------------------
    # Sparse variants
    # ------------------------------------------------------------
    for keep in KEEP_RATIOS:

        for rule in RULES:

            name = sparse_run_name(
                profile,
                rule,
                keep,
            )

            try:
                run_dir = latest_run_dir(
                    root,
                    name,
                )

            except FileNotFoundError:
                print(
                    f"\n[NOT FINISHED / NOT FOUND] "
                    f"{RULE_LABEL[rule]} keep={keep:.2f}"
                )
                continue

            rounds, comm, acc = read_curve(
                run_dir,
                "fola",
            )

            series.append(
                describe_series(
                    (
                        f"{RULE_LABEL[rule]} "
                        f"keep={keep:.2f}"
                    ),
                    run_dir,
                    rounds,
                    comm,
                    acc,
                )
            )

    # ------------------------------------------------------------
    # Candidate smoothing diagnostics.
    #
    # Fixed bandwidth is in GB, which matches the plot x-axis.
    # ------------------------------------------------------------
    bandwidths = [
        0.50,
        1.00,
        1.50,
        2.00,
        3.00,
    ]

    print()
    print("#" * 104)
    print(
        "CANDIDATE COMMUNICATION-DOMAIN SMOOTHING"
    )
    print(
        "Gaussian kernel evaluated in GB, not in training-round count."
    )
    print(
        "MAE/change values are smoothing distortion in accuracy percentage points."
    )
    print("#" * 104)

    print(
        f"{'BW(GB)':>8s} | "
        f"{'mean MAE':>10s} | "
        f"{'p95 |change|':>13s} | "
        f"{'max |change|':>13s} | "
        f"{'rough raw':>10s} | "
        f"{'rough smooth':>12s} | "
        f"{'rough reduction':>15s}"
    )

    print("-" * 104)

    for bw in bandwidths:

        maes = []
        all_changes = []
        raw_rough = []
        smooth_rough = []

        for s in series:

            x = s["x"]
            y = s["y"]

            sm = gaussian_smooth_x(
                x,
                y,
                bw,
            )

            change = np.abs(
                y - sm
            )

            maes.append(
                float(
                    np.mean(
                        change
                    )
                )
            )

            all_changes.extend(
                change.tolist()
            )

            raw_rough.append(
                roughness(y)
            )

            smooth_rough.append(
                roughness(sm)
            )

        mean_mae = float(
            np.mean(maes)
        )

        p95_change = float(
            np.percentile(
                all_changes,
                95,
            )
        )

        max_change = float(
            np.max(
                all_changes
            )
        )

        rr = float(
            np.mean(
                raw_rough
            )
        )

        sr = float(
            np.mean(
                smooth_rough
            )
        )

        reduction = (
            100.0 * (1.0 - sr / rr)
            if rr > 0
            else 0.0
        )

        print(
            f"{bw:8.2f} | "
            f"{mean_mae:10.4f} | "
            f"{p95_change:13.4f} | "
            f"{max_change:13.4f} | "
            f"{rr:10.4f} | "
            f"{sr:12.4f} | "
            f"{reduction:14.2f}%"
        )

    # ------------------------------------------------------------
    # Compact overview, useful for pasting back.
    # ------------------------------------------------------------
    print()
    print("#" * 104)
    print("COMPACT SERIES SUMMARY")
    print("#" * 104)

    print(
        f"{'SERIES':32s} | "
        f"{'R':>4s} | "
        f"{'GB':>9s} | "
        f"{'FINAL ACC':>10s} | "
        f"{'RAW ROUGHNESS':>13s}"
    )

    print("-" * 104)

    for s in series:

        print(
            f"{s['name'][:32]:32s} | "
            f"{len(s['y']):4d} | "
            f"{s['x'][-1]:9.3f} | "
            f"{s['y'][-1]:9.3f}% | "
            f"{s['raw_roughness']:13.4f}"
        )

    print()
    print(
        "END OF DIAGNOSTIC — paste this terminal output back into ChatGPT."
    )


if __name__ == "__main__":
    main()
