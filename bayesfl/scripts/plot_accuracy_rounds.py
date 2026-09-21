from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
PLOT_DIR = OUTPUTS / "plots" / "accuracy_round_final"


# ============================================================
# General helpers
# ============================================================

def strip_timestamp(name: str) -> str:
    return re.sub(
        r"_20\d{6}_\d{6}$",
        "",
        name,
    )


def infer_seed_from_name(name: str):
    m = re.search(
        r"(?:^|_)seed(\d+)(?:_|$)",
        name,
        flags=re.I,
    )
    if m:
        return int(m.group(1))
    return None


def flatten(obj, prefix=""):
    out = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten(v, key))

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{prefix}.{i}"
            out.update(flatten(v, key))

    else:
        out[prefix] = obj

    return out


def read_config_metadata(run_dir: Path):
    """
    Try to recover metadata from config files saved inside a run.

    The code deliberately accepts several possible filenames so it also
    works with older experiments.
    """

    candidates = []

    preferred = [
        "config.yaml",
        "config.yml",
        "resolved_config.yaml",
        "resolved_config.yml",
        "run_config.yaml",
        "run_config.yml",
        "config.json",
        "resolved_config.json",
    ]

    for name in preferred:
        p = run_dir / name
        if p.exists():
            candidates.append(p)

    # Limited fallback search.
    if not candidates:
        candidates.extend(
            list(run_dir.glob("*.yaml"))
            + list(run_dir.glob("*.yml"))
            + list(run_dir.glob("*.json"))
        )

    for path in candidates:
        try:
            if path.suffix in {".yaml", ".yml"}:
                data = yaml.safe_load(
                    path.read_text(encoding="utf-8")
                )
            else:
                data = json.loads(
                    path.read_text(encoding="utf-8")
                )

            if isinstance(data, dict):
                return data, path

        except Exception:
            pass

    return {}, None


def pick_flat(flat, suffixes):
    suffixes = tuple(
        s.lower()
        for s in suffixes
    )

    for key, value in flat.items():
        kl = key.lower()

        if any(
            kl == suffix
            or kl.endswith("." + suffix)
            for suffix in suffixes
        ):
            return value

    return None


def metadata(run_dir: Path):
    name = run_dir.name
    low = name.lower()

    cfg, cfg_path = read_config_metadata(run_dir)
    flat = flatten(cfg)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    dataset = pick_flat(
        flat,
        ["dataset", "dataset_name"],
    )

    if dataset is not None:
        dataset = str(dataset).lower()

    if dataset not in {"mnist", "cifar10", "cifar-10"}:
        if "mnist" in low:
            dataset = "mnist"
        elif "cifar10" in low or "cifar_" in low:
            dataset = "cifar10"

    if dataset == "cifar-10":
        dataset = "cifar10"

    # --------------------------------------------------------
    # Method
    # --------------------------------------------------------
    method = pick_flat(
        flat,
        ["method", "algorithm"],
    )

    if method is not None:
        method = str(method).lower()

    if "fola" in low:
        method = "fola"
    elif "fedavg" in low:
        method = "fedavg"

    # --------------------------------------------------------
    # Alpha
    # --------------------------------------------------------
    alpha = pick_flat(
        flat,
        [
            "dirichlet_alpha",
            "partition_alpha",
        ],
    )

    try:
        alpha = float(alpha)
    except Exception:
        alpha = None

    # Folder fallback.
    if alpha is None:
        if re.search(r"(?:^|_)a0p01(?:_|$)", low):
            alpha = 0.01
        elif re.search(r"(?:^|_)a0p1(?:_|$)", low):
            alpha = 0.1

    # --------------------------------------------------------
    # Fraction
    # --------------------------------------------------------
    fraction = pick_flat(
        flat,
        ["local_data_fraction"],
    )

    try:
        fraction = float(fraction)
    except Exception:
        fraction = None

    # Full-data naming convention.
    if fraction is None:
        if (
            "_full_" in low
            or "_f1_" in low
            or low.endswith("_f1")
        ):
            fraction = 1.0

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------
    seed = infer_seed_from_name(name)

    if seed is None:
        val = pick_flat(
            flat,
            [
                "runtime.seed",
                "seed",
            ],
        )
        try:
            seed = int(val)
        except Exception:
            seed = None

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = pick_flat(
        flat,
        [
            "model.name",
            "model_name",
        ],
    )

    if model is not None:
        model = str(model).lower()

    if model is None:
        if "r56" in low or "resnet56" in low:
            model = "resnet56_gn8"
        elif "basiccnn" in low:
            model = "basiccnn"
        elif "mlp" in low:
            model = "mlp"

    return {
        "dataset": dataset,
        "method": method,
        "alpha": alpha,
        "fraction": fraction,
        "seed": seed,
        "model": model,
        "config_path": cfg_path,
    }


# ============================================================
# Metric loading
# ============================================================

def find_global_metrics(run_dir: Path):
    candidates = [
        run_dir / "metrics" / "global_metrics.csv",
        run_dir / "global_metrics.csv",
    ]

    for p in candidates:
        if p.exists():
            return p

    matches = list(
        run_dir.rglob("global_metrics.csv")
    )

    if matches:
        return matches[0]

    return None


def read_accuracy_curve(
    run_dir: Path,
    method: str,
):
    path = find_global_metrics(run_dir)

    if path is None:
        raise FileNotFoundError(
            f"No global_metrics.csv: {run_dir}"
        )

    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(
            f"Empty metrics file: {path}"
        )

    headers = set(rows[0])

    if method == "fola":
        candidate_keys = [
            "fola_mean_accuracy",
            "accuracy",
            "global_accuracy",
            "test_accuracy",
        ]
    else:
        candidate_keys = [
            "accuracy",
            "global_accuracy",
            "test_accuracy",
        ]

    acc_key = next(
        (
            key
            for key in candidate_keys
            if key in headers
        ),
        None,
    )

    if acc_key is None:
        raise RuntimeError(
            f"No accuracy column in {path}. "
            f"Columns={sorted(headers)}"
        )

    points = {}

    for row in rows:
        try:
            rnd = int(
                float(row["round"])
            )
            acc = float(
                row[acc_key]
            )
        except Exception:
            continue

        # Convert percentages to fractions only if necessary.
        if acc > 1.5:
            acc = acc / 100.0

        points[rnd] = acc

    if not points:
        raise RuntimeError(
            f"No valid accuracy rows: {path}"
        )

    return points


# ============================================================
# Run discovery
# ============================================================

def discover_runs():
    records = []

    for run_dir in OUTPUTS.iterdir():
        if not run_dir.is_dir():
            continue

        if run_dir.name in {
            "plots",
            "partitions",
        }:
            continue

        metrics = find_global_metrics(
            run_dir
        )

        if metrics is None:
            continue

        info = metadata(
            run_dir
        )

        if (
            info["dataset"] is None
            or info["method"] not in {
                "fedavg",
                "fola",
            }
        ):
            continue

        records.append(
            {
                "path": run_dir,
                "name": run_dir.name,
                **info,
            }
        )

    return records


def close(a, b, tol=1e-8):
    return (
        a is not None
        and abs(float(a) - float(b)) <= tol
    )


def choose_mnist_runs(
    records,
    alpha,
    method,
):
    """
    Pick the current final full-data MNIST multi-seed group.

    We favor:
      * MNIST
      * requested alpha/method
      * full-data
      * run_* rather than debug_*
      * groups with the largest number of distinct seeds
      * 50-round completed runs
    """

    candidates = []

    for r in records:
        if r["dataset"] != "mnist":
            continue

        if r["method"] != method:
            continue

        if not close(
            r["alpha"],
            alpha,
        ):
            continue

        # Full data.
        if (
            r["fraction"] is not None
            and not close(r["fraction"], 1.0)
        ):
            continue

        try:
            curve = read_accuracy_curve(
                r["path"],
                method,
            )
        except Exception:
            continue

        if max(curve) < 50:
            continue

        candidates.append(
            (r, curve)
        )

    if not candidates:
        return []

    # Prefer real/final run names.
    run_candidates = [
        x
        for x in candidates
        if x[0]["name"].startswith("run_")
    ]

    if run_candidates:
        candidates = run_candidates

    # Ignore historical paperenv experiments for the final MNIST plot.
    non_paperenv = [
        x
        for x in candidates
        if "paperenv" not in x[0]["name"].lower()
    ]

    if non_paperenv:
        candidates = non_paperenv

    # Group by normalized run family.
    groups = defaultdict(list)

    for r, curve in candidates:
        family = strip_timestamp(
            r["name"]
        )

        family = re.sub(
            r"_seed\d+(?=_|$)",
            "_seedX",
            family,
            flags=re.I,
        )

        groups[family].append(
            (r, curve)
        )

    # Family with most distinct explicit seeds.
    def group_score(item):
        family, xs = item

        seeds = {
            x[0]["seed"]
            for x in xs
            if x[0]["seed"] is not None
        }

        return (
            len(seeds),
            len(xs),
            max(
                x[0]["path"].stat().st_mtime
                for x in xs
            ),
        )

    best_family, best = max(
        groups.items(),
        key=group_score,
    )

    # If multiple timestamps for a seed exist, keep latest.
    per_seed = {}

    for r, curve in best:
        seed = r["seed"]

        # Allow runs with no explicit seed only as a single-seed fallback.
        key = seed if seed is not None else r["name"]

        old = per_seed.get(key)

        if (
            old is None
            or r["path"].stat().st_mtime
            > old[0]["path"].stat().st_mtime
        ):
            per_seed[key] = (
                r,
                curve,
            )

    return list(
        per_seed.values()
    )


def choose_cifar_runs(
    records,
    alpha,
    method,
):
    """
    For current CIFAR figures, explicitly use the locked
    independent replication seeds 1-4.
    """

    atag = (
        "a0p1"
        if close(alpha, 0.1)
        else "a0p01"
    )

    prefix = (
        f"run_cifar_r56_{atag}_"
        "multiseed_"
    )

    candidates = []

    for r in records:
        low = r["name"].lower()

        if not low.startswith(prefix):
            continue

        if r["method"] != method:
            continue

        if "_full_" not in low:
            continue

        if r["seed"] not in {
            1, 2, 3, 4,
        }:
            continue

        try:
            curve = read_accuracy_curve(
                r["path"],
                method,
            )
        except Exception:
            continue

        if max(curve) < 50:
            continue

        candidates.append(
            (r, curve)
        )

    per_seed = {}

    for r, curve in candidates:
        seed = r["seed"]

        old = per_seed.get(seed)

        if (
            old is None
            or r["path"].stat().st_mtime
            > old[0]["path"].stat().st_mtime
        ):
            per_seed[seed] = (
                r,
                curve,
            )

    return [
        per_seed[s]
        for s in sorted(per_seed)
    ]


# ============================================================
# Averaging
# ============================================================

def aggregate_curves(
    selected,
    max_round=50,
):
    if not selected:
        raise RuntimeError(
            "No runs selected."
        )

    common_rounds = set(
        range(1, max_round + 1)
    )

    for _, curve in selected:
        common_rounds &= set(
            curve.keys()
        )

    rounds = sorted(
        common_rounds
    )

    if not rounds:
        raise RuntimeError(
            "Selected runs have no common rounds."
        )

    matrix = np.asarray(
        [
            [
                curve[r]
                for r in rounds
            ]
            for _, curve in selected
        ],
        dtype=np.float64,
    )

    mean = matrix.mean(
        axis=0
    )

    if matrix.shape[0] > 1:
        sd = matrix.std(
            axis=0,
            ddof=1,
        )
    else:
        sd = np.zeros_like(
            mean
        )

    return (
        np.asarray(rounds),
        mean,
        sd,
        matrix,
    )


def print_selected(
    title,
    method,
    selected,
):
    print()
    print(
        f"{title} | {method}"
    )

    for r, _ in selected:
        print(
            f"  seed={r['seed']!s:<4} "
            f"{r['name']}"
        )


# ============================================================
# Plotting
# ============================================================

def make_plot(
    dataset_label,
    alpha,
    fed_selected,
    fola_selected,
    output_stem,
):
    fed_r, fed_m, fed_sd, fed_mat = (
        aggregate_curves(
            fed_selected,
            max_round=50,
        )
    )

    fola_r, fola_m, fola_sd, fola_mat = (
        aggregate_curves(
            fola_selected,
            max_round=50,
        )
    )

    common = sorted(
        set(fed_r.tolist())
        & set(fola_r.tolist())
    )

    fed_idx = {
        r: i
        for i, r in enumerate(
            fed_r.tolist()
        )
    }

    fola_idx = {
        r: i
        for i, r in enumerate(
            fola_r.tolist()
        )
    }

    x = np.asarray(
        common,
        dtype=int,
    )

    fm = np.asarray(
        [
            fed_m[fed_idx[r]]
            for r in common
        ]
    )

    fs = np.asarray(
        [
            fed_sd[fed_idx[r]]
            for r in common
        ]
    )

    om = np.asarray(
        [
            fola_m[fola_idx[r]]
            for r in common
        ]
    )

    osd = np.asarray(
        [
            fola_sd[fola_idx[r]]
            for r in common
        ]
    )

    fig, ax = plt.subplots(
        figsize=(7.2, 5.0)
    )

    fed_line = ax.plot(
        x,
        100.0 * fm,
        linewidth=2.0,
        label=(
            f"FedAvg "
            f"(n={len(fed_selected)})"
        ),
    )[0]

    fola_line = ax.plot(
        x,
        100.0 * om,
        linewidth=2.0,
        label=(
            f"FOLA "
            f"(n={len(fola_selected)})"
        ),
    )[0]

    if len(fed_selected) > 1:
        ax.fill_between(
            x,
            100.0 * (fm - fs),
            100.0 * (fm + fs),
            alpha=0.16,
            color=fed_line.get_color(),
            linewidth=0,
        )

    if len(fola_selected) > 1:
        ax.fill_between(
            x,
            100.0 * (om - osd),
            100.0 * (om + osd),
            alpha=0.16,
            color=fola_line.get_color(),
            linewidth=0,
        )

    ax.set_xlabel(
        "Communication round"
    )

    ax.set_ylabel(
        "Global test accuracy (%)"
    )

    ax.set_title(
        f"{dataset_label} — "
        rf"$\alpha={alpha:g}$"
    )

    ax.set_xlim(
        x.min(),
        x.max(),
    )

    # Data-dependent y range while retaining some visual margin.
    lower = min(
        np.min(100 * (fm - fs)),
        np.min(100 * (om - osd)),
    )

    upper = max(
        np.max(100 * (fm + fs)),
        np.max(100 * (om + osd)),
    )

    margin = max(
        2.0,
        0.07 * (upper - lower),
    )

    ax.set_ylim(
        max(0.0, lower - margin),
        min(100.0, upper + margin),
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend(
        frameon=False,
    )

    fig.tight_layout()

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    png = (
        PLOT_DIR /
        f"{output_stem}.png"
    )

    pdf = (
        PLOT_DIR /
        f"{output_stem}.pdf"
    )

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(fig)

    # Terminal summary.
    print()
    print(
        "=" * 88
    )
    print(
        f"{dataset_label} alpha={alpha:g}"
    )
    print(
        "=" * 88
    )

    print(
        f"FedAvg seeds/runs = "
        f"{len(fed_selected)}"
    )

    print(
        f"FOLA seeds/runs   = "
        f"{len(fola_selected)}"
    )

    print(
        f"FedAvg mean R50   = "
        f"{100*fm[-1]:.2f}%"
    )

    print(
        f"FOLA mean R50     = "
        f"{100*om[-1]:.2f}%"
    )

    print(
        f"FOLA-Fed R50 gap  = "
        f"{100*(om[-1]-fm[-1]):+.2f} pp"
    )

    print(
        f"PNG = {png}"
    )

    print(
        f"PDF = {pdf}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only print discovered runs.",
    )

    args = parser.parse_args()

    records = discover_runs()

    if args.list_only:
        for r in sorted(
            records,
            key=lambda x: x["name"],
        ):
            print(
                r["dataset"],
                r["alpha"],
                r["fraction"],
                r["seed"],
                r["method"],
                r["model"],
                r["name"],
                sep=" | ",
            )
        return

    cases = [
        (
            "MNIST",
            "mnist",
            0.1,
            "mnist_a0p1_accuracy_vs_round",
        ),
        (
            "MNIST",
            "mnist",
            0.01,
            "mnist_a0p01_accuracy_vs_round",
        ),
        (
            "CIFAR-10",
            "cifar10",
            0.1,
            "cifar10_a0p1_accuracy_vs_round",
        ),
        (
            "CIFAR-10",
            "cifar10",
            0.01,
            "cifar10_a0p01_accuracy_vs_round",
        ),
    ]

    for (
        label,
        dataset,
        alpha,
        stem,
    ) in cases:

        if dataset == "mnist":
            fed = choose_mnist_runs(
                records,
                alpha,
                "fedavg",
            )

            fola = choose_mnist_runs(
                records,
                alpha,
                "fola",
            )

        else:
            fed = choose_cifar_runs(
                records,
                alpha,
                "fedavg",
            )

            fola = choose_cifar_runs(
                records,
                alpha,
                "fola",
            )

        print_selected(
            f"{label} alpha={alpha:g}",
            "FedAvg",
            fed,
        )

        print_selected(
            f"{label} alpha={alpha:g}",
            "FOLA",
            fola,
        )

        if not fed or not fola:
            print()
            print(
                "ERROR: could not identify the complete "
                f"{label} alpha={alpha:g} run set."
            )
            print(
                "Run this for diagnosis:"
            )
            print(
                "  python scripts/"
                "plot_accuracy_rounds.py --list-only"
            )
            continue

        make_plot(
            dataset_label=label,
            alpha=alpha,
            fed_selected=fed,
            fola_selected=fola,
            output_stem=stem,
        )


if __name__ == "__main__":
    main()
