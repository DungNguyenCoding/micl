from __future__ import annotations

import csv
from pathlib import Path

import yaml

from bayesfl.utils import latest_matching_run, plot_accuracy_comparison


def _write_run(
    root: Path,
    name: str,
    *,
    method: str,
    model: str,
    values: list[tuple[int, float]],
) -> Path:
    run = root / name
    metrics = run / "metrics"
    metrics.mkdir(parents=True)

    with (run / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"method": method, "model": {"name": model}}, handle)

    with (metrics / "global_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        if method == "fola":
            fieldnames = ["round", "accuracy", "fola_mean_accuracy"]
        else:
            fieldnames = ["round", "accuracy"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rnd, acc in values:
            row = {"round": rnd, "accuracy": acc}
            if method == "fola":
                row["fola_mean_accuracy"] = acc + 0.01
            writer.writerow(row)

    return run


def test_plot_accuracy_comparison(tmp_path: Path) -> None:
    fed = _write_run(
        tmp_path,
        "run_fedavg",
        method="fedavg",
        model="paper_basiccnn",
        values=[(0, 0.1), (1, 0.2)],
    )
    fola = _write_run(
        tmp_path,
        "run_fola",
        method="fola",
        model="resnet56_gn8",
        values=[(0, 0.1), (1, 0.25)],
    )

    out = tmp_path / "plots"
    paths = plot_accuracy_comparison(
        [fed, fola],
        output_dir=out,
        output_stem="comparison",
    )

    assert [p.suffix for p in paths] == [".png", ".pdf", ".csv"]
    assert all(p.exists() for p in paths)

    with (out / "comparison.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert {row["label"] for row in rows} == {
        "FedAvg — BasicCNN",
        "FOLA — ResNet-56",
    }

    fola_r1 = next(
        row for row in rows if row["label"] == "FOLA — ResNet-56" and row["round"] == "1"
    )
    assert float(fola_r1["accuracy_percent"]) == 26.0


def test_latest_matching_run(tmp_path: Path) -> None:
    older = tmp_path / "run_test_old"
    newer = tmp_path / "run_test_new"
    older.mkdir()
    newer.mkdir()

    # Explicit mtimes avoid filesystem timestamp-resolution assumptions.
    import os

    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    assert latest_matching_run(str(tmp_path / "run_test_*")) == newer
