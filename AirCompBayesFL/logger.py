"""Append-only CSV loggers used by the server strategy."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np


METRIC_FIELDS = [
    "run_id",
    "experiment",
    "condition",
    "method",
    "realization",
    "seed",
    "round",
    "num_clients",
    "labels_per_client",
    "mean_samples_per_client",
    "power_dbm",
    "noise_dbm",
    "accuracy",
    "nll",
    "ece",
    "train_loss",
    "posterior_variance",
    "channel_uses_round",
    "channel_uses_cumulative",
    "ofdm_symbols_round",
    "ofdm_symbols_cumulative",
    "aircomp_nmse",
    "aircomp_distortion_nmse",
    "aircomp_clipped_fraction",
    "aircomp_average_symbol_power_watts",
    "aircomp_maximum_symbol_power_watts",
    "aircomp_noise_l2",
    "wall_time_sec",
]

RELIABILITY_FIELDS = [
    "run_id",
    "experiment",
    "condition",
    "method",
    "realization",
    "round",
    "bin",
    "lower",
    "upper",
    "count",
    "confidence",
    "accuracy",
]

CLIENT_FIELDS = [
    "run_id",
    "round",
    "client_id",
    "num_examples",
    "distance_m",
    "train_loss",
    "phase1_loss",
    "phase2_loss",
    "local_steps",
]


class CsvAppender:
    def __init__(self, path: str | Path, fields: Iterable[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fields = list(fields)

    def append(self, row: Mapping[str, object]) -> None:
        exists = self.path.exists() and self.path.stat().st_size > 0
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow({field: row.get(field, "") for field in self.fields})


class RunLogger:
    def __init__(
        self,
        output_dir: str | Path,
        metrics_filename: str,
        reliability_filename: str,
        clients_filename: str,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.metrics = CsvAppender(output_dir / metrics_filename, METRIC_FIELDS)
        self.reliability = CsvAppender(
            output_dir / reliability_filename, RELIABILITY_FIELDS
        )
        self.clients = CsvAppender(output_dir / clients_filename, CLIENT_FIELDS)

    def log_reliability(self, base: Dict[str, object], evaluation: object) -> None:
        for index in range(len(evaluation.bin_count)):
            row = dict(base)
            row.update(
                {
                    "bin": index,
                    "lower": float(evaluation.bin_lower[index]),
                    "upper": float(evaluation.bin_upper[index]),
                    "count": int(evaluation.bin_count[index]),
                    "confidence": float(evaluation.bin_confidence[index]),
                    "accuracy": float(evaluation.bin_accuracy[index]),
                }
            )
            self.reliability.append(row)

    def save_checkpoint(
        self,
        run_id: str,
        parameters: list[np.ndarray],
        metadata: Mapping[str, object],
    ) -> Path:
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"{run_id}.npz"
        arrays = {f"parameter_{index}": value for index, value in enumerate(parameters)}
        arrays["metadata_json"] = np.asarray(json.dumps(dict(metadata)))
        np.savez_compressed(path, **arrays)
        return path
