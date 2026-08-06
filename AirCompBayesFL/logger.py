"""Append-only CSV loggers used by the server strategy."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np


_BASE_AIRCOMP_FIELDS = [
    "nmse",
    "distortion_nmse",
    "clipped_fraction",
    "average_symbol_power_watts",
    "maximum_symbol_power_watts",
    "noise_l2",
    "ideal_l2",
    "received_l2",
    "delta_bar",
    "retained_magnitude_ratio",
    "distorted_to_ideal_norm_ratio",
]


def _prefixed(prefix: str) -> list[str]:
    return [f"{prefix}_{name}" for name in _BASE_AIRCOMP_FIELDS]


METRIC_FIELDS = [
    "run_id",
    "experiment",
    "condition",
    "method",
    "realization",
    "seed",
    "round",
    "logical_round",
    "physical_round",
    "phase",
    "num_clients",
    "labels_per_client",
    "mean_samples_per_client",
    "power_dbm",
    "noise_dbm",
    "num_subchannels",
    "path_loss_exponent",
    "path_loss_reference_m",
    "gamma_db",
    "accuracy",
    "nll",
    "ece",
    "train_loss",
    "phase1_train_loss",
    "phase2_train_loss",
    "posterior_variance",
    "posterior_precision_mean",
    "posterior_precision_min",
    "posterior_precision_max",
    "posterior_precision_std",
    "posterior_precision_offset_l2",
    "posterior_precision_offset_max_abs",
    "channel_uses_round",
    "channel_uses_cumulative",
    "ofdm_symbols_round",
    "ofdm_symbols_cumulative",
    *_prefixed("aircomp"),
    *_prefixed("precision_aircomp"),
    *_prefixed("mean_aircomp"),
    "wall_time_sec",
]

RELIABILITY_FIELDS = [
    "run_id",
    "experiment",
    "condition",
    "method",
    "realization",
    "round",
    "logical_round",
    "physical_round",
    "phase",
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
    "logical_round",
    "physical_round",
    "phase",
    "client_id",
    "num_examples",
    "distance_m",
    "train_loss",
    "phase1_loss",
    "phase2_loss",
    "local_steps",
    "local_precision_mean",
    "local_precision_min",
    "local_precision_max",
    "local_precision_delta_l2",
    "local_precision_delta_max_abs",
    "local_precision_changed_fraction",
    "local_precision_gradient_l2_mean",
    "local_precision_gradient_max_abs",
    "local_nu_l2",
    "local_implied_mean_l2",
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
