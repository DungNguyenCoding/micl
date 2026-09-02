"""Append-only CSV logging for the unified baseline."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


METRIC_FIELDS = [
    "run_id", "dataset", "model", "method", "seed", "round",
    "num_clients", "selected_clients", "client_fraction", "selected_client_ids",
    "partition", "dirichlet_alpha", "partition_total_samples", "partition_mean_size",
    "partition_min_size", "partition_max_size", "mean_classes_per_client",
    "local_epochs", "batch_size", "optimizer", "momentum", "weight_decay",
    "lr_scheduler", "lr_decay_rounds", "min_learning_rate", "learning_rate",
    "train_loss", "accuracy", "nll", "ece",
    "posterior_predictive_accuracy", "posterior_predictive_nll", "posterior_predictive_ece",
    "posterior_mean_accuracy", "posterior_mean_nll", "posterior_mean_ece",
    "bayesian_dimension", "deterministic_dimension", "model_dimension", "payload_scalars_per_client",
    "kl_weight_config", "kl_weight_resolved", "kl_weight_schedule", "kl_warmup_rounds",
    "lambda_scale_by_size", "mc_train", "mc_eval", "variance_floor_ratio",
    "posterior_sigma_mean", "posterior_sigma_min", "posterior_sigma_max",
    "global_state_update_l2", "upload_scalars_round", "upload_scalars_cumulative",
    "wall_time_sec",
]

CLIENT_FIELDS = [
    "run_id", "round", "client_id", "num_examples", "learning_rate",
    "train_loss", "ce_loss", "kl_sum", "local_steps",
    "kl_weight_base", "kl_weight_client", "kl_warmup_factor",
    "mu_update_l2", "rho_update_l2", "deterministic_update_l2",
    "sigma_mean", "sigma_min", "sigma_max", "variance_floor_clipped_fraction",
    "model_update_l2", "model_update_max_abs",
]

PARTICIPATION_FIELDS = ["run_id", "round", "client_id", "num_examples"]

RELIABILITY_FIELDS = [
    "run_id", "method", "round", "evaluation", "bin", "lower", "upper",
    "count", "confidence", "accuracy",
]


class CsvAppender:
    def __init__(self, path: str | Path, fields: Iterable[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fields = list(fields)

    def append(self, row: Mapping[str, object]) -> None:
        exists = self.path.exists() and self.path.stat().st_size > 0
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow({field: row.get(field, "") for field in self.fields})


class RunLogger:
    def __init__(self, output_dir, metrics_filename, clients_filename, reliability_filename, participation_filename):
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.metrics = CsvAppender(root / metrics_filename, METRIC_FIELDS)
        self.clients = CsvAppender(root / clients_filename, CLIENT_FIELDS)
        self.participation = CsvAppender(root / participation_filename, PARTICIPATION_FIELDS)
        self.reliability = CsvAppender(root / reliability_filename, RELIABILITY_FIELDS)
        self.output_dir = root

    def log_reliability(self, base: Mapping[str, object], evaluation, label: str) -> None:
        for i in range(len(evaluation.bin_count)):
            row = dict(base)
            row.update(
                {
                    "evaluation": label,
                    "bin": i,
                    "lower": float(evaluation.bin_lower[i]),
                    "upper": float(evaluation.bin_upper[i]),
                    "count": int(evaluation.bin_count[i]),
                    "confidence": float(evaluation.bin_confidence[i]),
                    "accuracy": float(evaluation.bin_accuracy[i]),
                }
            )
            self.reliability.append(row)

    def save_checkpoint(self, run_id: str, state: np.ndarray, metadata: Mapping[str, object]) -> Path:
        root = self.output_dir / "checkpoints"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{run_id}.npz"
        np.savez_compressed(
            path,
            state=np.asarray(state),
            metadata_json=np.asarray(json.dumps(dict(metadata))),
        )
        return path
