"""Run-directory and logging helpers."""

from __future__ import annotations

import csv
import json
import logging
from importlib.metadata import PackageNotFoundError, version
import platform
import sys
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict

import numpy as np
import torch
import yaml

from bayesfl.config import ExperimentConfig


def create_run_paths(cfg: ExperimentConfig) -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{cfg.run_name}_{stamp}"
    run_dir = Path(cfg.output.outputs_dir).resolve() / run_id
    log_dir = Path(cfg.output.logs_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    for sub in ("metrics", "posterior", "checkpoints", "reliability", "plots"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, log_dir / f"{run_id}.log"


def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("bayesfl")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def save_resolved_config(cfg: ExperimentConfig, run_dir: Path) -> None:
    with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg.to_dict(), handle, sort_keys=False)


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def save_environment(run_dir: Path) -> None:
    gpu = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    payload = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu,
        "numpy": np.__version__,
        "torchvision": _package_version("torchvision"),
        "flwr": _package_version("flwr"),
        "ray": _package_version("ray"),
        "bayesian_torch": _package_version("bayesian-torch"),
    }
    with (run_dir / "environment.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


class CsvRecorder:
    """Append dictionaries to CSV while allowing columns to expand safely."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._rows: list[Dict[str, Any]] = []
        self._fields: list[str] = []

    def append(self, row: Dict[str, Any]) -> None:
        clean = {str(k): _scalar(v) for k, v in row.items()}
        with self._lock:
            self._rows.append(clean)
            for key in clean:
                if key not in self._fields:
                    self._fields.append(key)
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self._fields)
                writer.writeheader()
                writer.writerows(self._rows)


def _scalar(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if torch.is_tensor(value) and value.numel() == 1:
        return value.item()
    return value
