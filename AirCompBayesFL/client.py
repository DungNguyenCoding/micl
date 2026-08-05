"""Flower virtual client implementation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch

from bayes_vi import BayesianVITrainer
from config import SimulationConfig
from dataset import load_client_loader
from deterministic import train_deterministic
from models import build_model
from runtime_utils import release_cuda_memory, resolve_device, should_pin_memory
from serialization import ParameterLayout, initial_model_vector


class AirCompNumPyClient(fl.client.NumPyClient):
    """Flower client for deterministic and Bayesian local training.

    Ray can reuse worker processes across virtual clients.  All returned model
    data are converted to NumPy before ``fit`` exits, and CUDA cache cleanup is
    performed in a ``finally`` block to reduce memory fragmentation on a single
    laptop GPU.
    """

    def __init__(
        self,
        client_id: int,
        method: str,
        config: SimulationConfig,
        partition_path: str,
        run_seed: int,
        state_dir: str,
    ) -> None:
        self.client_id = int(client_id)
        self.method = method.lower()
        self.config = config
        self.partition_path = partition_path
        self.run_seed = int(run_seed)
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        torch.set_num_threads(int(config.runtime.torch_num_threads))
        self.device = resolve_device(config.runtime.client_device)
        if self.device.type == "cuda" and self.device.index is not None:
            torch.cuda.set_device(self.device.index)

        self.model = build_model(config.model.name, config.model.num_classes)
        self.layout = ParameterLayout(self.model)
        self.dimension = self.layout.total_numel

    def _initial_parameters(self) -> List[np.ndarray]:
        model_vector = initial_model_vector(self.model, self.run_seed)
        if self.method == "proposed":
            precision = np.full(
                self.dimension,
                1.0 / (self.config.model.initial_prior_std**2),
                dtype=np.float32,
            )
            return [model_vector, precision]
        if self.method == "scaffold":
            return [model_vector, np.zeros(self.dimension, dtype=np.float32)]
        return [model_vector]

    def get_parameters(self, config: Dict[str, fl.common.Scalar]) -> List[np.ndarray]:
        del config
        return self._initial_parameters()

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, fl.common.Scalar],
    ) -> Tuple[List[np.ndarray], int, Dict[str, fl.common.Scalar]]:
        try:
            return self._fit_impl(parameters, config)
        finally:
            if self.config.runtime.cleanup_cuda_after_fit:
                release_cuda_memory(self.device)

    def _fit_impl(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, fl.common.Scalar],
    ) -> Tuple[List[np.ndarray], int, Dict[str, fl.common.Scalar]]:
        server_round = int(config.get("server_round", 0))
        round_seed = self.run_seed + 100_003 * server_round + self.client_id
        loader, metadata = load_client_loader(
            self.config.data,
            self.config.training,
            self.partition_path,
            self.client_id,
            shuffle_seed=round_seed,
            pin_memory=should_pin_memory(self.config.data.pin_memory, self.device),
        )
        num_examples = int(metadata["num_examples"])

        if self.method == "proposed":
            if len(parameters) != 2:
                raise ValueError("Proposed method expects [global_mean, global_precision]")
            trainer = BayesianVITrainer(
                self.model,
                self.layout,
                self.config.model,
                self.config.training,
                self.device,
            )
            result = trainer.fit(
                np.asarray(parameters[0], dtype=np.float32),
                np.asarray(parameters[1], dtype=np.float32),
                loader,
                seed=round_seed,
            )
            metrics: Dict[str, fl.common.Scalar] = {
                "client_id": self.client_id,
                "distance_m": float(metadata["distance_m"]),
                "train_loss": float(result.average_loss),
                "phase1_loss": float(result.phase1_loss),
                "phase2_loss": float(result.phase2_loss),
                "local_steps": int(len(loader) * self.config.training.local_epochs),
                "device": str(self.device),
            }
            return [result.mean, result.precision], num_examples, metrics

        if self.method in {"fedavg", "fedprox"}:
            if len(parameters) != 1:
                raise ValueError(f"{self.method} expects one global model vector")
            result = train_deterministic(
                model=self.model,
                layout=self.layout,
                global_vector=np.asarray(parameters[0], dtype=np.float32),
                loader=loader,
                train_cfg=self.config.training,
                device=self.device,
                method=self.method,
                seed=round_seed,
            )
            metrics = {
                "client_id": self.client_id,
                "distance_m": float(metadata["distance_m"]),
                "train_loss": float(result.average_loss),
                "phase1_loss": 0.0,
                "phase2_loss": 0.0,
                "local_steps": int(result.local_steps),
                "device": str(self.device),
            }
            return [result.model_vector], num_examples, metrics

        if self.method == "scaffold":
            if len(parameters) != 2:
                raise ValueError("SCAFFOLD expects [global_model, global_control]")
            client_control = self._load_scaffold_control()
            result = train_deterministic(
                model=self.model,
                layout=self.layout,
                global_vector=np.asarray(parameters[0], dtype=np.float32),
                loader=loader,
                train_cfg=self.config.training,
                device=self.device,
                method="scaffold",
                seed=round_seed,
                global_control=np.asarray(parameters[1], dtype=np.float32),
                client_control=client_control,
            )
            assert result.new_client_control is not None
            assert result.control_delta is not None
            self._save_scaffold_control(result.new_client_control)
            metrics = {
                "client_id": self.client_id,
                "distance_m": float(metadata["distance_m"]),
                "train_loss": float(result.average_loss),
                "phase1_loss": 0.0,
                "phase2_loss": 0.0,
                "local_steps": int(result.local_steps),
                "device": str(self.device),
            }
            return [result.model_vector, result.control_delta], num_examples, metrics

        raise ValueError(f"Unknown method: {self.method}")

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, fl.common.Scalar],
    ) -> Tuple[float, int, Dict[str, fl.common.Scalar]]:
        del parameters, config
        # Central evaluation is used to reproduce the paper figures.
        return 0.0, 0, {}

    @property
    def scaffold_state_path(self) -> Path:
        return self.state_dir / f"client_{self.client_id:05d}_control.npy"

    def _load_scaffold_control(self) -> np.ndarray:
        path = self.scaffold_state_path
        if not path.exists():
            return np.zeros(self.dimension, dtype=np.float32)
        value = np.load(path)
        if value.shape != (self.dimension,):
            raise ValueError(f"Invalid SCAFFOLD state in {path}: {value.shape}")
        return value.astype(np.float32)

    def _save_scaffold_control(self, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=np.float32)
        fd, tmp_name = tempfile.mkstemp(
            prefix=self.scaffold_state_path.name,
            suffix=".tmp.npy",
            dir=self.state_dir,
        )
        os.close(fd)
        try:
            np.save(tmp_name, value)
            os.replace(tmp_name, self.scaffold_state_path)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
