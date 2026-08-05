"""Flower/local virtual client implementation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch

from bayes_vi import BayesianVITrainer
from bayesian_protocol import NATURAL_MEAN_PHASE, PRECISION_PHASE
from config import SimulationConfig
from dataset import load_client_loader
from deterministic import train_deterministic
from models import build_model
from runtime_utils import release_cuda_memory, resolve_device, should_pin_memory
from serialization import ParameterLayout, initial_model_vector


class AirCompNumPyClient(fl.client.NumPyClient):
    """Virtual client shared by the Flower/Ray and local backends.

    The proposed method deliberately needs two calls per logical FL round.  The
    precision phase saves ``rho_{t,k}`` in the run's client-state directory;
    the natural-mean phase loads that client-specific value after the server has
    broadcast ``rho_{t+1}``.  This preserves Algorithm 1 even when Flower uses
    different actor instances for the two phases on a single simulation host.
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
            if self.device.type == "cuda":
                try:
                    self.model.to("cpu")
                except Exception:
                    pass
            if self.config.runtime.cleanup_cuda_after_fit:
                release_cuda_memory(self.device)

    def _fit_impl(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, fl.common.Scalar],
    ) -> Tuple[List[np.ndarray], int, Dict[str, fl.common.Scalar]]:
        logical_round = int(config.get("server_round", config.get("logical_round", 0)))
        physical_round = int(config.get("physical_round", logical_round))
        phase = str(config.get("phase", "model"))
        base_seed = self.run_seed + 100_003 * logical_round + self.client_id
        phase_seed = base_seed + (50_000_021 if phase == NATURAL_MEAN_PHASE else 0)

        loader, metadata = load_client_loader(
            self.config.data,
            self.config.training,
            self.partition_path,
            self.client_id,
            shuffle_seed=base_seed,
            pin_memory=should_pin_memory(self.config.data.pin_memory, self.device),
        )
        num_examples = int(metadata["num_examples"])

        if self.method == "proposed":
            return self._fit_proposed_phase(
                parameters=parameters,
                phase=phase,
                logical_round=logical_round,
                physical_round=physical_round,
                loader=loader,
                metadata=metadata,
                num_examples=num_examples,
                phase_seed=phase_seed,
            )

        if phase != "model":
            raise ValueError(f"{self.method} received unexpected phase {phase!r}")

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
                seed=phase_seed,
            )
            metrics = self._base_metrics(
                metadata,
                logical_round,
                physical_round,
                phase,
                train_loss=float(result.average_loss),
                phase1_loss=0.0,
                phase2_loss=0.0,
                local_steps=int(result.local_steps),
            )
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
                seed=phase_seed,
                global_control=np.asarray(parameters[1], dtype=np.float32),
                client_control=client_control,
            )
            assert result.new_client_control is not None
            assert result.control_delta is not None
            self._save_scaffold_control(result.new_client_control)
            metrics = self._base_metrics(
                metadata,
                logical_round,
                physical_round,
                phase,
                train_loss=float(result.average_loss),
                phase1_loss=0.0,
                phase2_loss=0.0,
                local_steps=int(result.local_steps),
            )
            return [result.model_vector, result.control_delta], num_examples, metrics

        raise ValueError(f"Unknown method: {self.method}")

    def _fit_proposed_phase(
        self,
        *,
        parameters: List[np.ndarray],
        phase: str,
        logical_round: int,
        physical_round: int,
        loader,
        metadata: Dict[str, object],
        num_examples: int,
        phase_seed: int,
    ) -> Tuple[List[np.ndarray], int, Dict[str, fl.common.Scalar]]:
        expected_count = 2 if phase == PRECISION_PHASE else 3
        if len(parameters) != expected_count:
            if phase == PRECISION_PHASE:
                expected = "[global_mean, round_start_global_precision]"
            else:
                expected = (
                    "[global_mean, next_global_precision, "
                    "round_start_global_precision]"
                )
            raise ValueError(
                f"Proposed phase {phase!r} expects {expected}; "
                f"received {len(parameters)} array(s)"
            )
        global_mean = np.asarray(parameters[0], dtype=np.float32)
        global_precision = np.asarray(parameters[1], dtype=np.float32)
        trainer = BayesianVITrainer(
            self.model,
            self.layout,
            self.config.model,
            self.config.training,
            self.device,
        )

        if phase == PRECISION_PHASE:
            result = trainer.train_precision_phase(
                global_mean=global_mean,
                global_precision=global_precision,
                loader=loader,
                seed=phase_seed,
            )
            self._save_proposed_precision(logical_round, result.precision)
            metrics = self._base_metrics(
                metadata,
                logical_round,
                physical_round,
                phase,
                train_loss=float(result.average_loss),
                phase1_loss=float(result.average_loss),
                phase2_loss=0.0,
                local_steps=int(result.local_steps),
            )
            metrics.update(
                {
                    "local_precision_mean": float(np.mean(result.precision)),
                    "local_precision_min": float(np.min(result.precision)),
                    "local_precision_max": float(np.max(result.precision)),
                    "local_nu_l2": 0.0,
                    "local_implied_mean_l2": 0.0,
                }
            )
            # The server aggregates rho and retains the global mean.
            return [result.precision], num_examples, metrics

        if phase == NATURAL_MEAN_PHASE:
            local_precision = self._load_proposed_precision(logical_round)
            prior_global_precision = np.asarray(parameters[2], dtype=np.float32)
            result = trainer.train_natural_mean_phase(
                global_mean=global_mean,
                prior_global_precision=prior_global_precision,
                next_global_precision=global_precision,
                local_precision=local_precision,
                loader=loader,
                seed=phase_seed,
            )
            metrics = self._base_metrics(
                metadata,
                logical_round,
                physical_round,
                phase,
                train_loss=float(result.average_loss),
                phase1_loss=0.0,
                phase2_loss=float(result.average_loss),
                local_steps=int(result.local_steps),
            )
            metrics.update(
                {
                    "local_precision_mean": float(np.mean(local_precision)),
                    "local_precision_min": float(np.min(local_precision)),
                    "local_precision_max": float(np.max(local_precision)),
                    "local_nu_l2": float(np.linalg.norm(result.nu)),
                    "local_implied_mean_l2": float(np.linalg.norm(result.implied_mean)),
                }
            )
            if self.config.runtime.cleanup_phase_state:
                self._remove_proposed_precision(logical_round)
            # The server aggregates nu; it does not aggregate implied local means.
            return [result.nu], num_examples, metrics

        raise ValueError(
            "Proposed method requires phase='precision' or phase='natural_mean'; "
            f"received {phase!r}"
        )

    def _base_metrics(
        self,
        metadata: Dict[str, object],
        logical_round: int,
        physical_round: int,
        phase: str,
        *,
        train_loss: float,
        phase1_loss: float,
        phase2_loss: float,
        local_steps: int,
    ) -> Dict[str, fl.common.Scalar]:
        return {
            "client_id": self.client_id,
            "distance_m": float(metadata["distance_m"]),
            "logical_round": int(logical_round),
            "physical_round": int(physical_round),
            "phase": str(phase),
            "train_loss": float(train_loss),
            "phase1_loss": float(phase1_loss),
            "phase2_loss": float(phase2_loss),
            "local_steps": int(local_steps),
            "device": str(self.device),
        }

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, fl.common.Scalar],
    ) -> Tuple[float, int, Dict[str, fl.common.Scalar]]:
        del parameters, config
        return 0.0, 0, {}

    @property
    def scaffold_state_path(self) -> Path:
        return self.state_dir / f"client_{self.client_id:05d}_control.npy"

    def proposed_precision_state_path(self, logical_round: int) -> Path:
        return self.state_dir / (
            f"client_{self.client_id:05d}_round_{int(logical_round):06d}_precision.npy"
        )

    def _load_scaffold_control(self) -> np.ndarray:
        path = self.scaffold_state_path
        if not path.exists():
            return np.zeros(self.dimension, dtype=np.float32)
        value = np.load(path)
        if value.shape != (self.dimension,):
            raise ValueError(f"Invalid SCAFFOLD state in {path}: {value.shape}")
        return value.astype(np.float32)

    def _save_scaffold_control(self, value: np.ndarray) -> None:
        self._atomic_save(self.scaffold_state_path, value)

    def _save_proposed_precision(self, logical_round: int, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        if value.shape != (self.dimension,):
            raise ValueError(
                f"Local precision has shape {value.shape}; expected {(self.dimension,)}"
            )
        self._atomic_save(self.proposed_precision_state_path(logical_round), value)

    def _load_proposed_precision(self, logical_round: int) -> np.ndarray:
        path = self.proposed_precision_state_path(logical_round)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing phase-1 precision state for client {self.client_id}, "
                f"logical round {logical_round}: {path}. The natural-mean phase "
                "cannot run before server-side precision aggregation."
            )
        value = np.load(path)
        if value.shape != (self.dimension,):
            raise ValueError(f"Invalid proposed phase state in {path}: {value.shape}")
        return value.astype(np.float32)

    def _remove_proposed_precision(self, logical_round: int) -> None:
        self.proposed_precision_state_path(logical_round).unlink(missing_ok=True)

    def _atomic_save(self, path: Path, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=np.float32)
        fd, tmp_name = tempfile.mkstemp(
            prefix=path.name,
            suffix=".tmp.npy",
            dir=path.parent,
        )
        os.close(fd)
        try:
            np.save(tmp_name, value)
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
