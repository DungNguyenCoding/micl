"""Flower/local virtual client implementation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch

from bayesian_backend import create_bayesian_trainer
from bayesian_protocol import NATURAL_MEAN_PHASE, PRECISION_PHASE
from config import SimulationConfig
from dataset import load_client_loader
from deterministic import train_deterministic
from models import build_model
from runtime_utils import release_cuda_memory, resolve_device, should_pin_memory
from serialization import ParameterLayout, initial_model_vector
from sparse_posterior import (
    bayesian_update_snr_score,
    full_mask_info,
    select_sparse_mask,
)


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
                dtype=np.float64,
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
        effective_learning_rate = float(
            config.get("learning_rate", self.config.training.learning_rate)
        )
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
                learning_rate=effective_learning_rate,
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
                learning_rate=effective_learning_rate,
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
                learning_rate=effective_learning_rate,
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
                learning_rate=effective_learning_rate,
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
                learning_rate=effective_learning_rate,
            )
            return [result.model_vector, result.control_delta], num_examples, metrics

        raise ValueError(f"Unknown method: {self.method}")

    def _sparse_proposed_enabled(self) -> bool:
        return bool(self.method == "proposed" and self.config.sparse.enabled)

    def _sparse_mask_for_precision_phase(
        self,
        *,
        trainer,
        global_mean: np.ndarray,
        global_precision: np.ndarray,
        local_precision: np.ndarray,
        loader,
        logical_round: int,
        phase_seed: int,
    ):
        """Return one coordinate mask shared by Delta-rho and Delta-nu.

        keep=100% is a strict pass-through and skips the score probe so the
        numerical path is identical to the dense Figure-2 Proposed method.

        For keep<100%, a non-communicated local posterior-mean probe is trained
        with the just-learned local precision held as its covariance.  This
        provides q_k=N(mu_k, diag(rho_k)^-1) before communication, allowing the
        requested score |mu_k-mu_G|/(sigma_k+eps) to choose a same-round mask.
        The probe changes no transmitted model state; it is used only to rank
        coordinates.
        """
        if float(self.config.sparse.keep_ratio) >= 1.0:
            return full_mask_info(self.dimension), None

        probe = trainer.train_natural_mean_phase(
            global_mean=global_mean,
            prior_global_precision=global_precision,
            next_global_precision=local_precision,
            local_precision=local_precision,
            loader=loader,
            seed=int(phase_seed) + 70_000_003,
        )
        safe_precision = np.maximum(
            np.asarray(local_precision, dtype=np.float64),
            float(self.config.model.min_precision),
        )
        local_sigma = np.sqrt(1.0 / safe_precision)
        bayesian_scores = bayesian_update_snr_score(
            probe.implied_mean,
            global_mean,
            local_sigma,
            epsilon=float(self.config.sparse.score_epsilon),
        )
        random_seed = (
            int(self.run_seed)
            + 81_000_019 * int(logical_round)
            + 10_007 * int(self.client_id)
        )
        info = select_sparse_mask(
            selection=self.config.sparse.selection,
            keep_ratio=float(self.config.sparse.keep_ratio),
            min_keep=int(self.config.sparse.min_keep),
            bayesian_scores=bayesian_scores,
            random_seed=random_seed,
        )
        return info, probe

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
        learning_rate: float,
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
        global_precision = np.asarray(parameters[1], dtype=np.float64)
        trainer = create_bayesian_trainer(
            model=self.model,
            layout=self.layout,
            model_cfg=self.config.model,
            train_cfg=self.config.training,
            device=self.device,
            learning_rate=learning_rate,
        )

        if phase == PRECISION_PHASE:
            result = trainer.train_precision_phase(
                global_mean=global_mean,
                global_precision=global_precision,
                loader=loader,
                seed=phase_seed,
            )
            self._save_proposed_precision(logical_round, result.precision)
            sparse_info = None
            sparse_probe = None
            transmitted_precision = result.precision
            if self._sparse_proposed_enabled():
                sparse_info, sparse_probe = self._sparse_mask_for_precision_phase(
                    trainer=trainer,
                    global_mean=global_mean,
                    global_precision=global_precision,
                    local_precision=result.precision,
                    loader=loader,
                    logical_round=logical_round,
                    phase_seed=phase_seed,
                )
                if float(self.config.sparse.keep_ratio) < 1.0:
                    self._save_proposed_sparse_mask(logical_round, sparse_info.mask)
                    transmitted_precision = global_precision.copy()
                    transmitted_precision[sparse_info.mask] = result.precision[
                        sparse_info.mask
                    ]
            metrics = self._base_metrics(
                metadata,
                logical_round,
                physical_round,
                phase,
                train_loss=float(result.average_loss),
                phase1_loss=float(result.average_loss),
                phase2_loss=0.0,
                local_steps=int(result.local_steps),
                learning_rate=learning_rate,
            )
            metrics.update(
                {
                    "local_precision_mean": float(np.mean(result.precision)),
                    "local_precision_min": float(np.min(result.precision)),
                    "local_precision_max": float(np.max(result.precision)),
                    "local_precision_delta_l2": float(result.precision_delta_l2),
                    "local_precision_delta_max_abs": float(
                        result.precision_delta_max_abs
                    ),
                    "local_precision_changed_fraction": float(
                        result.precision_changed_fraction
                    ),
                    "local_precision_gradient_l2_mean": float(
                        result.applied_gradient_l2_mean
                    ),
                    "local_precision_gradient_max_abs": float(
                        result.applied_gradient_max_abs
                    ),
                    "local_nu_l2": 0.0,
                    "local_implied_mean_l2": 0.0,
                }
            )
            if sparse_info is not None:
                metrics.update(
                    {
                        "sparse_enabled": True,
                        "sparse_selection": str(self.config.sparse.selection),
                        "sparse_keep_ratio": float(self.config.sparse.keep_ratio),
                        "sparse_kept_coordinates": int(sparse_info.kept),
                        "sparse_total_coordinates": int(sparse_info.total),
                        "sparse_score_threshold": float(sparse_info.threshold),
                        "sparse_score_mean": float(sparse_info.score_mean),
                        "sparse_selected_score_mean": float(
                            sparse_info.selected_score_mean
                        ),
                        "sparse_dropped_score_mean": float(
                            sparse_info.dropped_score_mean
                        ),
                        "sparse_probe_mean_l2": (
                            float(np.linalg.norm(sparse_probe.implied_mean))
                            if sparse_probe is not None
                            else 0.0
                        ),
                    }
                )
            # The server aggregates rho and retains the global mean. Dropped
            # coordinates equal rho_t, so their Delta-rho is exactly zero.
            return [transmitted_precision], num_examples, metrics

        if phase == NATURAL_MEAN_PHASE:
            local_precision = self._load_proposed_precision(logical_round)
            prior_global_precision = np.asarray(parameters[2], dtype=np.float64)
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
                learning_rate=learning_rate,
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
            transmitted_nu = result.nu
            if self._sparse_proposed_enabled():
                if float(self.config.sparse.keep_ratio) >= 1.0:
                    mask = np.ones(self.dimension, dtype=bool)
                else:
                    mask = self._load_proposed_sparse_mask(logical_round)
                    transmitted_nu = global_mean.copy()
                    transmitted_nu[mask] = result.nu[mask]
                metrics.update(
                    {
                        "sparse_enabled": True,
                        "sparse_selection": str(self.config.sparse.selection),
                        "sparse_keep_ratio": float(self.config.sparse.keep_ratio),
                        "sparse_kept_coordinates": int(np.count_nonzero(mask)),
                        "sparse_total_coordinates": int(mask.size),
                    }
                )
            if self.config.runtime.cleanup_phase_state:
                self._remove_proposed_precision(logical_round)
                self._remove_proposed_sparse_mask(logical_round)
            # The server aggregates nu; dropped coordinates equal mu_t, so
            # their Delta-nu is exactly zero.
            return [transmitted_nu], num_examples, metrics

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
        learning_rate: float,
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
            "learning_rate": float(learning_rate),
            "device": str(self.device),
            "sparse_enabled": bool(self._sparse_proposed_enabled()),
            "sparse_selection": (
                str(self.config.sparse.selection)
                if self._sparse_proposed_enabled() else ""
            ),
            "sparse_keep_ratio": (
                float(self.config.sparse.keep_ratio)
                if self._sparse_proposed_enabled() else 1.0
            ),
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

    def proposed_sparse_mask_state_path(self, logical_round: int) -> Path:
        return self.state_dir / (
            f"client_{self.client_id:05d}_round_{int(logical_round):06d}_sparse_mask.npy"
        )

    def _load_scaffold_control(self) -> np.ndarray:
        path = self.scaffold_state_path
        if not path.exists():
            return np.zeros(self.dimension, dtype=np.float32)
        value = np.load(path)
        if value.shape != (self.dimension,):
            raise ValueError(f"Invalid SCAFFOLD state in {path}: {value.shape}")
        return value.astype(np.float64)

    def _save_scaffold_control(self, value: np.ndarray) -> None:
        self._atomic_save(self.scaffold_state_path, value)

    def _save_proposed_precision(self, logical_round: int, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=np.float64).reshape(-1)
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
        return value.astype(np.float64)

    def _save_proposed_sparse_mask(self, logical_round: int, mask: np.ndarray) -> None:
        value = np.asarray(mask, dtype=np.uint8).reshape(-1)
        if value.shape != (self.dimension,):
            raise ValueError(
                f"Sparse mask has shape {value.shape}; expected {(self.dimension,)}"
            )
        self._atomic_save(self.proposed_sparse_mask_state_path(logical_round), value)

    def _load_proposed_sparse_mask(self, logical_round: int) -> np.ndarray:
        path = self.proposed_sparse_mask_state_path(logical_round)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing sparse mask for client {self.client_id}, logical round "
                f"{logical_round}: {path}"
            )
        value = np.load(path).reshape(-1)
        if value.shape != (self.dimension,):
            raise ValueError(f"Invalid sparse mask state in {path}: {value.shape}")
        return value.astype(bool)

    def _remove_proposed_sparse_mask(self, logical_round: int) -> None:
        self.proposed_sparse_mask_state_path(logical_round).unlink(missing_ok=True)

    def _remove_proposed_precision(self, logical_round: int) -> None:
        self.proposed_precision_state_path(logical_round).unlink(missing_ok=True)

    def _atomic_save(self, path: Path, value: np.ndarray) -> None:
        # Preserve the incoming dtype.  Proposed precision state is float64;
        # SCAFFOLD control state remains float32.
        value = np.asarray(value)
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
