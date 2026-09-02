"""Flower NumPyClient for FedAvg and one-phase Bayesian-Torch BayesAvg."""

from __future__ import annotations

from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch

from bayesian_torch_backend import build_initial_states
from bayesian_training import train_bayesavg
from config import SimulationConfig
from dataset import load_client_loader
from deterministic import train_fedavg
from models import build_model
from runtime_utils import release_cuda_memory, resolve_device, should_pin_memory
from serialization import ParameterLayout


class BaselineNumPyClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: int,
        method: str,
        config: SimulationConfig,
        partition_path: str,
        run_seed: int,
    ) -> None:
        self.client_id = int(client_id)
        self.method = str(method).strip().lower()
        if self.method == "proposed":
            self.method = "bayesavg"
        if self.method not in {"fedavg", "bayesavg"}:
            raise ValueError(f"Unsupported method: {method!r}")
        self.config = config
        self.partition_path = str(partition_path)
        self.run_seed = int(run_seed)

        torch.set_num_threads(int(config.runtime.torch_num_threads))
        self.device = resolve_device(config.runtime.client_device)
        if self.device.type == "cuda" and self.device.index is not None:
            torch.cuda.set_device(self.device.index)

        self.model = build_model(
            config.data.dataset,
            config.model.name,
            config.model.num_classes,
        )
        self.layout = ParameterLayout(self.model)

    def _initial_state(self) -> np.ndarray:
        bayes_state, mean_state, _bayesian_d, _det_d = build_initial_states(
            dataset=self.config.data.dataset,
            model_cfg=self.config.model,
            variational_cfg=self.config.variational,
            seed=self.run_seed,
        )
        return bayes_state if self.method == "bayesavg" else mean_state

    def get_parameters(self, config: Dict[str, fl.common.Scalar]) -> List[np.ndarray]:
        del config
        return [self._initial_state()]

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, fl.common.Scalar],
    ) -> Tuple[List[np.ndarray], int, Dict[str, fl.common.Scalar]]:
        if len(parameters) != 1:
            raise ValueError(f"{self.method} expects exactly one global state vector")
        try:
            return self._fit_impl(parameters[0], config)
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
        global_state: np.ndarray,
        config: Dict[str, fl.common.Scalar],
    ) -> Tuple[List[np.ndarray], int, Dict[str, fl.common.Scalar]]:
        server_round = int(config.get("server_round", 1))
        learning_rate = float(config.get("learning_rate", self.config.training.learning_rate))
        local_seed = int(self.run_seed + 100_003 * server_round + self.client_id)

        loader, metadata = load_client_loader(
            self.config.data,
            self.config.training,
            self.partition_path,
            self.client_id,
            shuffle_seed=local_seed,
            pin_memory=should_pin_memory(self.config.data.pin_memory, self.device),
        )
        num_examples = int(metadata["num_examples"])
        average_client_size = float(metadata["partition_mean_size"])

        if self.method == "fedavg":
            result = train_fedavg(
                model=self.model,
                layout=self.layout,
                global_vector=np.asarray(global_state, dtype=np.float32),
                loader=loader,
                train_cfg=self.config.training,
                device=self.device,
                seed=local_seed,
                learning_rate=learning_rate,
            )
            metrics = {
                "client_id": self.client_id,
                "num_examples": num_examples,
                "train_loss": float(result.average_loss),
                "ce_loss": float(result.average_loss),
                "kl_sum": 0.0,
                "local_steps": int(result.local_steps),
                "learning_rate": learning_rate,
                "kl_weight_base": 0.0,
                "kl_weight_client": 0.0,
                "kl_warmup_factor": 0.0,
                "mu_update_l2": 0.0,
                "rho_update_l2": 0.0,
                "deterministic_update_l2": 0.0,
                "sigma_mean": 0.0,
                "sigma_min": 0.0,
                "sigma_max": 0.0,
                "variance_floor_clipped_fraction": 0.0,
                "model_update_l2": float(result.update_l2),
                "model_update_max_abs": float(result.update_max_abs),
            }
            return [result.model_vector], num_examples, metrics

        result = train_bayesavg(
            deterministic_model=self.model,
            layout=self.layout,
            global_state=np.asarray(global_state, dtype=np.float32),
            loader=loader,
            model_cfg=self.config.model,
            train_cfg=self.config.training,
            variational_cfg=self.config.variational,
            device=self.device,
            seed=local_seed,
            server_round=server_round,
            learning_rate=learning_rate,
            client_size=num_examples,
            average_client_size=average_client_size,
        )
        metrics = {
            "client_id": self.client_id,
            "num_examples": num_examples,
            "train_loss": float(result.average_loss),
            "ce_loss": float(result.average_ce),
            "kl_sum": float(result.average_kl_sum),
            "local_steps": int(result.local_steps),
            "learning_rate": learning_rate,
            "kl_weight_base": float(result.kl_weight_base),
            "kl_weight_client": float(result.kl_weight_client),
            "kl_warmup_factor": float(result.kl_warmup_factor),
            "mu_update_l2": float(result.mu_update_l2),
            "rho_update_l2": float(result.rho_update_l2),
            "deterministic_update_l2": float(result.deterministic_update_l2),
            "sigma_mean": float(result.sigma_mean),
            "sigma_min": float(result.sigma_min),
            "sigma_max": float(result.sigma_max),
            "variance_floor_clipped_fraction": float(
                result.variance_floor_clipped_fraction
            ),
            "model_update_l2": 0.0,
            "model_update_max_abs": 0.0,
        }
        return [result.state_vector], num_examples, metrics
