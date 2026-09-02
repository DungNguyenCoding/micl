"""Flower/Ray and local execution for one-phase FedAvg and BayesAvg."""

from __future__ import annotations

import gc
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import flwr as fl
import numpy as np
from flwr.client import ClientApp
from flwr.common import Context, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation

from aggregation import normalized_weights, weighted_average
from bayesian_torch_backend import BayesianTorchStateAdapter, build_initial_states
from bayesian_training import resolved_base_kl_weight
from client import BaselineNumPyClient
from config import SimulationConfig
from dataset import load_partition, load_test_loader
from experiments import RunSpec
from logger import RunLogger
from metrics import evaluate_bayesian_state, evaluate_deterministic
from models import build_model
from runtime_utils import (
    configure_runtime_environment,
    release_cuda_memory,
    resolve_backend,
    resolve_device,
    should_pin_memory,
)
from serialization import ParameterLayout
from training_schedule import learning_rate_for_round


@dataclass
class ClientFitPayload:
    state: np.ndarray
    num_examples: int
    metrics: Mapping[str, object]


class BaselineStrategy(FedAvg):
    def __init__(
        self,
        *,
        config: SimulationConfig,
        run_spec: RunSpec,
        partition_path: str,
    ) -> None:
        self.config_obj = config
        self.run_spec = run_spec
        self.partition_path = str(partition_path)
        self.method = str(run_spec.method).lower()
        self.model = build_model(config.data.dataset, config.model.name, config.model.num_classes)
        self.layout = ParameterLayout(self.model)
        self.server_device = resolve_device(config.runtime.server_device)
        self.test_loader = load_test_loader(
            config.data,
            pin_memory=should_pin_memory(config.data.pin_memory, self.server_device),
        )
        self.partition_summary = load_partition(partition_path)

        bayes_state, matched_mean, bayesian_d, deterministic_d = build_initial_states(
            dataset=config.data.dataset,
            model_cfg=config.model,
            variational_cfg=config.variational,
            seed=run_spec.seed,
        )
        self.bayesian_dimension = int(bayesian_d)
        self.deterministic_dimension = int(deterministic_d)
        self.model_dimension = int(self.layout.total_numel)
        self.current_state = (
            bayes_state.copy() if self.method == "bayesavg" else matched_mean.copy()
        )
        self.payload_dimension = int(self.current_state.size)

        self.last_train_loss = float("nan")
        self.last_state_update_l2 = 0.0
        self.last_selected_client_ids: List[int] = []
        self.last_selected_count = 0
        self.upload_scalars_cumulative = 0
        self.started_at = time.perf_counter()
        self.logger = RunLogger(
            config.output.directory,
            config.output.metrics_filename,
            config.output.clients_filename,
            config.output.reliability_filename,
            config.output.participation_filename,
        )

        selected = config.participating_clients()
        super().__init__(
            fraction_fit=float(config.federation.client_fraction),
            fraction_evaluate=0.0,
            min_fit_clients=int(selected),
            min_evaluate_clients=0,
            min_available_clients=int(config.data.num_clients),
            evaluate_fn=None,
            on_fit_config_fn=self.fit_config_for_round,
            accept_failures=not bool(config.runtime.fail_on_client_failure),
            initial_parameters=ndarrays_to_parameters([self.current_state]),
        )

    def fit_config_for_round(self, server_round: int) -> Dict[str, fl.common.Scalar]:
        return {
            "server_round": int(server_round),
            "learning_rate": float(
                learning_rate_for_round(self.config_obj.training, int(server_round))
            ),
        }

    def configure_fit(self, server_round, parameters, client_manager):  # type: ignore[override]
        # Flower's SimpleClientManager samples using Python's random module.
        # Seed it per round so separate method runs receive reproducible cohorts
        # when the same supernode IDs are available.
        state = random.getstate()
        random.seed(int(self.run_spec.seed) + 1_000_003 * int(server_round))
        try:
            return super().configure_fit(server_round, parameters, client_manager)
        finally:
            random.setstate(state)

    def aggregate_fit(self, server_round, results, failures):  # type: ignore[override]
        if failures and self.config_obj.runtime.fail_on_client_failure:
            raise RuntimeError(
                f"Round {server_round}: {len(failures)} client job(s) failed; "
                f"first failure: {failures[0]!r}"
            )
        if not results:
            raise RuntimeError(f"Round {server_round}: no successful client results")

        payloads: List[ClientFitPayload] = []
        for _proxy, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            if len(arrays) != 1:
                raise ValueError("Each client must return one flat state vector")
            payloads.append(
                ClientFitPayload(
                    state=np.asarray(arrays[0], dtype=np.float32),
                    num_examples=int(fit_res.num_examples),
                    metrics=dict(fit_res.metrics or {}),
                )
            )
        return self.aggregate_payloads(int(server_round), payloads)

    def aggregate_payloads(
        self,
        server_round: int,
        payloads: Sequence[ClientFitPayload],
    ) -> tuple[Parameters, Dict[str, float]]:
        ordered = sorted(payloads, key=lambda x: int(x.metrics.get("client_id", -1)))
        examples = [int(p.num_examples) for p in ordered]
        states = [np.asarray(p.state, dtype=np.float32) for p in ordered]
        weights = normalized_weights(examples)

        previous = self.current_state.astype(np.float64, copy=True)
        self.current_state = weighted_average(states, examples).astype(np.float32)
        self.last_state_update_l2 = float(
            np.linalg.norm(self.current_state.astype(np.float64) - previous)
        )
        losses = np.asarray(
            [float(p.metrics.get("train_loss", np.nan)) for p in ordered],
            dtype=np.float64,
        )
        finite = np.isfinite(losses)
        self.last_train_loss = (
            float(np.sum(weights[finite] * losses[finite]) / np.sum(weights[finite]))
            if np.any(finite) and float(np.sum(weights[finite])) > 0
            else float("nan")
        )

        self.last_selected_client_ids = [
            int(p.metrics.get("client_id", -1)) for p in ordered
        ]
        self.last_selected_count = len(ordered)
        self.upload_scalars_cumulative += self.payload_dimension * len(ordered)

        for payload in ordered:
            m = dict(payload.metrics)
            client_id = int(m.get("client_id", -1))
            base = {
                "run_id": self.run_spec.run_id,
                "round": int(server_round),
                "client_id": client_id,
                "num_examples": int(payload.num_examples),
            }
            base.update(m)
            self.logger.clients.append(base)
            self.logger.participation.append(
                {
                    "run_id": self.run_spec.run_id,
                    "round": int(server_round),
                    "client_id": client_id,
                    "num_examples": int(payload.num_examples),
                }
            )

        return ndarrays_to_parameters([self.current_state]), {
            "train_loss": float(self.last_train_loss),
            "selected_clients": float(self.last_selected_count),
        }

    def _metric_base(self, server_round: int) -> Dict[str, object]:
        p = self.partition_summary
        cfg = self.config_obj
        base_kl = (
            resolved_base_kl_weight(cfg.variational, self.bayesian_dimension)
            if self.method == "bayesavg"
            else 0.0
        )
        return {
            "run_id": self.run_spec.run_id,
            "dataset": cfg.data.dataset,
            "model": cfg.model.name,
            "method": self.method,
            "seed": self.run_spec.seed,
            "round": int(server_round),
            "num_clients": int(cfg.data.num_clients),
            "selected_clients": int(self.last_selected_count if server_round else 0),
            "client_fraction": float(cfg.federation.client_fraction),
            "selected_client_ids": json.dumps(self.last_selected_client_ids),
            "partition": cfg.data.partition,
            "dirichlet_alpha": (
                float(cfg.data.dirichlet_alpha)
                if cfg.data.partition == "sparse_dirichlet"
                else ""
            ),
            "partition_total_samples": int(p["total_samples_used"]),
            "partition_mean_size": float(p["mean_size"]),
            "partition_min_size": int(p["min_size"]),
            "partition_max_size": int(p["max_size"]),
            "mean_classes_per_client": float(p["mean_classes_per_client"]),
            "local_epochs": int(cfg.training.local_epochs),
            "batch_size": int(cfg.training.batch_size),
            "optimizer": cfg.training.optimizer,
            "momentum": float(cfg.training.momentum),
            "weight_decay": float(cfg.training.weight_decay),
            "lr_scheduler": cfg.training.lr_scheduler,
            "lr_decay_rounds": int(cfg.training.lr_decay_rounds),
            "min_learning_rate": float(cfg.training.min_learning_rate),
            "learning_rate": float(
                learning_rate_for_round(cfg.training, max(1, int(server_round)))
            ),
            "train_loss": float(self.last_train_loss),
            "bayesian_dimension": int(self.bayesian_dimension),
            "deterministic_dimension": int(self.deterministic_dimension),
            "model_dimension": int(self.model_dimension),
            "payload_scalars_per_client": int(self.payload_dimension),
            "kl_weight_config": (
                "" if cfg.variational.kl_weight is None else float(cfg.variational.kl_weight)
            ),
            "kl_weight_resolved": float(base_kl),
            "kl_weight_schedule": bool(cfg.variational.kl_weight_schedule),
            "kl_warmup_rounds": int(cfg.variational.kl_warmup_rounds),
            "lambda_scale_by_size": bool(cfg.variational.lambda_scale_by_size),
            "mc_train": int(cfg.variational.mc_train),
            "mc_eval": int(cfg.variational.mc_eval),
            "variance_floor_ratio": float(cfg.variational.variance_floor_ratio),
            "global_state_update_l2": float(self.last_state_update_l2),
            "upload_scalars_round": int(
                0 if int(server_round) == 0 else self.payload_dimension * self.last_selected_count
            ),
            "upload_scalars_cumulative": int(self.upload_scalars_cumulative),
            "wall_time_sec": float(time.perf_counter() - self.started_at),
        }

    def evaluate(self, server_round: int, parameters: Parameters):  # type: ignore[override]
        if (
            int(server_round) != 0
            and int(server_round) != int(self.run_spec.rounds)
            and int(server_round) % int(self.config_obj.training.evaluate_every) != 0
        ):
            return None

        arrays = parameters_to_ndarrays(parameters)
        if len(arrays) != 1:
            raise ValueError("Server state expects one flat vector")
        self.current_state = np.asarray(arrays[0], dtype=np.float32)
        base = self._metric_base(int(server_round))

        if self.method == "fedavg":
            evaluation = evaluate_deterministic(
                self.model,
                self.layout,
                self.current_state,
                self.test_loader,
                self.server_device,
            )
            mean_eval = evaluation
            sigma_stats = (0.0, 0.0, 0.0)
        else:
            evaluation, mean_eval, sigma_stats, _mean_vector = evaluate_bayesian_state(
                self.model,
                self.layout,
                self.current_state,
                self.config_obj.variational,
                self.test_loader,
                self.server_device,
                seed=int(self.run_spec.seed) + 7_000_001 * int(server_round),
            )

        base.update(
            {
                "accuracy": float(evaluation.accuracy),
                "nll": float(evaluation.nll),
                "ece": float(evaluation.ece),
                "posterior_predictive_accuracy": float(evaluation.accuracy),
                "posterior_predictive_nll": float(evaluation.nll),
                "posterior_predictive_ece": float(evaluation.ece),
                "posterior_mean_accuracy": float(mean_eval.accuracy),
                "posterior_mean_nll": float(mean_eval.nll),
                "posterior_mean_ece": float(mean_eval.ece),
                "posterior_sigma_mean": float(sigma_stats[0]),
                "posterior_sigma_min": float(sigma_stats[1]),
                "posterior_sigma_max": float(sigma_stats[2]),
            }
        )
        self.logger.metrics.append(base)
        rel_base = {
            "run_id": self.run_spec.run_id,
            "method": self.method,
            "round": int(server_round),
        }
        self.logger.log_reliability(rel_base, evaluation, "predictive")
        if self.method == "bayesavg":
            self.logger.log_reliability(rel_base, mean_eval, "posterior_mean")

        if int(server_round) == int(self.run_spec.rounds) and self.config_obj.output.save_checkpoints:
            self.logger.save_checkpoint(
                self.run_spec.run_id,
                self.current_state,
                {
                    "method": self.method,
                    "dataset": self.config_obj.data.dataset,
                    "round": int(server_round),
                    "seed": self.run_spec.seed,
                },
            )

        print(
            f"Round {server_round}/{self.run_spec.rounds} | {self.method} | "
            f"accuracy={evaluation.accuracy:.4f} | "
            f"posterior_mean={mean_eval.accuracy:.4f} | "
            f"nll={evaluation.nll:.4f} | lr={base['learning_rate']:.6f}"
        )
        return float(evaluation.nll), {"accuracy": float(evaluation.accuracy)}


def run_flower_simulation(config: SimulationConfig, run_spec: RunSpec, partition_path: str) -> None:
    def client_fn(context: Context):
        client_id = int(context.node_config["partition-id"])
        return BaselineNumPyClient(
            client_id=client_id,
            method=run_spec.method,
            config=config,
            partition_path=partition_path,
            run_seed=run_spec.seed,
        ).to_client()

    def server_fn(context: Context) -> ServerAppComponents:
        del context
        strategy = BaselineStrategy(
            config=config,
            run_spec=run_spec,
            partition_path=partition_path,
        )
        return ServerAppComponents(
            strategy=strategy,
            config=ServerConfig(num_rounds=int(run_spec.rounds)),
        )

    client_app = ClientApp(client_fn=client_fn)
    server_app = ServerApp(server_fn=server_fn)
    backend_config: Dict[str, object] = {
        "client_resources": {
            "num_cpus": float(config.runtime.client_num_cpus),
            "num_gpus": float(config.runtime.client_num_gpus),
        },
        "init_args": {
            "include_dashboard": bool(config.runtime.ray_include_dashboard),
            "ignore_reinit_error": True,
        },
    }
    configure_runtime_environment(config)
    try:
        run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=int(config.data.num_clients),
            backend_name="ray",
            backend_config=backend_config,
            verbose_logging=bool(config.runtime.verbose_flower),
        )
    finally:
        try:
            import ray

            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass


def run_local_simulation(config: SimulationConfig, run_spec: RunSpec, partition_path: str) -> None:
    strategy = BaselineStrategy(config=config, run_spec=run_spec, partition_path=partition_path)
    parameters = ndarrays_to_parameters([strategy.current_state])
    strategy.evaluate(0, parameters)

    for server_round in range(1, int(run_spec.rounds) + 1):
        count = config.participating_clients()
        rng = np.random.default_rng(int(run_spec.seed) + 1_000_003 * server_round)
        if count >= int(config.data.num_clients):
            selected = np.arange(int(config.data.num_clients), dtype=np.int64)
        else:
            selected = np.sort(
                rng.choice(int(config.data.num_clients), size=count, replace=False)
            )
        payloads: List[ClientFitPayload] = []
        fit_config = strategy.fit_config_for_round(server_round)
        for client_id in selected.tolist():
            client = BaselineNumPyClient(
                client_id=int(client_id),
                method=run_spec.method,
                config=config,
                partition_path=partition_path,
                run_seed=run_spec.seed,
            )
            arrays, n, metrics = client.fit([strategy.current_state.copy()], fit_config)
            payloads.append(
                ClientFitPayload(
                    state=np.asarray(arrays[0], dtype=np.float32),
                    num_examples=int(n),
                    metrics=metrics,
                )
            )
            del client
            gc.collect()
            if strategy.server_device.type == "cuda":
                release_cuda_memory(strategy.server_device)
        parameters, _ = strategy.aggregate_payloads(server_round, payloads)
        strategy.evaluate(server_round, parameters)


def run_configured_simulation(config: SimulationConfig, run_spec: RunSpec, partition_path: str) -> None:
    backend = resolve_backend(config)
    if backend == "ray":
        run_flower_simulation(config, run_spec, partition_path)
    elif backend == "local":
        run_local_simulation(config, run_spec, partition_path)
    else:
        raise ValueError(f"Unsupported runtime backend: {backend}")
