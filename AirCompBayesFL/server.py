"""Flower ServerApp, custom AirComp strategy, evaluation, and simulation runner."""

from __future__ import annotations

import math
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.client import ClientApp
from flwr.common import Context, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation

from aggregation import (
    aggregate_deterministic,
    aggregate_gaussian_natural_parameters,
    aggregate_scaffold,
    normalized_weights,
)
from aircomp import AirCompStats
from client import AirCompNumPyClient
from config import SimulationConfig
from dataset import load_test_loader
from experiments import RunSpec, payload_multiplier
from logger import RunLogger
from metrics import evaluate_bayesian, evaluate_deterministic
from models import build_model
from runtime_utils import configure_runtime_environment, resolve_device, should_pin_memory
from serialization import ParameterLayout, initial_model_vector
from wireless import sample_rayleigh_channels


class AirCompStrategy(FedAvg):
    """Full-participation Flower strategy with wireless analog aggregation."""

    def __init__(
        self,
        *,
        config: SimulationConfig,
        run_spec: RunSpec,
        partition_path: str,
    ) -> None:
        self.config_obj = config
        self.run_spec = run_spec
        self.partition_path = partition_path
        self.method = run_spec.method
        self.model = build_model(config.model.name, config.model.num_classes)
        self.layout = ParameterLayout(self.model)
        self.dimension = self.layout.total_numel
        self.server_device = resolve_device(config.runtime.server_device)
        # Ray can hide CUDA devices from the ServerApp because GPU resources are
        # assigned only to ClientApps.  Never request CUDA-pinned batches when
        # the central evaluator itself runs on CPU.
        self.test_loader = load_test_loader(
            config.data,
            pin_memory=should_pin_memory(
                config.data.pin_memory, self.server_device
            ),
        )

        initial_model = initial_model_vector(self.model, run_spec.seed)
        if self.method == "proposed":
            initial_precision = np.full(
                self.dimension,
                1.0 / (config.model.initial_prior_std**2),
                dtype=np.float32,
            )
            initial_arrays = [initial_model, initial_precision]
        elif self.method == "scaffold":
            initial_arrays = [
                initial_model,
                np.zeros(self.dimension, dtype=np.float32),
            ]
        else:
            initial_arrays = [initial_model]

        self.current_arrays = [array.copy() for array in initial_arrays]
        self.last_train_loss = float("nan")
        self.last_aircomp_stats = AirCompStats.zero()
        self.channel_uses_cumulative = 0
        self.ofdm_symbols_cumulative = 0
        self.started_at = time.perf_counter()
        self.logger = RunLogger(
            config.output.directory,
            config.output.metrics_filename,
            config.output.reliability_filename,
            config.output.clients_filename,
        )

        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=config.data.num_clients,
            min_evaluate_clients=0,
            min_available_clients=config.data.num_clients,
            evaluate_fn=None,
            on_fit_config_fn=lambda server_round: {"server_round": server_round},
            accept_failures=False,
            initial_parameters=ndarrays_to_parameters(initial_arrays),
        )

    def aggregate_fit(self, server_round, results, failures):  # type: ignore[override]
        if not results:
            return None, {}
        if failures and not self.accept_failures:
            return None, {}

        local_arrays: List[List[np.ndarray]] = []
        examples: List[int] = []
        distances: List[float] = []
        losses: List[float] = []

        for client_proxy, fit_res in results:
            arrays = [
                np.asarray(value, dtype=np.float32)
                for value in parameters_to_ndarrays(fit_res.parameters)
            ]
            local_arrays.append(arrays)
            examples.append(int(fit_res.num_examples))
            metrics = fit_res.metrics or {}
            distances.append(float(metrics.get("distance_m", 1.0)))
            losses.append(float(metrics.get("train_loss", float("nan"))))
            self.logger.clients.append(
                {
                    "run_id": self.run_spec.run_id,
                    "round": server_round,
                    "client_id": int(metrics.get("client_id", -1)),
                    "num_examples": int(fit_res.num_examples),
                    "distance_m": float(metrics.get("distance_m", float("nan"))),
                    "train_loss": float(metrics.get("train_loss", float("nan"))),
                    "phase1_loss": float(metrics.get("phase1_loss", 0.0)),
                    "phase2_loss": float(metrics.get("phase2_loss", 0.0)),
                    "local_steps": int(metrics.get("local_steps", 0)),
                }
            )

        weights = normalized_weights(examples)
        valid_losses = np.asarray(losses, dtype=np.float64)
        valid_mask = np.isfinite(valid_losses)
        if np.any(valid_mask):
            renormalized = weights[valid_mask]
            renormalized = renormalized / renormalized.sum()
            self.last_train_loss = float(np.dot(renormalized, valid_losses[valid_mask]))

        rng = np.random.default_rng(self.run_spec.seed + 1_000_033 * int(server_round))
        channels = sample_rayleigh_channels(
            np.asarray(distances, dtype=np.float64),
            self.config_obj.wireless.num_subchannels,
            self.config_obj.wireless.path_loss_exponent,
            rng,
        )

        if self.method in {"fedavg", "fedprox"}:
            aggregation = aggregate_deterministic(
                self.current_arrays[0],
                [arrays[0] for arrays in local_arrays],
                weights,
                channels,
                self.config_obj.wireless,
                rng,
            )
        elif self.method == "scaffold":
            aggregation = aggregate_scaffold(
                self.current_arrays[0],
                self.current_arrays[1],
                [arrays[0] for arrays in local_arrays],
                [arrays[1] for arrays in local_arrays],
                weights,
                channels,
                self.config_obj.wireless,
                rng,
            )
        elif self.method == "proposed":
            aggregation = aggregate_gaussian_natural_parameters(
                self.current_arrays[0],
                self.current_arrays[1],
                [arrays[0] for arrays in local_arrays],
                [arrays[1] for arrays in local_arrays],
                weights,
                channels,
                self.config_obj.wireless,
                self.config_obj.model,
                rng,
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.current_arrays = [array.astype(np.float32) for array in aggregation.parameters]
        self.last_aircomp_stats = aggregation.aircomp_stats
        multiplier = payload_multiplier(self.method)
        self.channel_uses_cumulative += multiplier * self.dimension
        self.ofdm_symbols_cumulative += multiplier * math.ceil(
            self.dimension / self.config_obj.wireless.num_subchannels
        )
        parameters = ndarrays_to_parameters(self.current_arrays)
        return parameters, {
            "train_loss": self.last_train_loss,
            "aircomp_nmse": self.last_aircomp_stats.nmse,
        }

    def evaluate(self, server_round: int, parameters: Parameters):  # type: ignore[override]
        if (
            server_round != 0
            and server_round != self.run_spec.rounds
            and server_round % self.config_obj.training.evaluate_every != 0
        ):
            return None

        arrays = [
            np.asarray(value, dtype=np.float32)
            for value in parameters_to_ndarrays(parameters)
        ]
        self.current_arrays = arrays

        if self.method == "proposed":
            evaluation = evaluate_bayesian(
                self.model,
                self.layout,
                arrays[0],
                arrays[1],
                self.test_loader,
                self.server_device,
                mc_samples=self.config_obj.training.mc_eval_samples,
                seed=self.run_spec.seed + server_round,
            )
            posterior_variance = float(np.mean(1.0 / np.maximum(arrays[1], 1.0e-12)))
        else:
            evaluation = evaluate_deterministic(
                self.model,
                self.layout,
                arrays[0],
                self.test_loader,
                self.server_device,
            )
            posterior_variance = 0.0

        multiplier = payload_multiplier(self.method)
        channel_uses_round = 0 if server_round == 0 else multiplier * self.dimension
        ofdm_symbols_round = (
            0
            if server_round == 0
            else multiplier
            * math.ceil(self.dimension / self.config_obj.wireless.num_subchannels)
        )
        base = {
            "run_id": self.run_spec.run_id,
            "experiment": self.run_spec.experiment,
            "condition": self.run_spec.condition,
            "method": self.method,
            "realization": self.run_spec.realization,
            "seed": self.run_spec.seed,
            "round": server_round,
            "num_clients": self.config_obj.data.num_clients,
            "labels_per_client": self.config_obj.data.labels_per_client,
            "mean_samples_per_client": self.config_obj.data.mean_samples_per_client,
            "power_dbm": self.config_obj.wireless.power_dbm,
            "noise_dbm": self.config_obj.wireless.noise_dbm,
        }
        self.logger.metrics.append(
            {
                **base,
                "accuracy": evaluation.accuracy,
                "nll": evaluation.nll,
                "ece": evaluation.ece,
                "train_loss": self.last_train_loss,
                "posterior_variance": posterior_variance,
                "channel_uses_round": channel_uses_round,
                "channel_uses_cumulative": self.channel_uses_cumulative,
                "ofdm_symbols_round": ofdm_symbols_round,
                "ofdm_symbols_cumulative": self.ofdm_symbols_cumulative,
                "aircomp_nmse": self.last_aircomp_stats.nmse,
                "aircomp_distortion_nmse": self.last_aircomp_stats.distortion_nmse,
                "aircomp_clipped_fraction": self.last_aircomp_stats.clipped_fraction,
                "aircomp_average_symbol_power_watts": self.last_aircomp_stats.average_symbol_power_watts,
                "aircomp_maximum_symbol_power_watts": self.last_aircomp_stats.maximum_symbol_power_watts,
                "aircomp_noise_l2": self.last_aircomp_stats.noise_l2,
                "wall_time_sec": time.perf_counter() - self.started_at,
            }
        )
        self.logger.log_reliability(base, evaluation)

        if server_round == self.run_spec.rounds and self.config_obj.output.save_checkpoints:
            self.logger.save_checkpoint(
                self.run_spec.run_id,
                arrays,
                {
                    **base,
                    "accuracy": evaluation.accuracy,
                    "ece": evaluation.ece,
                    "nll": evaluation.nll,
                },
            )
        return float(evaluation.nll), {
            "accuracy": float(evaluation.accuracy),
            "ece": float(evaluation.ece),
        }


def run_flower_simulation(
    config: SimulationConfig,
    run_spec: RunSpec,
    partition_path: str,
) -> None:
    """Execute one method/condition/realization with Flower's Ray backend."""
    state_dir = Path(config.output.directory) / "client_state" / run_spec.run_id
    if state_dir.exists():
        shutil.rmtree(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    def client_fn(context: Context):
        client_id = int(context.node_config["partition-id"])
        return AirCompNumPyClient(
            client_id=client_id,
            method=run_spec.method,
            config=config,
            partition_path=partition_path,
            run_seed=run_spec.seed,
            state_dir=str(state_dir),
        ).to_client()

    def server_fn(context: Context) -> ServerAppComponents:
        del context
        strategy = AirCompStrategy(
            config=config,
            run_spec=run_spec,
            partition_path=partition_path,
        )
        return ServerAppComponents(
            strategy=strategy,
            config=ServerConfig(num_rounds=run_spec.rounds),
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
            num_supernodes=config.data.num_clients,
            backend_name="ray",
            backend_config=backend_config,
            verbose_logging=config.runtime.verbose_flower,
        )
    finally:
        # Flower normally manages Ray lifecycle, but a failed native-Windows
        # simulation can leave worker processes and GPU allocations alive.  A
        # finally block makes sequential runs and retries deterministic.
        try:
            import ray

            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass
