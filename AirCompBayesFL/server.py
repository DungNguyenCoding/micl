"""Flower/Ray and native-Windows local backends for AirCompBayesFL.

Version 1.3 maps every logical proposed-method round to two physical fit
rounds.  Precision is aggregated and broadcast before any client starts the
natural-mean phase, matching Algorithm 1 of the paper.
"""

from __future__ import annotations

import gc
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

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
    aggregate_gaussian_natural_mean_phase,
    aggregate_gaussian_precision_phase,
    aggregate_scaffold,
    normalized_weights,
)
from aircomp import AirCompStats, combine_stats
from bayesian_protocol import (
    MODEL_PHASE,
    NATURAL_MEAN_PHASE,
    PRECISION_PHASE,
    PhaseContext,
    phase_context,
    physical_round_count,
)
from client import AirCompNumPyClient
from config import SimulationConfig
from dataset import load_test_loader
from experiments import RunSpec, payload_multiplier
from logger import RunLogger
from metrics import evaluate_bayesian, evaluate_bayesian_mean, evaluate_deterministic
from models import build_model
from runtime_utils import (
    configure_runtime_environment,
    release_cuda_memory,
    resolve_backend,
    resolve_device,
    should_pin_memory,
)
from sparse_posterior import kept_coordinate_count
from serialization import (
    ParameterLayout,
    initial_model_vector,
    normalize_server_state_dtypes,
)
from wireless import sample_rayleigh_channels


@dataclass
class ClientFitPayload:
    """Backend-neutral representation of one successful client fit result."""

    arrays: List[np.ndarray]
    num_examples: int
    metrics: Mapping[str, object]


class AirCompStrategy(FedAvg):
    """Full-participation strategy with phase-aware wireless aggregation."""

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
        self.method = run_spec.method.lower()
        self.model = build_model(config.model.name, config.model.num_classes)
        self.layout = ParameterLayout(self.model)
        self.dimension = self.layout.total_numel
        self.server_device = resolve_device(config.runtime.server_device)
        self.test_loader = load_test_loader(
            config.data,
            pin_memory=should_pin_memory(config.data.pin_memory, self.server_device),
        )

        initial_model = initial_model_vector(self.model, run_spec.seed)
        if self.method == "proposed":
            initial_precision = np.full(
                self.dimension,
                1.0 / (config.model.initial_prior_std**2),
                dtype=np.float64,
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
        self.last_phase1_train_loss = 0.0
        self.last_phase2_train_loss = 0.0
        self.last_aircomp_stats = AirCompStats.zero()
        self.last_precision_aircomp_stats = AirCompStats.zero()
        self.last_mean_aircomp_stats = AirCompStats.zero()
        self.last_global_mean_update_l2 = 0.0
        self.last_global_mean_update_max_abs = 0.0
        self.last_global_model_update_l2 = 0.0
        self.last_ideal_model_update_l2 = 0.0
        self.last_received_model_update_l2 = 0.0
        self.last_global_precision_update_l2 = 0.0
        self.last_global_precision_update_max_abs = 0.0
        self.pending_precision_logical_round: int | None = None
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
            on_fit_config_fn=self.fit_config_for_round,
            accept_failures=not config.runtime.fail_on_client_failure,
            initial_parameters=ndarrays_to_parameters(initial_arrays),
        )

    @property
    def physical_rounds(self) -> int:
        return physical_round_count(self.method, self.run_spec.rounds)

    def context_for_round(self, physical_round: int) -> PhaseContext:
        return phase_context(self.method, int(physical_round))

    def fit_config_for_round(self, physical_round: int) -> Dict[str, fl.common.Scalar]:
        context = self.context_for_round(physical_round)
        return {
            "server_round": int(context.logical_round),
            "logical_round": int(context.logical_round),
            "physical_round": int(context.physical_round),
            "phase": str(context.phase),
        }

    def aggregate_fit(self, server_round, results, failures):  # type: ignore[override]
        """Flower callback; convert results then use the backend-neutral path."""
        if failures and self.config_obj.runtime.fail_on_client_failure:
            raise RuntimeError(
                f"Physical round {server_round}: {len(failures)} client job(s) "
                "failed. The run is aborted so invalid metrics are not recorded. "
                f"First failure: {failures[0]!r}"
            )
        if not results:
            raise RuntimeError(
                f"Physical round {server_round}: Flower returned zero successful "
                "client results."
            )

        payloads: List[ClientFitPayload] = []
        for _client_proxy, fit_res in results:
            payloads.append(
                ClientFitPayload(
                    arrays=[np.asarray(value) for value in parameters_to_ndarrays(
                        fit_res.parameters
                    )],
                    num_examples=int(fit_res.num_examples),
                    metrics=dict(fit_res.metrics or {}),
                )
            )
        return self.aggregate_payloads(int(server_round), payloads)

    def _sparse_proposed_active(self) -> bool:
        return bool(
            self.method == "proposed"
            and self.config_obj.sparse.enabled
            and float(self.config_obj.sparse.keep_ratio) < 1.0
        )

    def _phase_payload_dimension(self) -> int:
        if self.method == "proposed" and self.config_obj.sparse.enabled:
            return kept_coordinate_count(
                self.dimension,
                float(self.config_obj.sparse.keep_ratio),
                int(self.config_obj.sparse.min_keep),
            )
        return int(self.dimension)

    def aggregate_payloads(
        self,
        physical_round: int,
        payloads: Sequence[ClientFitPayload],
    ) -> tuple[Parameters, Dict[str, float]]:
        """Aggregate one physical phase from either execution backend."""
        context = self.context_for_round(physical_round)
        if not payloads:
            raise RuntimeError(
                f"Physical round {physical_round}: no client payloads to aggregate"
            )
        expected = int(self.config_obj.data.num_clients)
        if len(payloads) != expected and self.config_obj.runtime.fail_on_client_failure:
            raise RuntimeError(
                f"Physical round {physical_round}: expected {expected} client "
                f"payloads, received {len(payloads)}."
            )

        # Flower/Ray result arrival order is not guaranteed. Stable client order
        # is required so one channel row always belongs to the same client in
        # both phases of a logical Bayesian round.
        ordered = sorted(
            payloads,
            key=lambda payload: int(payload.metrics.get("client_id", -1)),
        )
        local_arrays: List[List[np.ndarray]] = []
        examples: List[int] = []
        distances: List[float] = []
        losses: List[float] = []

        for payload in ordered:
            payload_dtype = (
                np.float64
                if self.method == "proposed" and context.phase == PRECISION_PHASE
                else np.float32
            )
            arrays = [np.asarray(value, dtype=payload_dtype) for value in payload.arrays]
            local_arrays.append(arrays)
            examples.append(int(payload.num_examples))
            metrics = payload.metrics
            distances.append(float(metrics.get("distance_m", 1.0)))
            losses.append(float(metrics.get("train_loss", float("nan"))))
            self.logger.clients.append(
                {
                    "run_id": self.run_spec.run_id,
                    "round": int(context.logical_round),
                    "logical_round": int(context.logical_round),
                    "physical_round": int(context.physical_round),
                    "phase": str(context.phase),
                    "client_id": int(metrics.get("client_id", -1)),
                    "num_examples": int(payload.num_examples),
                    "distance_m": float(metrics.get("distance_m", float("nan"))),
                    "train_loss": float(metrics.get("train_loss", float("nan"))),
                    "phase1_loss": float(metrics.get("phase1_loss", 0.0)),
                    "phase2_loss": float(metrics.get("phase2_loss", 0.0)),
                    "local_steps": int(metrics.get("local_steps", 0)),
                    "local_precision_mean": float(
                        metrics.get("local_precision_mean", float("nan"))
                    ),
                    "local_precision_min": float(
                        metrics.get("local_precision_min", float("nan"))
                    ),
                    "local_precision_max": float(
                        metrics.get("local_precision_max", float("nan"))
                    ),
                    "local_precision_delta_l2": float(
                        metrics.get("local_precision_delta_l2", float("nan"))
                    ),
                    "local_precision_delta_max_abs": float(
                        metrics.get(
                            "local_precision_delta_max_abs", float("nan")
                        )
                    ),
                    "local_precision_changed_fraction": float(
                        metrics.get(
                            "local_precision_changed_fraction", float("nan")
                        )
                    ),
                    "local_precision_gradient_l2_mean": float(
                        metrics.get(
                            "local_precision_gradient_l2_mean", float("nan")
                        )
                    ),
                    "local_precision_gradient_max_abs": float(
                        metrics.get(
                            "local_precision_gradient_max_abs", float("nan")
                        )
                    ),
                    "local_nu_l2": float(metrics.get("local_nu_l2", 0.0)),
                    "local_implied_mean_l2": float(
                        metrics.get("local_implied_mean_l2", 0.0)
                    ),
                    "sparse_enabled": bool(metrics.get("sparse_enabled", False)),
                    "sparse_selection": str(metrics.get("sparse_selection", "")),
                    "sparse_keep_ratio": float(metrics.get("sparse_keep_ratio", 1.0)),
                    "sparse_kept_coordinates": int(metrics.get("sparse_kept_coordinates", 0)),
                    "sparse_total_coordinates": int(metrics.get("sparse_total_coordinates", 0)),
                    "sparse_score_threshold": float(metrics.get("sparse_score_threshold", float("nan"))),
                    "sparse_score_mean": float(metrics.get("sparse_score_mean", float("nan"))),
                    "sparse_selected_score_mean": float(metrics.get("sparse_selected_score_mean", float("nan"))),
                    "sparse_dropped_score_mean": float(metrics.get("sparse_dropped_score_mean", float("nan"))),
                }
            )

        weights = normalized_weights(examples)
        weighted_loss = self._weighted_finite_mean(losses, weights)
        channels = self._sample_channels(context.logical_round, distances)
        phase_rng = self._phase_rng(context)

        if self.method in {"fedavg", "fedprox"}:
            self._expect_array_count(local_arrays, 1, context)
            previous_model = self.current_arrays[0].astype(np.float64, copy=True)
            aggregation = aggregate_deterministic(
                self.current_arrays[0],
                [arrays[0] for arrays in local_arrays],
                weights,
                channels,
                self.config_obj.wireless,
                phase_rng,
            )
            self.current_arrays = [aggregation.parameters[0].astype(np.float32)]
            model_delta = self.current_arrays[0].astype(np.float64) - previous_model
            self.last_global_mean_update_l2 = float(np.linalg.vector_norm(model_delta))
            self.last_global_mean_update_max_abs = float(np.max(np.abs(model_delta)))
            self.last_global_model_update_l2 = float(
                aggregation.diagnostics.get(
                    "global_model_update_l2", self.last_global_mean_update_l2
                )
            )
            self.last_ideal_model_update_l2 = float(
                aggregation.diagnostics.get("ideal_model_update_l2", 0.0)
            )
            self.last_received_model_update_l2 = float(
                aggregation.diagnostics.get("received_model_update_l2", 0.0)
            )
            self.last_global_precision_update_l2 = 0.0
            self.last_global_precision_update_max_abs = 0.0
            self.last_train_loss = weighted_loss
            self.last_phase1_train_loss = 0.0
            self.last_phase2_train_loss = weighted_loss
            self.last_aircomp_stats = aggregation.aircomp_stats
            self.last_precision_aircomp_stats = AirCompStats.zero()
            self.last_mean_aircomp_stats = aggregation.aircomp_stats
            phase_multiplier = 1

        elif self.method == "scaffold":
            self._expect_array_count(local_arrays, 2, context)
            previous_model = self.current_arrays[0].astype(np.float64, copy=True)
            aggregation = aggregate_scaffold(
                self.current_arrays[0],
                self.current_arrays[1],
                [arrays[0] for arrays in local_arrays],
                [arrays[1] for arrays in local_arrays],
                weights,
                channels,
                self.config_obj.wireless,
                phase_rng,
            )
            self.current_arrays = [
                array.astype(np.float32) for array in aggregation.parameters
            ]
            model_delta = self.current_arrays[0].astype(np.float64) - previous_model
            self.last_global_mean_update_l2 = float(np.linalg.vector_norm(model_delta))
            self.last_global_mean_update_max_abs = float(np.max(np.abs(model_delta)))
            self.last_global_model_update_l2 = self.last_global_mean_update_l2
            # The SCAFFOLD AirCompStats object combines model and control-variate
            # transmissions, so the FedAvg-specific ideal/received model-update
            # diagnostics are intentionally left at zero here.
            self.last_ideal_model_update_l2 = 0.0
            self.last_received_model_update_l2 = 0.0
            self.last_global_precision_update_l2 = 0.0
            self.last_global_precision_update_max_abs = 0.0
            self.last_train_loss = weighted_loss
            self.last_phase1_train_loss = 0.0
            self.last_phase2_train_loss = weighted_loss
            self.last_aircomp_stats = aggregation.aircomp_stats
            self.last_precision_aircomp_stats = AirCompStats.zero()
            self.last_mean_aircomp_stats = aggregation.aircomp_stats
            phase_multiplier = 2

        elif self.method == "proposed" and context.phase == PRECISION_PHASE:
            self._expect_array_count(local_arrays, 1, context)
            previous_precision = self.current_arrays[1].astype(np.float64, copy=True)
            aggregation = aggregate_gaussian_precision_phase(
                current_precision=self.current_arrays[1],
                local_precisions=[arrays[0] for arrays in local_arrays],
                weights=weights,
                channels=channels,
                wireless_cfg=self.config_obj.wireless,
                model_cfg=self.config_obj.model,
                rng=phase_rng,
                sparse_missing_is_silent=self._sparse_proposed_active(),
            )
            # Critical Algorithm-1 boundary: keep mu_t unchanged and broadcast
            # the newly aggregated rho_{t+1} before phase 2 starts.  Phase 2
            # also needs rho_t because Eq. (15) regularizes against the
            # round-start global posterior q_{theta_t}.  The temporary third
            # array is removed after the natural-mean aggregation.
            round_start_precision = self.current_arrays[1].astype(
                np.float64, copy=True
            )
            self.current_arrays = [
                self.current_arrays[0].astype(np.float32, copy=True),
                aggregation.parameters[0].astype(np.float64),
                round_start_precision,
            ]
            precision_delta = self.current_arrays[1] - previous_precision
            self.last_global_precision_update_l2 = float(
                np.linalg.vector_norm(precision_delta)
            )
            self.last_global_precision_update_max_abs = float(
                np.max(np.abs(precision_delta))
            )
            self.last_global_model_update_l2 = 0.0
            self.last_ideal_model_update_l2 = 0.0
            self.last_received_model_update_l2 = 0.0
            # Keep the previous logical round's mean-update diagnostic until
            # phase 2 completes; it is overwritten below before evaluation.
            self.last_phase1_train_loss = weighted_loss
            self.last_phase2_train_loss = 0.0
            self.last_train_loss = weighted_loss
            self.last_precision_aircomp_stats = aggregation.aircomp_stats
            self.last_mean_aircomp_stats = AirCompStats.zero()
            self.last_aircomp_stats = aggregation.aircomp_stats
            self.pending_precision_logical_round = context.logical_round
            phase_multiplier = 1

        elif self.method == "proposed" and context.phase == NATURAL_MEAN_PHASE:
            if self.pending_precision_logical_round != context.logical_round:
                raise RuntimeError(
                    "Natural-mean phase started without the matching server-side "
                    f"precision aggregation. pending={self.pending_precision_logical_round}, "
                    f"requested={context.logical_round}."
                )
            self._expect_array_count(local_arrays, 1, context)
            previous_mean = self.current_arrays[0].astype(np.float64, copy=True)
            aggregation = aggregate_gaussian_natural_mean_phase(
                current_mean=self.current_arrays[0],
                local_nus=[arrays[0] for arrays in local_arrays],
                weights=weights,
                channels=channels,
                wireless_cfg=self.config_obj.wireless,
                rng=phase_rng,
                sparse_missing_is_silent=self._sparse_proposed_active(),
            )
            if len(self.current_arrays) != 3:
                raise RuntimeError(
                    "Natural-mean phase requires [mu_t, rho_{t+1}, rho_t] "
                    f"from the preceding precision phase; got "
                    f"{len(self.current_arrays)} arrays"
                )
            self.current_arrays = [
                aggregation.parameters[0].astype(np.float32),
                self.current_arrays[1].astype(np.float64, copy=True),
            ]
            mean_delta = self.current_arrays[0].astype(np.float64) - previous_mean
            self.last_global_mean_update_l2 = float(np.linalg.vector_norm(mean_delta))
            self.last_global_mean_update_max_abs = float(np.max(np.abs(mean_delta)))
            self.last_phase2_train_loss = weighted_loss
            self.last_train_loss = (
                self.last_phase1_train_loss + self.last_phase2_train_loss
            )
            self.last_mean_aircomp_stats = aggregation.aircomp_stats
            self.last_aircomp_stats = combine_stats(
                self.last_precision_aircomp_stats,
                self.last_mean_aircomp_stats,
            )
            self.pending_precision_logical_round = None
            phase_multiplier = 1

        else:
            raise ValueError(
                f"Unknown method/phase combination: {self.method}/{context.phase}"
            )

        phase_payload_dimension = self._phase_payload_dimension()
        self.channel_uses_cumulative += phase_multiplier * phase_payload_dimension
        self.ofdm_symbols_cumulative += phase_multiplier * math.ceil(
            phase_payload_dimension / self.config_obj.wireless.num_subchannels
        )
        parameters = ndarrays_to_parameters(self.current_arrays)
        return parameters, {
            "train_loss": float(self.last_train_loss),
            "phase_train_loss": float(weighted_loss),
            "aircomp_nmse": float(aggregation.aircomp_stats.nmse),
            "logical_round": float(context.logical_round),
            "physical_round": float(context.physical_round),
        }

    def evaluate(self, server_round: int, parameters: Parameters):  # type: ignore[override]
        """Evaluate at round zero and after complete logical rounds only."""
        context = self.context_for_round(int(server_round))
        if self.method == "proposed" and context.phase == PRECISION_PHASE:
            return None

        logical_round = int(context.logical_round)
        if (
            logical_round != 0
            and logical_round != self.run_spec.rounds
            and logical_round % self.config_obj.training.evaluate_every != 0
        ):
            return None

        # Do not blanket-cast decoded Flower parameters to float32.  The
        # proposed method keeps rho in float64; downcasting here used to erase
        # the sub-float32-ULP precision update after every logical round.
        arrays = normalize_server_state_dtypes(
            self.method, parameters_to_ndarrays(parameters)
        )
        self.current_arrays = [value.copy() for value in arrays]

        if self.method == "proposed":
            if len(arrays) != 2:
                raise ValueError("Proposed evaluation expects [global_mean, precision]")
            evaluation = evaluate_bayesian(
                self.model,
                self.layout,
                arrays[0],
                arrays[1],
                self.test_loader,
                self.server_device,
                mc_samples=self.config_obj.training.mc_eval_samples,
                seed=self.run_spec.seed + logical_round,
            )
            posterior_mean_evaluation = evaluate_bayesian_mean(
                self.model,
                self.layout,
                arrays[0],
                self.test_loader,
                self.server_device,
            )
            posterior_variance = float(
                np.mean(1.0 / np.maximum(arrays[1], 1.0e-12))
            )
        else:
            evaluation = evaluate_deterministic(
                self.model,
                self.layout,
                arrays[0],
                self.test_loader,
                self.server_device,
            )
            posterior_mean_evaluation = evaluation
            posterior_variance = 0.0

        multiplier = payload_multiplier(self.method)
        payload_dimension = self._phase_payload_dimension()
        channel_uses_round = (
            0 if logical_round == 0 else multiplier * payload_dimension
        )
        ofdm_symbols_round = (
            0
            if logical_round == 0
            else multiplier
            * math.ceil(
                payload_dimension / self.config_obj.wireless.num_subchannels
            )
        )
        base = {
            "run_id": self.run_spec.run_id,
            "experiment": self.run_spec.experiment,
            "condition": self.run_spec.condition,
            "method": self.method,
            "realization": self.run_spec.realization,
            "seed": self.run_spec.seed,
            "round": logical_round,
            "logical_round": logical_round,
            "physical_round": int(context.physical_round),
            "phase": MODEL_PHASE if logical_round == 0 else context.phase,
            "num_clients": self.config_obj.data.num_clients,
            "labels_per_client": self.config_obj.data.labels_per_client,
            "mean_samples_per_client": self.config_obj.data.mean_samples_per_client,
            "power_dbm": self.config_obj.wireless.power_dbm,
            "noise_dbm": self.config_obj.wireless.noise_dbm,
            "num_subchannels": self.config_obj.wireless.num_subchannels,
            "path_loss_exponent": self.config_obj.wireless.path_loss_exponent,
            "path_loss_reference_m": self.config_obj.wireless.path_loss_reference_m,
            "gamma_db": self.config_obj.wireless.gamma_db,
            "power_control_mode": self.config_obj.wireless.power_control_mode,
            "deterministic_payload_mode": (
                self.config_obj.wireless.deterministic_payload_mode
            ),
            "deterministic_reference_power_mode": (
                self.config_obj.wireless.deterministic_reference_power_mode
            ),
            "sparse_enabled": bool(
                self.method == "proposed" and self.config_obj.sparse.enabled
            ),
            "sparse_selection": (
                str(self.config_obj.sparse.selection)
                if self.method == "proposed" and self.config_obj.sparse.enabled
                else ""
            ),
            "sparse_keep_ratio": (
                float(self.config_obj.sparse.keep_ratio)
                if self.method == "proposed" and self.config_obj.sparse.enabled
                else 1.0
            ),
            "sparse_kept_coordinates": (
                self._phase_payload_dimension()
                if self.method == "proposed" and self.config_obj.sparse.enabled
                else self.dimension
            ),
        }
        if self.method == "proposed":
            initial_precision_value = 1.0 / (
                self.config_obj.model.initial_prior_std**2
            )
            precision_offset = arrays[1] - initial_precision_value
            posterior_precision_std = float(np.std(arrays[1], dtype=np.float64))
            posterior_precision_offset_l2 = float(
                np.linalg.vector_norm(precision_offset.astype(np.float64))
            )
            posterior_precision_offset_max_abs = float(
                np.max(np.abs(precision_offset))
            )
        else:
            posterior_precision_std = 0.0
            posterior_precision_offset_l2 = 0.0
            posterior_precision_offset_max_abs = 0.0

        self.logger.metrics.append(
            {
                **base,
                "accuracy": evaluation.accuracy,
                "nll": evaluation.nll,
                "ece": evaluation.ece,
                "posterior_predictive_accuracy": (
                    evaluation.accuracy if self.method == "proposed" else 0.0
                ),
                "posterior_predictive_nll": (
                    evaluation.nll if self.method == "proposed" else 0.0
                ),
                "posterior_predictive_ece": (
                    evaluation.ece if self.method == "proposed" else 0.0
                ),
                "posterior_mean_accuracy": (
                    posterior_mean_evaluation.accuracy
                    if self.method == "proposed" else evaluation.accuracy
                ),
                "posterior_mean_nll": (
                    posterior_mean_evaluation.nll
                    if self.method == "proposed" else evaluation.nll
                ),
                "posterior_mean_ece": (
                    posterior_mean_evaluation.ece
                    if self.method == "proposed" else evaluation.ece
                ),
                "train_loss": self.last_train_loss,
                "phase1_train_loss": self.last_phase1_train_loss,
                "phase2_train_loss": self.last_phase2_train_loss,
                "posterior_variance": posterior_variance,
                "posterior_precision_mean": (
                    float(np.mean(arrays[1])) if self.method == "proposed" else 0.0
                ),
                "posterior_precision_min": (
                    float(np.min(arrays[1])) if self.method == "proposed" else 0.0
                ),
                "posterior_precision_max": (
                    float(np.max(arrays[1])) if self.method == "proposed" else 0.0
                ),
                "posterior_precision_std": posterior_precision_std,
                "posterior_precision_offset_l2": posterior_precision_offset_l2,
                "posterior_precision_offset_max_abs": (
                    posterior_precision_offset_max_abs
                ),
                "global_mean_update_l2": (
                    0.0 if logical_round == 0 else self.last_global_mean_update_l2
                ),
                "global_mean_update_max_abs": (
                    0.0 if logical_round == 0 else self.last_global_mean_update_max_abs
                ),
                "global_model_update_l2": (
                    0.0 if logical_round == 0 else self.last_global_model_update_l2
                ),
                "ideal_model_update_l2": (
                    0.0 if logical_round == 0 else self.last_ideal_model_update_l2
                ),
                "received_model_update_l2": (
                    0.0 if logical_round == 0 else self.last_received_model_update_l2
                ),
                "global_precision_update_l2": (
                    0.0 if logical_round == 0 else self.last_global_precision_update_l2
                ),
                "global_precision_update_max_abs": (
                    0.0
                    if logical_round == 0
                    else self.last_global_precision_update_max_abs
                ),
                "channel_uses_round": channel_uses_round,
                "channel_uses_cumulative": self.channel_uses_cumulative,
                "ofdm_symbols_round": ofdm_symbols_round,
                "ofdm_symbols_cumulative": self.ofdm_symbols_cumulative,
                **self._stats_row("aircomp", self.last_aircomp_stats),
                **self._stats_row(
                    "precision_aircomp", self.last_precision_aircomp_stats
                ),
                **self._stats_row("mean_aircomp", self.last_mean_aircomp_stats),
                "wall_time_sec": time.perf_counter() - self.started_at,
            }
        )
        self.logger.log_reliability(base, evaluation)

        if logical_round == self.run_spec.rounds and self.config_obj.output.save_checkpoints:
            self.logger.save_checkpoint(
                self.run_spec.run_id,
                arrays,
                {
                    **base,
                    "accuracy": evaluation.accuracy,
                    "ece": evaluation.ece,
                    "nll": evaluation.nll,
                    "protocol": "server_separated_rho_nu",
                },
            )
        return float(evaluation.nll), {
            "accuracy": float(evaluation.accuracy),
            "ece": float(evaluation.ece),
        }

    def _sample_channels(
        self,
        logical_round: int,
        distances: Sequence[float],
    ) -> np.ndarray:
        # Both proposed phases reuse the same block-fading realization. Noise
        # remains independent because phase-specific RNGs are used below.
        channel_rng = np.random.default_rng(
            self.run_spec.seed + 1_000_033 * int(logical_round)
        )
        return sample_rayleigh_channels(
            np.asarray(distances, dtype=np.float64),
            self.config_obj.wireless.num_subchannels,
            self.config_obj.wireless.path_loss_exponent,
            channel_rng,
            self.config_obj.wireless.path_loss_reference_m,
        )

    def _phase_rng(self, context: PhaseContext) -> np.random.Generator:
        phase_offset = {
            MODEL_PHASE: 11,
            PRECISION_PHASE: 101,
            NATURAL_MEAN_PHASE: 211,
        }[context.phase]
        return np.random.default_rng(
            self.run_spec.seed
            + 9_000_091 * int(context.logical_round)
            + phase_offset
        )

    @staticmethod
    def _weighted_finite_mean(losses: Sequence[float], weights: np.ndarray) -> float:
        values = np.asarray(losses, dtype=np.float64)
        mask = np.isfinite(values)
        if not np.any(mask):
            return float("nan")
        selected = np.asarray(weights, dtype=np.float64)[mask]
        selected = selected / max(float(selected.sum()), 1.0e-30)
        return float(np.dot(selected, values[mask]))

    @staticmethod
    def _expect_array_count(
        local_arrays: Sequence[Sequence[np.ndarray]],
        count: int,
        context: PhaseContext,
    ) -> None:
        invalid = [index for index, arrays in enumerate(local_arrays) if len(arrays) != count]
        if invalid:
            raise ValueError(
                f"Phase {context.phase} expected {count} returned array(s) per "
                f"client; invalid payload indices: {invalid[:5]}"
            )

    @staticmethod
    def _stats_row(prefix: str, stats: AirCompStats) -> Dict[str, float]:
        return {
            f"{prefix}_nmse": stats.nmse,
            f"{prefix}_distortion_nmse": stats.distortion_nmse,
            f"{prefix}_clipped_fraction": stats.clipped_fraction,
            f"{prefix}_average_symbol_power_watts": stats.average_symbol_power_watts,
            f"{prefix}_maximum_symbol_power_watts": stats.maximum_symbol_power_watts,
            f"{prefix}_noise_l2": stats.noise_l2,
            f"{prefix}_ideal_l2": stats.ideal_l2,
            f"{prefix}_received_l2": stats.received_l2,
            f"{prefix}_delta_bar": stats.delta_bar,
            f"{prefix}_retained_magnitude_ratio": stats.retained_magnitude_ratio,
            f"{prefix}_distorted_to_ideal_norm_ratio": stats.distorted_to_ideal_norm_ratio,
        }


def _prepare_state_dir(config: SimulationConfig, run_spec: RunSpec) -> Path:
    state_dir = Path(config.output.directory) / "client_state" / run_spec.run_id
    if state_dir.exists():
        shutil.rmtree(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def run_local_simulation(
    config: SimulationConfig,
    run_spec: RunSpec,
    partition_path: str,
) -> None:
    """Run physical client phases sequentially in the launcher process."""
    state_dir = _prepare_state_dir(config, run_spec)
    strategy = AirCompStrategy(
        config=config,
        run_spec=run_spec,
        partition_path=partition_path,
    )

    local_client_device = resolve_device(config.runtime.client_device)
    print(
        "Local backend: clients execute sequentially; "
        f"client_device={local_client_device}, "
        f"server_device={config.runtime.server_device}."
    )
    if run_spec.method == "proposed":
        print(
            "Proposed protocol: each logical round executes precision -> "
            "server aggregation/broadcast -> natural_mean."
        )

    initial = strategy.evaluate(0, ndarrays_to_parameters(strategy.current_arrays))
    if initial is not None:
        initial_loss, initial_metrics = initial
        print(
            f"Round 0: loss={initial_loss:.6f}, "
            f"accuracy={initial_metrics['accuracy']:.4f}, "
            f"ece={initial_metrics['ece']:.4f}"
        )

    for physical_round in range(1, strategy.physical_rounds + 1):
        phase_started = time.perf_counter()
        context = strategy.context_for_round(physical_round)
        global_arrays = [value.copy() for value in strategy.current_arrays]
        payloads: List[ClientFitPayload] = []
        fit_config = strategy.fit_config_for_round(physical_round)

        for client_id in range(int(config.data.num_clients)):
            client: AirCompNumPyClient | None = None
            try:
                client = AirCompNumPyClient(
                    client_id=client_id,
                    method=run_spec.method,
                    config=config,
                    partition_path=partition_path,
                    run_seed=run_spec.seed,
                    state_dir=str(state_dir),
                )
                arrays, num_examples, metrics = client.fit(
                    [value.copy() for value in global_arrays],
                    fit_config,
                )
                payloads.append(
                    ClientFitPayload(
                        arrays=[np.asarray(value) for value in arrays],
                        num_examples=int(num_examples),
                        metrics=dict(metrics),
                    )
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Local backend client {client_id} failed in logical round "
                    f"{context.logical_round}, phase {context.phase}: {exc}"
                ) from exc
            finally:
                if client is not None:
                    del client
                gc.collect()
                if local_client_device.type == "cuda":
                    release_cuda_memory(local_client_device)

        parameters, aggregate_metrics = strategy.aggregate_payloads(
            physical_round, payloads
        )
        evaluation = strategy.evaluate(physical_round, parameters)
        elapsed = time.perf_counter() - phase_started

        if evaluation is None:
            print(
                f"Logical round {context.logical_round}/{run_spec.rounds}, "
                f"phase={context.phase}: "
                f"phase_loss={aggregate_metrics['phase_train_loss']:.6f}, "
                f"aircomp_nmse={aggregate_metrics['aircomp_nmse']:.6f}, "
                f"elapsed={elapsed:.2f}s"
            )
        else:
            loss, metrics = evaluation
            print(
                f"Round {context.logical_round}/{run_spec.rounds}: "
                f"loss={loss:.6f}, accuracy={metrics['accuracy']:.4f}, "
                f"ece={metrics['ece']:.4f}, "
                f"train_loss={aggregate_metrics['train_loss']:.6f}, "
                f"elapsed={elapsed:.2f}s"
            )


def run_flower_simulation(
    config: SimulationConfig,
    run_spec: RunSpec,
    partition_path: str,
) -> None:
    """Execute one simulation with Flower's Ray backend."""
    state_dir = _prepare_state_dir(config, run_spec)

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
            config=ServerConfig(num_rounds=strategy.physical_rounds),
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
        try:
            import ray

            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass


def run_configured_simulation(
    config: SimulationConfig,
    run_spec: RunSpec,
    partition_path: str,
) -> None:
    """Dispatch to the resolved backend."""
    backend = resolve_backend(config)
    if backend == "local":
        run_local_simulation(config, run_spec, partition_path)
    elif backend == "ray":
        run_flower_simulation(config, run_spec, partition_path)
    else:
        raise ValueError(f"Unsupported runtime backend: {backend}")
