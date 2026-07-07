"""Custom Flower strategy for grouped physical-device Bayesian FL with rich logs."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common import FitIns, Metrics, NDArrays, Parameters, Scalar
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from torch.utils.data import DataLoader

import model
import observability as obs
from config import RunConfig
from selector import SelectionResult, build_selector


class GroupedBayesStrategy(FedAvg):
    """Server strategy handling FedAvg, VI posterior aggregation, and FOLA.

    The server samples physical devices, sends that selection to every Flower
    virtual client, and receives one grouped response per virtual client. In
    addition to the global model/posterior, this strategy collects round-level,
    client-level, posterior, calibration, selection, aggregation, and placeholder
    communication metrics.
    """

    def __init__(
        self,
        cfg: RunConfig,
        initial_payload: NDArrays,
        testloader: DataLoader,
        input_shape: Sequence[int],
        num_classes: int,
        output_dir: Path,
        client_sizes: Sequence[int],
        label_counts: np.ndarray,
        device_positions: np.ndarray,
        device_groups: Sequence[Sequence[int]],
    ) -> None:
        self.cfg = cfg
        self.testloader = testloader
        self.input_shape = tuple(int(x) for x in input_shape)
        self.num_classes = int(num_classes)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = self.output_dir.name
        self.selector = build_selector(cfg.selector, cfg.seed)
        self.device = model.resolve_device(cfg.device)
        self.latest_payload = [np.asarray(x, dtype=np.float32).copy() for x in initial_payload]
        self.last_payload_before_aggregation = [np.asarray(x, dtype=np.float32).copy() for x in initial_payload]
        self.client_sizes = [int(x) for x in client_sizes]
        self.label_counts = np.asarray(label_counts, dtype=np.int64)
        self.device_positions = np.asarray(device_positions, dtype=np.float64)
        self.device_groups = [list(map(int, g)) for g in device_groups]
        self.label_entropy, self.label_kl, self.dominant_label, self.dominant_label_fraction = obs.label_metadata(self.label_counts)
        self.gid_lookup = {did: gid for gid, members in enumerate(self.device_groups) for did in members}
        self.base_eval_model = model.build_model(cfg, self.input_shape, self.num_classes)
        self.param_meta = model.parameter_metadata(self.base_eval_model)

        self.history_rows: list[dict[str, Any]] = []
        self.run_summary_rows: list[dict[str, Any]] = []
        self.selection_summary_rows: list[dict[str, Any]] = []
        self.selected_client_rows: list[dict[str, Any]] = []
        self.communication_rows: list[dict[str, Any]] = []
        self.client_train_rows: list[dict[str, Any]] = []
        self.client_eval_rows: list[dict[str, Any]] = []
        self.calibration_rows: list[dict[str, Any]] = []
        self.posterior_summary_rows: list[dict[str, Any]] = []
        self.snr_histogram_rows: list[dict[str, Any]] = []
        self.aggregation_rows: list[dict[str, Any]] = []
        self.last_selection = SelectionResult(0, [], cfg.selector)
        self.last_fit_metrics: dict[str, Any] = {}
        self.round_start_time: dict[int, float] = {}
        self.fit_start_time: dict[int, float] = {}
        self.aggregate_time_sec: float = 0.0
        self.start_time = time.perf_counter()

        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=int(cfg.num_virtual_clients),
            min_available_clients=int(cfg.num_virtual_clients),
            min_evaluate_clients=0,
            initial_parameters=ndarrays_to_parameters(self.latest_payload),
            accept_failures=bool(cfg.accept_failures),
        )

    # ------------------------------------------------------------------
    # Selection/configuration
    # ------------------------------------------------------------------
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        del parameters
        now = time.perf_counter()
        self.round_start_time[int(server_round)] = now
        self.fit_start_time[int(server_round)] = now
        selection = self.selector.select(
            round_idx=int(server_round),
            num_devices=int(self.cfg.num_devices),
            fraction=float(self.cfg.client_fraction),
        )
        self.last_selection = selection
        self._record_selection_rows(server_round, selection)

        fit_config: Dict[str, Scalar] = {
            "server_round": int(server_round),
            "selected_ids": selection.as_csv_string(),
            "method": self.cfg.method,
        }
        fit_ins = FitIns(ndarrays_to_parameters(self.latest_payload), fit_config)
        clients = client_manager.sample(
            num_clients=int(self.cfg.num_virtual_clients),
            min_num_clients=int(self.cfg.num_virtual_clients),
        )
        return [(client_proxy, fit_ins) for client_proxy in clients]

    def _record_selection_rows(self, server_round: int, selection: SelectionResult) -> None:
        selected = list(map(int, selection.selected_ids))
        selected_set = set(selected)
        counts = np.asarray([self.client_sizes[did] for did in selected], dtype=np.float64) if selected else np.asarray([], dtype=np.float64)
        ent = self.label_entropy[selected] if selected else np.asarray([], dtype=np.float64)
        kl = self.label_kl[selected] if selected else np.asarray([], dtype=np.float64)
        dist = self.device_positions[selected, 2] if selected else np.asarray([], dtype=np.float64)
        selected_examples = int(counts.sum()) if counts.size else 0
        row = {
            "schema_version": obs.SCHEMA_VERSION,
            "run_id": self.run_id,
            "round": int(server_round),
            "method": self.cfg.method,
            "selection_policy": selection.policy_name,
            "selected_count": int(len(selected)),
            "available_count": int(self.cfg.num_devices),
            "selected_fraction": float(len(selected) / max(int(self.cfg.num_devices), 1)),
            "selected_examples": int(selected_examples),
            "selected_examples_fraction": float(selected_examples / max(sum(self.client_sizes), 1)),
            "selected_label_entropy_mean": float(ent.mean()) if ent.size else obs.nan(),
            "selected_label_entropy_std": float(ent.std()) if ent.size else obs.nan(),
            "selected_kl_to_global_label_mean": float(kl.mean()) if kl.size else obs.nan(),
            "selected_distance_m_mean": float(dist.mean()) if dist.size else obs.nan(),
            "selected_distance_m_std": float(dist.std()) if dist.size else obs.nan(),
            "selected_channel_snr_db_mean": obs.nan(),
            "selected_channel_snr_db_std": obs.nan(),
            "selection_score_mean": obs.nan(),
            "selection_score_std": obs.nan(),
        }
        self.selection_summary_rows.append(row)
        for did in selected:
            pos = self.device_positions[did]
            client_row = {
                "schema_version": obs.SCHEMA_VERSION,
                "run_id": self.run_id,
                "round": int(server_round),
                "physical_client_id": int(did),
                "virtual_client_id": int(self.gid_lookup.get(did, -1)),
                "selected": True,
                "selected_count": int(len(selected)),
                "selection_policy": selection.policy_name,
                "selection_score": obs.nan(),
                "selection_probability": float(self.cfg.client_fraction),
                "selection_reason": "random_uniform" if selection.policy_name == "random" else selection.policy_name,
                "num_examples": int(self.client_sizes[did]),
                "label_entropy": float(self.label_entropy[did]),
                "dominant_label": int(self.dominant_label[did]),
                "kl_to_global_label_distribution": float(self.label_kl[did]),
                "distance_m": float(pos[2]),
                "angle_rad": float(pos[3]),
                "channel_gain": obs.nan(),
                "channel_snr_db": obs.nan(),
                "pathloss_db": obs.nan(),
                "rate_mbps": obs.nan(),
                "delay_ms": obs.nan(),
                "energy_j": obs.nan(),
                "outage": "",
            }
            self.selected_client_rows.append(client_row)
            comm_row = {
                "schema_version": obs.SCHEMA_VERSION,
                "run_id": self.run_id,
                "round": int(server_round),
                "physical_client_id": int(did),
                "virtual_client_id": int(self.gid_lookup.get(did, -1)),
                "selected": True,
                "selection_policy": selection.policy_name,
                "distance_m": float(pos[2]),
                "angle_rad": float(pos[3]),
                "channel_gain": obs.nan(),
                "channel_snr_db": obs.nan(),
                "pathloss_db": obs.nan(),
                "noise_power": obs.nan(),
                "tx_power": obs.nan(),
                "rate_mbps": obs.nan(),
                "delay_ms": obs.nan(),
                "energy_j": obs.nan(),
                "outage": "",
                "analog_ota_enabled": False,
                "ota_noise_power": obs.nan(),
                "ota_distortion": obs.nan(),
                "ota_mse": obs.nan(),
                "ota_contribution_norm": obs.nan(),
                "digital_enabled": False,
                "packet_error_rate": obs.nan(),
                "payload_bytes": obs.nan(),
                "communication_success": "",
            }
            self.communication_rows.append(comm_row)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Metrics]:
        aggregate_start = time.perf_counter()
        if failures and not bool(self.cfg.accept_failures):
            raise RuntimeError(f"Flower fit failures in round {server_round}: {failures}")
        if not results:
            return ndarrays_to_parameters(self.latest_payload), {}

        self.last_payload_before_aggregation = [np.asarray(x, dtype=np.float32).copy() for x in self.latest_payload]
        active_results = [(client, fit_res) for client, fit_res in results if int(fit_res.num_examples) > 0]
        total_examples = int(sum(int(fit_res.num_examples) for _, fit_res in active_results))
        active_virtual_clients = int(len(active_results))
        active_physical_devices = int(sum(int(fit_res.metrics.get("active_devices", 0)) for _, fit_res in results))
        avg_train_loss = 0.0
        if total_examples > 0:
            avg_train_loss = float(
                sum(float(fit_res.metrics.get("train_loss", 0.0)) * int(fit_res.num_examples) for _, fit_res in active_results)
                / total_examples
            )

        round_client_train_rows: list[dict[str, Any]] = []
        round_client_eval_rows: list[dict[str, Any]] = []
        for _client, fit_res in results:
            for row in self._load_metric_json(fit_res.metrics.get("client_train_json", "[]")):
                row["run_id"] = self.run_id
                self._augment_client_row(row)
                round_client_train_rows.append(row)
            for row in self._load_metric_json(fit_res.metrics.get("client_eval_json", "[]")):
                row["run_id"] = self.run_id
                self._augment_client_row(row)
                round_client_eval_rows.append(row)

        if total_examples == 0:
            new_payload = self.latest_payload
        elif self.cfg.method == "fedavg":
            new_payload = self._aggregate_fedavg(active_results, total_examples)
        elif self.cfg.method == "ola":
            new_payload = self._aggregate_product_precision(active_results, total_examples, return_scale=False)
        elif self.cfg.method == "vi":
            if self.cfg.bayes_aggregation == "product":
                new_payload = self._aggregate_product_precision(active_results, total_examples, return_scale=True)
            else:
                new_payload = self._aggregate_moment_match(active_results, total_examples)
        else:
            raise ValueError(f"Unsupported method: {self.cfg.method}")

        self.latest_payload = [np.asarray(x, dtype=np.float32).copy() for x in new_payload]
        self.aggregate_time_sec = time.perf_counter() - aggregate_start
        self.client_train_rows.extend(round_client_train_rows)
        self.client_eval_rows.extend(round_client_eval_rows)

        client_summary = obs.summarize_client_rows(round_client_train_rows)
        eval_summary = obs.summarize_eval_rows(round_client_eval_rows)
        weights = [float(fit_res.num_examples) for _, fit_res in active_results]
        agg_weight_stats = obs.aggregation_weight_stats(weights)
        before = np.asarray(self.last_payload_before_aggregation[0], dtype=np.float64)
        after = np.asarray(self.latest_payload[0], dtype=np.float64)
        delta = after - before
        agg_row: dict[str, Any] = {
            "schema_version": obs.SCHEMA_VERSION,
            "run_id": self.run_id,
            "round": int(server_round),
            "method": self.cfg.method,
            "aggregation_mode": self.cfg.bayes_aggregation if self.cfg.method == "vi" else self.cfg.method,
            "num_results_received": int(len(results)),
            "num_failures": int(len(failures)),
            "total_selected_examples": int(total_examples),
            "global_before_l2": float(np.linalg.norm(before)),
            "global_after_l2": float(np.linalg.norm(after)),
            "aggregation_delta_l2": float(np.linalg.norm(delta)),
            "aggregation_delta_linf": float(np.max(np.abs(delta))) if delta.size else 0.0,
            "aggregation_delta_cosine": obs.vector_cosine(before, after),
            "fedavg_equivalent_delta_l2": obs.nan(),
            "bayes_product_delta_l2": float(np.linalg.norm(delta)) if self.cfg.method in {"vi", "ola"} else obs.nan(),
            "bayes_vs_fedavg_delta_l2": obs.nan(),
            "aggregation_energy_before": obs.nan(),
            "aggregation_energy_after": obs.nan(),
            "aggregation_error_proxy": obs.nan(),
        }
        agg_row.update(agg_weight_stats)
        for key in ["client_update_l2_mean", "client_update_l2_std", "client_update_l2_min", "client_update_l2_max", "client_update_cosine_mean", "client_update_cosine_std"]:
            if key in client_summary:
                agg_row[key] = client_summary[key]
        self.aggregation_rows.append(agg_row)

        product_metrics: dict[str, Any] = {}
        if self.cfg.method in {"vi", "ola"} and len(self.latest_payload) > 1:
            _mu_tmp, _sigma_tmp, _precision_tmp = obs.posterior_arrays(self.cfg, self.latest_payload)
            if _precision_tmp is not None:
                product_metrics = {
                    "posterior_product_precision_mean": float(np.mean(_precision_tmp)),
                    "posterior_product_precision_std": float(np.std(_precision_tmp)),
                    "posterior_product_mu_norm": float(np.linalg.norm(_mu_tmp.astype(np.float64))),
                    "posterior_product_sigma_mean": float(np.mean(_sigma_tmp)) if _sigma_tmp is not None else obs.nan(),
                }

        self.last_fit_metrics = {
            "total_examples": int(total_examples),
            "selected_examples": int(total_examples),
            "active_virtual_clients": int(active_virtual_clients),
            "active_physical_devices": int(active_physical_devices),
            "train_loss": float(avg_train_loss),
            "train_loss_mean": float(client_summary.get("train_loss_mean", avg_train_loss)),
            "num_fit_failures": int(len(failures)),
            "fit_time_sec": float(time.perf_counter() - self.fit_start_time.get(int(server_round), aggregate_start)),
            "aggregate_time_sec": float(self.aggregate_time_sec),
        }
        self.last_fit_metrics.update(product_metrics)
        self.last_fit_metrics.update(client_summary)
        self.last_fit_metrics.update(eval_summary)
        self.last_fit_metrics.update(agg_weight_stats)
        self.last_fit_metrics.update({k: v for k, v in agg_row.items() if k.startswith("aggregation_") or k.startswith("posterior_product_")})

        metrics: Metrics = {k: v for k, v in self.last_fit_metrics.items() if isinstance(v, (int, float, str, bool))}
        metrics["selected_count"] = int(self.last_selection.selected_count)
        return ndarrays_to_parameters(self.latest_payload), metrics

    def _load_metric_json(self, value: Any) -> list[dict[str, Any]]:
        try:
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            parsed = json.loads(str(value))
            if isinstance(parsed, list):
                return [dict(x) for x in parsed if isinstance(x, dict)]
        except Exception:
            return []
        return []

    def _augment_client_row(self, row: dict[str, Any]) -> None:
        did_raw = row.get("physical_client_id", "")
        try:
            did = int(did_raw)
        except Exception:
            return
        if 0 <= did < len(self.client_sizes):
            row.setdefault("num_examples", int(self.client_sizes[did]))
            row.setdefault("num_examples_train", int(self.client_sizes[did]))
            row["label_entropy"] = float(self.label_entropy[did])
            row["kl_to_global_label_distribution"] = float(self.label_kl[did])
            row.setdefault("virtual_client_id", int(self.gid_lookup.get(did, -1)))

    def _aggregate_fedavg(self, active_results: List[Tuple[ClientProxy, fl.common.FitRes]], total_examples: int) -> NDArrays:
        weighted: np.ndarray | None = None
        for _client, fit_res in active_results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            avg_params = np.asarray(arrays[0], dtype=np.float64)
            contribution = avg_params * float(fit_res.num_examples)
            weighted = contribution if weighted is None else weighted + contribution
        assert weighted is not None
        return [(weighted / float(total_examples)).astype(np.float32)]

    def _aggregate_product_precision(
        self,
        active_results: List[Tuple[ClientProxy, fl.common.FitRes]],
        total_examples: int,
        return_scale: bool,
    ) -> NDArrays:
        precision_sum: np.ndarray | None = None
        precision_mu_sum: np.ndarray | None = None
        for _client, fit_res in active_results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            local_precision_sum = np.asarray(arrays[0], dtype=np.float64)
            local_precision_mu_sum = np.asarray(arrays[1], dtype=np.float64)
            precision_sum = local_precision_sum if precision_sum is None else precision_sum + local_precision_sum
            precision_mu_sum = local_precision_mu_sum if precision_mu_sum is None else precision_mu_sum + local_precision_mu_sum
        assert precision_sum is not None and precision_mu_sum is not None
        precision_safe = np.maximum(precision_sum, float(self.cfg.precision_floor))
        mu = precision_mu_sum / precision_safe
        global_precision = np.maximum(precision_sum / float(total_examples), float(self.cfg.precision_floor))
        self.last_fit_metrics["posterior_product_precision_mean"] = float(global_precision.mean())
        self.last_fit_metrics["posterior_product_precision_std"] = float(global_precision.std())
        self.last_fit_metrics["posterior_product_mu_norm"] = float(np.linalg.norm(mu.astype(np.float64)))
        self.last_fit_metrics["posterior_product_sigma_mean"] = float(np.sqrt(1.0 / global_precision).mean())
        if return_scale:
            scale = np.sqrt(1.0 / global_precision)
            scale = np.maximum(scale, float(self.cfg.vi_min_scale))
            return [mu.astype(np.float32), scale.astype(np.float32)]
        return [mu.astype(np.float32), global_precision.astype(np.float32)]

    def _aggregate_moment_match(self, active_results: List[Tuple[ClientProxy, fl.common.FitRes]], total_examples: int) -> NDArrays:
        loc_sum: np.ndarray | None = None
        second_sum: np.ndarray | None = None
        for _client, fit_res in active_results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            group_loc_sum = np.asarray(arrays[0], dtype=np.float64)
            group_second_sum = np.asarray(arrays[1], dtype=np.float64)
            loc_sum = group_loc_sum if loc_sum is None else loc_sum + group_loc_sum
            second_sum = group_second_sum if second_sum is None else second_sum + group_second_sum
        assert loc_sum is not None and second_sum is not None
        loc = loc_sum / float(total_examples)
        second = second_sum / float(total_examples)
        var = np.maximum(second - loc * loc, float(self.cfg.vi_min_scale) ** 2)
        return [loc.astype(np.float32), np.sqrt(var).astype(np.float32)]

    # ------------------------------------------------------------------
    # Global evaluation and Bayesian summaries
    # ------------------------------------------------------------------
    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Metrics]]:
        if server_round > 0 and server_round % int(self.cfg.eval_every) != 0 and server_round != int(self.cfg.num_rounds):
            return None
        eval_start = time.perf_counter()
        payload = parameters_to_ndarrays(parameters)
        # Always evaluate the posterior mean deterministically.
        # For OLA/FOLA this is the first sanity metric: theta = mu.
        mean_metrics, mean_calibration_rows = obs.evaluate_payload(
            cfg=self.cfg,
            payload=payload,
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            dataloader=self.testloader,
            device=self.device,
            mc_samples=1,
            posterior_sample_scale=0.0,
            eval_scope="global_test",
            run_id=self.run_id,
            round_idx=int(server_round),
        )

        # Separately evaluate Bayesian MC prediction when a posterior exists.
        # posterior_sample_scale is critical for OLA/FOLA because raw diagonal
        # Laplace sigma can be far too large for direct sampling.
        if int(self.cfg.eval_mc_samples) > 1 and self.cfg.method in {"vi", "ola"}:
            mc_metrics, mc_calibration_rows = obs.evaluate_payload(
                cfg=self.cfg,
                payload=payload,
                input_shape=self.input_shape,
                num_classes=self.num_classes,
                dataloader=self.testloader,
                device=self.device,
                mc_samples=int(self.cfg.eval_mc_samples),
                posterior_sample_scale=float(self.cfg.posterior_sample_scale),
                eval_scope="global_test_mc",
                run_id=self.run_id,
                round_idx=int(server_round),
            )
        else:
            mc_metrics, mc_calibration_rows = mean_metrics, []

        # Existing global_* fields are now stable comparison metrics based on
        # posterior mean evaluation. MC metrics are stored under global_mc_*.
        global_metrics = mean_metrics
        self.calibration_rows.extend(mean_calibration_rows)
        self.calibration_rows.extend(mc_calibration_rows)
        posterior_metrics = obs.posterior_global_metrics(self.cfg, payload)
        do_heavy = server_round == int(self.cfg.num_rounds) or server_round == 0 or server_round % int(self.cfg.heavy_eval_every) == 0
        posterior_snapshot_path = ""
        if do_heavy and self.cfg.metrics_level in {"bayes", "full"}:
            self.posterior_summary_rows.extend(obs.posterior_summary_rows(self.cfg, self.run_id, int(server_round), payload, self.param_meta))
            self.snr_histogram_rows.extend(obs.snr_histogram_rows(self.cfg, self.run_id, int(server_round), payload, self.param_meta, int(self.cfg.snr_hist_bins)))
        if self._should_save_posterior(server_round):
            posterior_snapshot_path = str(self.save_posterior_snapshot(server_round, payload))

        row: dict[str, Any] = obs.base_round_row(self.cfg, self.run_id, int(server_round))
        row.update(self.last_fit_metrics)
        row.update(
            {
                "selected_count": int(self.last_selection.selected_count),
                "round_time_sec": float(time.perf_counter() - self.round_start_time.get(int(server_round), eval_start)),
                "eval_time_sec": float(time.perf_counter() - eval_start),
                "global_accuracy": float(mean_metrics["accuracy"]),
                "global_error_rate": float(mean_metrics["error_rate"]),
                "global_loss": float(mean_metrics["loss"]),
                "global_nll": float(mean_metrics["nll"]),
                "global_brier": float(mean_metrics["brier"]),
                "global_ece": float(mean_metrics["ece"]),
                "global_mce": float(mean_metrics["mce"]),
                "global_mean_confidence": float(mean_metrics["mean_confidence"]),
                "global_mean_entropy": float(mean_metrics["mean_entropy"]),
                "global_num_eval_examples": int(mean_metrics["num_eval_examples"]),
                # Explicit posterior-mean metric names
                "global_mean_accuracy": float(mean_metrics["accuracy"]),
                "global_mean_loss": float(mean_metrics["loss"]),
                "global_mean_nll": float(mean_metrics["nll"]),
                "global_mean_brier": float(mean_metrics["brier"]),
                "global_mean_ece": float(mean_metrics["ece"]),
                "global_mean_mce": float(mean_metrics["mce"]),
                "global_mean_prediction_confidence": float(mean_metrics["mean_confidence"]),
                "global_mean_prediction_entropy": float(mean_metrics["mean_entropy"]),
                # Explicit posterior-MC metric names
                "global_mc_accuracy": float(mc_metrics["accuracy"]),
                "global_mc_loss": float(mc_metrics["loss"]),
                "global_mc_nll": float(mc_metrics["nll"]),
                "global_mc_brier": float(mc_metrics["brier"]),
                "global_mc_ece": float(mc_metrics["ece"]),
                "global_mc_mce": float(mc_metrics["mce"]),
                "global_mc_mean_confidence": float(mc_metrics["mean_confidence"]),
                "global_mc_mean_entropy": float(mc_metrics["mean_entropy"]),
                "global_mc_posterior_sample_scale": float(mc_metrics.get("posterior_sample_scale", obs.nan())),
                "global_mc_samples": int(mc_metrics["mc_samples"]),
                "global_predictive_entropy": float(mc_metrics["predictive_entropy"]),
                "global_expected_entropy": float(mc_metrics["expected_entropy"]),
                "global_mutual_information": float(mc_metrics["mutual_information"]),
                "global_aleatoric_uncertainty": float(mc_metrics["aleatoric_uncertainty"]),
                "global_epistemic_uncertainty": float(mc_metrics["epistemic_uncertainty"]),
                "global_predictive_variance_mean": float(mc_metrics["predictive_variance_mean"]),
                "global_predictive_variance_std": float(mc_metrics["predictive_variance_std"]),
                # Backward-compatible aliases
                "accuracy": float(mean_metrics["accuracy"]),
                "loss": float(mean_metrics["loss"]),
                "train_loss": float(self.last_fit_metrics.get("train_loss", 0.0)),
                "posterior_snapshot_path": posterior_snapshot_path,
                "snr_histogram_path": "snr_histograms.csv" if self.snr_histogram_rows else "",
                "calibration_bins_path": "calibration_bins.csv",
            }
        )
        row.update(posterior_metrics)
        # Selection summary for selected devices in this round.
        if self.selection_summary_rows:
            latest_sel = self.selection_summary_rows[-1]
            for key in ["selected_label_entropy_mean", "selected_label_entropy_std", "selected_kl_to_global_label_mean"]:
                row[key] = latest_sel.get(key, obs.nan())
            counts = [self.client_sizes[did] for did in self.last_selection.selected_ids]
            if counts:
                arr = np.asarray(counts, dtype=np.float64)
                row.update(
                    {
                        "selected_num_examples_mean": float(arr.mean()),
                        "selected_num_examples_std": float(arr.std()),
                        "selected_num_examples_min": float(arr.min()),
                        "selected_num_examples_max": float(arr.max()),
                    }
                )
        self.history_rows.append(row)
        print(
            f"[round={server_round:04d} method={self.cfg.method}] "
            f"mean_acc={row['global_mean_accuracy']:.4f} mean_loss={row['global_mean_loss']:.4f} "
            f"mc_acc={row['global_mc_accuracy']:.4f} mc_loss={row['global_mc_loss']:.4f} ece={row['global_ece']:.4f} "
            f"active_physical={row.get('active_physical_devices', 0)} virtual={row.get('active_virtual_clients', 0)}"
        )
        return float(mean_metrics["loss"]), {
            "accuracy": float(mean_metrics["accuracy"]),
            "global_ece": float(mean_metrics["ece"]),
            "selected_count": int(self.last_selection.selected_count),
            "active_physical_devices": int(self.last_fit_metrics.get("active_physical_devices", 0)),
        }

    def _should_save_posterior(self, server_round: int) -> bool:
        if int(server_round) == int(self.cfg.num_rounds):
            return True
        return int(self.cfg.save_posterior_every) > 0 and int(server_round) > 0 and int(server_round) % int(self.cfg.save_posterior_every) == 0

    def save_posterior_snapshot(self, server_round: int, payload: Sequence[np.ndarray]) -> Path:
        out_dir = self.output_dir / "posterior_snapshots"
        out_dir.mkdir(parents=True, exist_ok=True)
        mu, sigma, precision = obs.posterior_arrays(self.cfg, payload)
        eval_model = model.build_model(self.cfg, self.input_shape, self.num_classes).to(self.device)
        model.set_flat_parameters(eval_model, mu, self.device)
        path = out_dir / ("final.pt" if int(server_round) == int(self.cfg.num_rounds) else f"round_{int(server_round):04d}.pt")
        torch.save(
            {
                "schema_version": obs.SCHEMA_VERSION,
                "run_id": self.run_id,
                "round": int(server_round),
                "method": self.cfg.method,
                "dataset": self.cfg.dataset,
                "model": self.cfg.model,
                "param_names": [str(x["name"]) for x in self.param_meta],
                "param_shapes": [tuple(x["shape"]) for x in self.param_meta],
                "flat_slices": [(int(x["start"]), int(x["end"])) for x in self.param_meta],
                "global": {
                    "mu_flat": torch.as_tensor(mu, dtype=torch.float32),
                    "sigma_flat": None if sigma is None else torch.as_tensor(sigma, dtype=torch.float32),
                    "precision_flat": None if precision is None else torch.as_tensor(precision, dtype=torch.float32),
                    "state_dict": eval_model.state_dict(),
                },
                "payload": [np.asarray(x, dtype=np.float32) for x in payload],
                "summary": self.history_rows[-1] if self.history_rows else {},
            },
            path,
        )
        return path

    # ------------------------------------------------------------------
    # Output files
    # ------------------------------------------------------------------
    def save_history_csv(self) -> Path:
        return obs.write_csv(self.output_dir / "metrics.csv", self.history_rows, obs.METRICS_FIELDS)

    def save_all_metrics(self, final_model_path: str | Path = "") -> dict[str, Path]:
        paths: dict[str, Path] = {}
        paths["metrics"] = self.save_history_csv()
        paths["selection_summary"] = obs.write_csv(self.output_dir / "selection_summary.csv", self.selection_summary_rows, obs.SELECTION_SUMMARY_FIELDS)
        paths["selected_clients"] = obs.write_csv(self.output_dir / "selected_clients.csv", self.selected_client_rows, obs.SELECTED_CLIENT_FIELDS)
        paths["communication_metrics"] = obs.write_csv(self.output_dir / "communication_metrics.csv", self.communication_rows, obs.COMMUNICATION_FIELDS)
        paths["client_train_metrics"] = obs.write_csv(self.output_dir / "client_train_metrics.csv", self.client_train_rows, obs.CLIENT_TRAIN_FIELDS)
        paths["client_eval_metrics"] = obs.write_csv(self.output_dir / "client_eval_metrics.csv", self.client_eval_rows, obs.CLIENT_EVAL_FIELDS)
        paths["calibration_bins"] = obs.write_csv(self.output_dir / "calibration_bins.csv", self.calibration_rows, obs.CALIBRATION_BIN_FIELDS)
        paths["posterior_summary"] = obs.write_csv(self.output_dir / "posterior_summary.csv", self.posterior_summary_rows, obs.POSTERIOR_SUMMARY_FIELDS)
        paths["snr_histograms"] = obs.write_csv(self.output_dir / "snr_histograms.csv", self.snr_histogram_rows, obs.SNR_HISTOGRAM_FIELDS)
        paths["aggregation_diagnostics"] = obs.write_csv(self.output_dir / "aggregation_diagnostics.csv", self.aggregation_rows, obs.AGGREGATION_FIELDS)
        paths["run_summary"] = obs.write_csv(self.output_dir / "run_summary.csv", [self._build_run_summary(final_model_path)], obs.RUN_SUMMARY_FIELDS)
        # Empty optional post-hoc file header so downstream scripts can rely on it.
        paths["pruning_eval"] = obs.write_csv(self.output_dir / "pruning_eval.csv", [], obs.PRUNING_EVAL_FIELDS)
        return paths

    # Backward-compatible name used by old main.py versions.
    def save_selection_csv(self) -> Path:
        return obs.write_csv(self.output_dir / "selected_clients.csv", self.selected_client_rows, obs.SELECTED_CLIENT_FIELDS)

    def _build_run_summary(self, final_model_path: str | Path = "") -> dict[str, Any]:
        row: dict[str, Any] = {
            "schema_version": obs.SCHEMA_VERSION,
            "run_id": self.run_id,
            "method": self.cfg.method,
            "dataset": self.cfg.dataset,
            "model": self.cfg.model,
            "iid": bool(self.cfg.iid),
            "balanced": bool(self.cfg.balanced),
            "noniid_alpha": float(self.cfg.noniid_alpha),
            "unbalanced_alpha": float(self.cfg.unbalanced_alpha),
            "num_devices": int(self.cfg.num_devices),
            "num_virtual_clients": int(self.cfg.num_virtual_clients),
            "client_fraction": float(self.cfg.client_fraction),
            "num_rounds": int(self.cfg.num_rounds),
            "local_epochs": int(self.cfg.local_epochs),
            "batch_size": int(self.cfg.batch_size),
            "lr": float(self.cfg.lr),
            "seed": int(self.cfg.seed),
            "total_time_sec": float(time.perf_counter() - self.start_time),
            "mean_round_time_sec": obs.safe_mean([float(r.get("round_time_sec", obs.nan())) for r in self.history_rows]),
            "final_model_path": str(final_model_path),
        }
        if self.history_rows:
            final = self.history_rows[-1]
            row.update(
                {
                    "final_global_accuracy": final.get("global_accuracy", obs.nan()),
                    "final_global_loss": final.get("global_loss", obs.nan()),
                    "final_global_nll": final.get("global_nll", obs.nan()),
                    "final_global_ece": final.get("global_ece", obs.nan()),
                    "final_local_accuracy_weighted": final.get("local_accuracy_weighted", obs.nan()),
                    "final_posterior_sigma_mean": final.get("posterior_sigma_mean", obs.nan()),
                    "final_posterior_snr_raw_p50": final.get("posterior_snr_raw_p50", obs.nan()),
                    "final_posterior_snr_frac_gt_1": final.get("posterior_snr_frac_gt_1", obs.nan()),
                }
            )
            best_acc = max(self.history_rows, key=lambda x: float(x.get("global_accuracy", -1.0) or -1.0))
            row["best_global_accuracy"] = best_acc.get("global_accuracy", obs.nan())
            row["best_global_accuracy_round"] = best_acc.get("round", "")
            ece_rows = [r for r in self.history_rows if obs.is_finite_number(r.get("global_ece", obs.nan()))]
            if ece_rows:
                best_ece = min(ece_rows, key=lambda x: float(x.get("global_ece", math.inf)))
                row["best_global_ece"] = best_ece.get("global_ece", obs.nan())
                row["best_global_ece_round"] = best_ece.get("round", "")
        return row

    def save_model(self) -> Path:
        path = self.output_dir / "final_model.pt"
        mean_flat = np.asarray(self.latest_payload[0], dtype=np.float32)
        eval_model = model.build_model(self.cfg, self.input_shape, self.num_classes).to(self.device)
        model.set_flat_parameters(eval_model, mean_flat, self.device)
        checkpoint = {
            "schema_version": obs.SCHEMA_VERSION,
            "run_id": self.run_id,
            "method": self.cfg.method,
            "model": self.cfg.model,
            "dataset": self.cfg.dataset,
            "payload": [np.asarray(x, dtype=np.float32) for x in self.latest_payload],
            "state_dict": eval_model.state_dict(),
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
            "mlp_hidden": self.cfg.normalized_hidden(),
            "param_metadata": self.param_meta,
        }
        torch.save(checkpoint, path)
        return path
