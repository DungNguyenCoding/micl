"""Custom Flower strategy for grouped physical-device Bayesian FL."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
from config import RunConfig
from selector import SelectionResult, build_selector


class GroupedBayesStrategy(FedAvg):
    """Server strategy handling FedAvg, VI posterior aggregation, and FOLA.

    The server samples physical devices, sends that selection to every Flower
    virtual client, and receives one grouped response per virtual client.
    """

    def __init__(
        self,
        cfg: RunConfig,
        initial_payload: NDArrays,
        testloader: DataLoader,
        input_shape: Sequence[int],
        num_classes: int,
        output_dir: Path,
    ) -> None:
        self.cfg = cfg
        self.testloader = testloader
        self.input_shape = tuple(int(x) for x in input_shape)
        self.num_classes = int(num_classes)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.selector = build_selector(cfg.selector, cfg.seed)
        self.device = model.resolve_device(cfg.device)
        self.latest_payload = [np.asarray(x, dtype=np.float32).copy() for x in initial_payload]
        self.history_rows: list[dict[str, float | int | str]] = []
        self.selection_rows: list[dict[str, float | int | str]] = []
        self.last_selection = SelectionResult(0, [], cfg.selector)
        self.last_fit_metrics: dict[str, float | int] = {}

        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=int(cfg.num_virtual_clients),
            min_available_clients=int(cfg.num_virtual_clients),
            min_evaluate_clients=0,
            initial_parameters=ndarrays_to_parameters(self.latest_payload),
            accept_failures=bool(cfg.accept_failures),
        )

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        del parameters
        selection = self.selector.select(
            round_idx=int(server_round),
            num_devices=int(self.cfg.num_devices),
            fraction=float(self.cfg.client_fraction),
        )
        self.last_selection = selection
        selected_set = set(selection.selected_ids)
        group_counts = self._count_selected_by_virtual_group(selected_set)
        row: dict[str, float | int | str] = {
            "round": int(server_round),
            "policy": selection.policy_name,
            "selected_count": int(selection.selected_count),
            "selected_ids": selection.as_csv_string(),
        }
        for gid, count in enumerate(group_counts):
            row[f"virtual_client_{gid}_active_devices"] = int(count)
        self.selection_rows.append(row)

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

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Metrics]:
        if failures and not bool(self.cfg.accept_failures):
            raise RuntimeError(f"Flower fit failures in round {server_round}: {failures}")
        if not results:
            return ndarrays_to_parameters(self.latest_payload), {}

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
        self.last_fit_metrics = {
            "total_examples": int(total_examples),
            "active_virtual_clients": int(active_virtual_clients),
            "active_physical_devices": int(active_physical_devices),
            "train_loss": float(avg_train_loss),
        }
        metrics: Metrics = dict(self.last_fit_metrics)
        metrics["selected_count"] = int(self.last_selection.selected_count)
        return ndarrays_to_parameters(self.latest_payload), metrics

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

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Metrics]]:
        if server_round > 0 and server_round % int(self.cfg.eval_every) != 0 and server_round != int(self.cfg.num_rounds):
            return None
        payload = parameters_to_ndarrays(parameters)
        mean_flat = np.asarray(payload[0], dtype=np.float32)
        eval_model = model.build_model(self.cfg, self.input_shape, self.num_classes).to(self.device)
        model.set_flat_parameters(eval_model, mean_flat, self.device)
        loss, accuracy = model.evaluate(eval_model, self.testloader, self.device)
        row: dict[str, float | int | str] = {
            "round": int(server_round),
            "method": self.cfg.method,
            "accuracy": float(accuracy),
            "loss": float(loss),
            "train_loss": float(self.last_fit_metrics.get("train_loss", 0.0)),
            "selected_count": int(self.last_selection.selected_count),
            "active_physical_devices": int(self.last_fit_metrics.get("active_physical_devices", 0)),
            "active_virtual_clients": int(self.last_fit_metrics.get("active_virtual_clients", 0)),
            "total_examples": int(self.last_fit_metrics.get("total_examples", 0)),
        }
        self.history_rows.append(row)
        print(
            f"[round={server_round:04d} method={self.cfg.method}] "
            f"acc={accuracy:.4f} loss={loss:.4f} "
            f"active_physical={row['active_physical_devices']} virtual={row['active_virtual_clients']}"
        )
        return float(loss), {
            "accuracy": float(accuracy),
            "selected_count": int(self.last_selection.selected_count),
            "active_physical_devices": int(self.last_fit_metrics.get("active_physical_devices", 0)),
        }

    def _count_selected_by_virtual_group(self, selected_set: set[int]) -> list[int]:
        counts = [0 for _ in range(int(self.cfg.num_virtual_clients))]
        for did in selected_set:
            gid = min(int(did * int(self.cfg.num_virtual_clients) / int(self.cfg.num_devices)), int(self.cfg.num_virtual_clients) - 1)
            counts[gid] += 1
        return counts

    def save_history_csv(self) -> Path:
        path = self.output_dir / "metrics.csv"
        fieldnames = [
            "round",
            "method",
            "accuracy",
            "loss",
            "train_loss",
            "selected_count",
            "active_physical_devices",
            "active_virtual_clients",
            "total_examples",
        ]
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.history_rows:
                writer.writerow(row)
        return path

    def save_selection_csv(self) -> Path:
        path = self.output_dir / "selected_clients.csv"
        if not self.selection_rows:
            path.write_text("round,policy,selected_count,selected_ids\n", encoding="utf-8")
            return path
        fieldnames = list(self.selection_rows[0].keys())
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.selection_rows:
                writer.writerow(row)
        return path

    def save_model(self) -> Path:
        path = self.output_dir / "final_model.pt"
        mean_flat = np.asarray(self.latest_payload[0], dtype=np.float32)
        eval_model = model.build_model(self.cfg, self.input_shape, self.num_classes).to(self.device)
        model.set_flat_parameters(eval_model, mean_flat, self.device)
        checkpoint = {
            "method": self.cfg.method,
            "model": self.cfg.model,
            "dataset": self.cfg.dataset,
            "payload": [np.asarray(x, dtype=np.float32) for x in self.latest_payload],
            "state_dict": eval_model.state_dict(),
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
            "mlp_hidden": self.cfg.normalized_hidden(),
        }
        torch.save(checkpoint, path)
        return path
