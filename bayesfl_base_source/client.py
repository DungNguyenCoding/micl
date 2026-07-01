"""Grouped Flower clients for physical-device simulation."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Dict, List, Sequence, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.client.client import Client
from flwr.common import Context
from flwr.common.typing import NDArrays, Scalar
from torch.utils.data import DataLoader, Dataset

import bayes_vi
import model
from config import RunConfig
from selector import parse_selected_ids


class GroupedBayesClient(fl.client.NumPyClient):
    """A Flower virtual client that sequentially simulates many physical devices."""

    def __init__(
        self,
        group_id: int,
        device_ids: Sequence[int],
        trainsets: Sequence[Dataset],
        cfg: RunConfig,
        input_shape: Sequence[int],
        num_classes: int,
        initial_payload: NDArrays,
    ) -> None:
        model.configure_torch_threads(cfg.torch_threads)
        self.group_id = int(group_id)
        self.device_ids = [int(x) for x in device_ids]
        self.trainsets = trainsets
        self.cfg = cfg
        self.input_shape = tuple(int(x) for x in input_shape)
        self.num_classes = int(num_classes)
        self.initial_payload = [np.asarray(x, dtype=np.float32) for x in initial_payload]
        self.device = model.resolve_device(cfg.device)
        self._loaders: Dict[int, DataLoader] = {}

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return [arr.copy() for arr in self.initial_payload]

    def _loader_for(self, device_id: int) -> DataLoader:
        if device_id not in self._loaders:
            self._loaders[device_id] = DataLoader(
                self.trainsets[device_id],
                batch_size=int(self.cfg.batch_size),
                shuffle=True,
                num_workers=int(self.cfg.num_workers),
                pin_memory=(self.device.type == "cuda"),
            )
        return self._loaders[device_id]

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        server_round = int(config.get("server_round", 0))
        selected_ids = parse_selected_ids(str(config.get("selected_ids", "")))
        active_ids = [did for did in self.device_ids if did in selected_ids]

        if self.cfg.method == "fedavg":
            payload, examples, avg_loss = self._fit_fedavg(parameters, active_ids, server_round)
        elif self.cfg.method == "ola":
            payload, examples, avg_loss = self._fit_ola(parameters, active_ids, server_round)
        elif self.cfg.method == "vi":
            payload, examples, avg_loss = self._fit_vi(parameters, active_ids, server_round)
        else:
            raise ValueError(f"Unsupported method: {self.cfg.method}")

        metrics: Dict[str, Scalar] = {
            "group_id": int(self.group_id),
            "active_devices": int(len(active_ids)),
            "train_loss": float(avg_loss),
            "group_examples": int(examples),
        }
        return payload, int(examples), metrics

    def _fit_fedavg(self, parameters: NDArrays, active_ids: Sequence[int], server_round: int) -> tuple[NDArrays, int, float]:
        global_mu = np.asarray(parameters[0], dtype=np.float32)
        if not active_ids:
            return [np.zeros_like(global_mu)], 0, 0.0

        weighted = np.zeros_like(global_mu, dtype=np.float64)
        total_examples = 0
        loss_sum = 0.0
        for did in active_ids:
            local_model = model.build_model(self.cfg, self.input_shape, self.num_classes).to(self.device)
            model.set_flat_parameters(local_model, global_mu, self.device)
            model.set_seed(int(self.cfg.seed + 1_000_003 * server_round + did))
            loader = self._loader_for(did)
            loss, _ = model.train_deterministic(local_model, loader, self.device, self.cfg)
            flat = model.flatten_parameters(local_model)
            n = len(self.trainsets[did])
            weighted += float(n) * flat.astype(np.float64, copy=False)
            total_examples += int(n)
            loss_sum += float(loss) * int(n)
        return [(weighted / max(total_examples, 1)).astype(np.float32)], total_examples, loss_sum / max(total_examples, 1)

    def _fit_ola(self, parameters: NDArrays, active_ids: Sequence[int], server_round: int) -> tuple[NDArrays, int, float]:
        global_mu = np.asarray(parameters[0], dtype=np.float32)
        global_precision = np.maximum(np.asarray(parameters[1], dtype=np.float32), float(self.cfg.precision_floor))
        if not active_ids:
            return [np.zeros_like(global_precision), np.zeros_like(global_precision)], 0, 0.0

        precision_count_sum = np.zeros_like(global_precision, dtype=np.float64)
        precision_mu_count_sum = np.zeros_like(global_precision, dtype=np.float64)
        total_examples = 0
        loss_sum = 0.0
        round_idx = max(int(server_round), 1)
        for did in active_ids:
            local_model = model.build_model(self.cfg, self.input_shape, self.num_classes).to(self.device)
            model.set_flat_parameters(local_model, global_mu, self.device)
            model.set_seed(int(self.cfg.seed + 2_000_003 * server_round + did))
            loader = self._loader_for(did)
            loss, fisher = model.train_deterministic(
                local_model,
                loader,
                self.device,
                self.cfg,
                prior_mu=global_mu,
                prior_precision=global_precision,
                prior_lambda=float(self.cfg.ola_prior_lambda),
                collect_fisher=True,
            )
            if fisher is None:
                fisher = np.zeros_like(global_mu)
            local_mu = model.flatten_parameters(local_model)
            # FOLA-style online precision. The constant initial precision plays
            # the role of the gamma-I term in the global online Laplace estimate,
            # so it is injected with weight 1/r while the previous global
            # precision carries the accumulated history.
            gamma = np.full_like(global_precision, fill_value=float(self.cfg.precision_init), dtype=np.float32)
            local_precision = (1.0 / round_idx) * fisher + ((round_idx - 1.0) / round_idx) * global_precision + (1.0 / round_idx) * gamma
            local_precision = np.maximum(local_precision.astype(np.float32, copy=False), float(self.cfg.precision_floor))
            n = len(self.trainsets[did])
            precision_count_sum += float(n) * local_precision.astype(np.float64, copy=False)
            precision_mu_count_sum += float(n) * (local_precision * local_mu).astype(np.float64, copy=False)
            total_examples += int(n)
            loss_sum += float(loss) * int(n)
        return [precision_count_sum.astype(np.float32), precision_mu_count_sum.astype(np.float32)], total_examples, loss_sum / max(total_examples, 1)

    def _fit_vi(self, parameters: NDArrays, active_ids: Sequence[int], server_round: int) -> tuple[NDArrays, int, float]:
        global_loc = np.asarray(parameters[0], dtype=np.float32)
        global_scale = np.maximum(np.asarray(parameters[1], dtype=np.float32), float(self.cfg.vi_min_scale))
        if not active_ids:
            zeros = np.zeros_like(global_loc)
            return [zeros, zeros], 0, 0.0

        total_examples = 0
        loss_sum = 0.0
        if self.cfg.bayes_aggregation == "product":
            first_sum = np.zeros_like(global_loc, dtype=np.float64)  # n * precision
            second_sum = np.zeros_like(global_loc, dtype=np.float64)  # n * precision * loc
        else:
            first_sum = np.zeros_like(global_loc, dtype=np.float64)  # n * loc
            second_sum = np.zeros_like(global_loc, dtype=np.float64)  # n * (var + loc^2)

        for did in active_ids:
            loader = self._loader_for(did)
            seed = int(self.cfg.seed + 3_000_003 * server_round + did)
            loc, scale, loss = bayes_vi.train_vi_local(
                trainloader=loader,
                input_shape=self.input_shape,
                num_classes=self.num_classes,
                hidden_dims=self.cfg.normalized_hidden(),
                global_loc=global_loc,
                global_scale=global_scale,
                device=self.device,
                cfg=self.cfg,
                seed=seed,
            )
            n = len(self.trainsets[did])
            if self.cfg.bayes_aggregation == "product":
                precision = 1.0 / np.maximum(scale * scale, float(self.cfg.vi_min_scale) ** 2)
                first_sum += float(n) * precision.astype(np.float64, copy=False)
                second_sum += float(n) * (precision * loc).astype(np.float64, copy=False)
            else:
                var = scale * scale
                first_sum += float(n) * loc.astype(np.float64, copy=False)
                second_sum += float(n) * (var + loc * loc).astype(np.float64, copy=False)
            total_examples += int(n)
            loss_sum += float(loss) * int(n)
        return [first_sum.astype(np.float32), second_sum.astype(np.float32)], total_examples, loss_sum / max(total_examples, 1)

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        del parameters, config
        return 0.0, 0, {"accuracy": 0.0}


def _partition_id_from_context(context: Context | str | int) -> int:
    """Support both current Context-based and older string-based client_fn APIs."""
    if isinstance(context, (str, int)):
        return int(context)
    node_config = getattr(context, "node_config", {})
    if "partition-id" in node_config:
        return int(node_config["partition-id"])
    if "partition_id" in node_config:
        return int(node_config["partition_id"])
    if "cid" in node_config:
        return int(node_config["cid"])
    raise KeyError("Could not infer partition id from Flower Context")


def gen_client_fn(
    device_groups: Sequence[Sequence[int]],
    trainsets: Sequence[Dataset],
    cfg: RunConfig,
    input_shape: Sequence[int],
    num_classes: int,
    initial_payload: NDArrays,
) -> Callable[[Context], Client]:
    """Generate Flower client factory."""
    cache: Dict[int, Client] = {}

    def client_fn(context: Context) -> Client:
        partition_id = _partition_id_from_context(context)
        if bool(cfg.cache_clients) and partition_id in cache:
            return cache[partition_id]
        fl_client = GroupedBayesClient(
            group_id=partition_id,
            device_ids=device_groups[partition_id],
            trainsets=trainsets,
            cfg=cfg,
            input_shape=input_shape,
            num_classes=num_classes,
            initial_payload=initial_payload,
        ).to_client()
        if bool(cfg.cache_clients):
            cache[partition_id] = fl_client
        return fl_client

    return client_fn
