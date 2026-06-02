"""Grouped Flower clients for CIFAR-10 BS-assisted OTA-FL."""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.client.client import Client
from flwr.common import Context
from flwr.common.typing import NDArrays, Scalar
from torch.utils.data import DataLoader, Dataset

import model
from config import SimConfig
from ota import simulate_device_ota_contribution


class GroupedOtaClient(fl.client.NumPyClient):
    """A Flower virtual client that simulates a group of edge devices.

    This keeps Flower in the loop, but avoids launching 300 separate Ray tasks per
    round. If K=300 and num_flower_clients=48, each Flower client trains/simulates
    about 6 or 7 physical devices and returns one aggregate OTA signal.
    """

    def __init__(
        self,
        group_id: int,
        device_ids: Sequence[int],
        client_datasets: Sequence[Dataset],
        client_distances_m: np.ndarray,
        cfg: SimConfig,
    ) -> None:
        model.configure_torch_threads(cfg.torch_threads)
        self.group_id = int(group_id)
        self.device_ids = [int(x) for x in device_ids]
        self.client_datasets = client_datasets
        self.client_distances_m = np.asarray(client_distances_m, dtype=np.float64)
        self.cfg = cfg
        self.device = model.resolve_device(cfg.device)
        self.net = model.Cifar10CNN().to(self.device)
        self.trainloaders: Dict[int, DataLoader] = {
            did: DataLoader(
                client_datasets[did],
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=(self.device.type == "cuda"),
            )
            for did in self.device_ids
        }

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return model.get_parameters(self.net)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train all devices assigned to this worker and return group OTA sum."""
        server_round = int(config["server_round"])
        rho_ref = float(config["rho_ref"])
        global_examples = int(config["global_examples"])
        d_model = int(config["d_model"])
        padded_size = int(config["padded_size"])

        before_flat = model.flatten_parameters(parameters)
        if before_flat.size != d_model:
            raise ValueError(f"Expected D={d_model}, got {before_flat.size}")

        rx_group = np.zeros(padded_size, dtype=np.float64)
        ideal_group = np.zeros(d_model, dtype=np.float64) if self.cfg.track_distortion else None
        group_examples = 0

        for did in self.device_ids:
            trainset = self.client_datasets[did]
            trainloader = self.trainloaders[did]
            group_examples += len(trainset)

            # Each physical device starts from the same theta'_t.
            model.set_parameters(self.net, parameters, self.device)
            train_seed = self.cfg.seed + 2_000_003 * server_round + did
            model.train(self.net, trainloader, self.device, self.cfg, deterministic_seed=train_seed)
            after = model.get_parameters(self.net)
            delta_flat = (model.flatten_parameters(after) - before_flat).astype(np.float32, copy=False)

            weight = len(trainset) / max(global_examples, 1)
            round_seed = self.cfg.seed + 1_000_003 * server_round + did
            rx, ideal = simulate_device_ota_contribution(
                delta_flat=delta_flat,
                dataset_weight=weight,
                distance_m=float(self.client_distances_m[did]),
                rho_ref=rho_ref,
                cfg=self.cfg,
                round_seed=round_seed,
            )
            rx_group += rx.astype(np.float64, copy=False)
            if ideal_group is not None:
                ideal_group += ideal.astype(np.float64, copy=False)

        response: NDArrays
        if ideal_group is not None:
            response = [
                rx_group.astype(np.float32, copy=False),
                ideal_group.astype(np.float32, copy=False),
            ]
        else:
            response = [rx_group.astype(np.float32, copy=False)]

        metrics: Dict[str, Scalar] = {
            "group_id": int(self.group_id),
            "num_devices_in_group": int(len(self.device_ids)),
            "group_examples": int(group_examples),
        }
        return response, int(group_examples), metrics

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        del parameters, config
        return 0.0, 0, {"accuracy": 0.0}


def gen_client_fn(
    device_groups: Sequence[Sequence[int]],
    client_datasets: Sequence[Dataset],
    client_distances_m: np.ndarray,
    cfg: SimConfig,
) -> Callable[[Context], Client]:
    """Generate Flower's Context-based client factory, matching your FedAvg style."""
    cache: Dict[int, Client] = {}

    def client_fn(context: Context) -> Client:
        partition_id = int(context.node_config["partition-id"])
        if cfg.cache_clients and partition_id in cache:
            return cache[partition_id]

        fl_client = GroupedOtaClient(
            group_id=partition_id,
            device_ids=device_groups[partition_id],
            client_datasets=client_datasets,
            client_distances_m=client_distances_m,
            cfg=cfg,
        ).to_client()
        if cfg.cache_clients:
            cache[partition_id] = fl_client
        return fl_client

    return client_fn
