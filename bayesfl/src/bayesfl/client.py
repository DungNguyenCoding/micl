"""Flower NumPyClient implementation shared by all experiment modes."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import flwr as fl
import numpy as np
import torch

from bayesfl.config import ExperimentConfig
from bayesfl.data.datasets import load_client_loader
from bayesfl.models.factory import build_model
from bayesfl.posterior.diagnostics import bbb_posterior_summary, fola_posterior_summary, update_norms
from bayesfl.posterior.packing import (
    ParameterLayout,
    model_to_ndarrays,
    ndarrays_to_model,
    pack_fola,
    unpack_fola,
)
from bayesfl.runtime_utils import release_cuda_memory, resolve_device, seed_everything
from bayesfl.training.bbb import train_bbb
from bayesfl.training.deterministic import train_fedavg
from bayesfl.training.fola import train_fola


class BayesFLNumPyClient(fl.client.NumPyClient):
    def __init__(
        self,
        *,
        client_id: int,
        cfg: ExperimentConfig,
        partition_path: str | Path,
        average_client_size: float,
    ) -> None:
        self.client_id = int(client_id)
        self.cfg = cfg
        self.partition_path = str(partition_path)
        self.average_client_size = float(average_client_size)

    def get_parameters(self, config):
        model = build_model(self.cfg)
        arrays = model_to_ndarrays(model)
        if self.cfg.method == "fola":
            precisions = [np.full_like(a, self.cfg.fola.initial_precision, dtype=np.float32) for a in arrays]
            return pack_fola(arrays, precisions)
        return arrays

    def fit(self, parameters, config):
        server_round = int(config.get("server_round", 1))
        # Stable but client- and round-specific stochasticity.
        seed = self.cfg.runtime.seed + 1_000_003 * self.client_id + 10_007 * server_round
        seed_everything(seed)
        torch.set_num_threads(max(1, int(self.cfg.runtime.torch_num_threads)))
        device = resolve_device("auto")
        loader, client_size = load_client_loader(
            self.cfg,
            self.partition_path,
            self.client_id,
            shuffle_seed=seed,
        )
        model = build_model(self.cfg).to(device)
        layout = ParameterLayout.from_model(model)

        try:
            if self.cfg.method == "fedavg":
                ndarrays_to_model(model, parameters)
                metrics = train_fedavg(
                    model,
                    loader,
                    self.cfg,
                    server_round=server_round,
                    device=device,
                )
                updated = model_to_ndarrays(model)
                metrics.update(update_norms(updated, parameters))

            elif self.cfg.method == "bbb":
                layout.validate(parameters)
                ndarrays_to_model(model, parameters)
                metrics = train_bbb(
                    model,
                    loader,
                    self.cfg,
                    server_round=server_round,
                    device=device,
                    client_size=client_size,
                    average_client_size=self.average_client_size,
                    global_parameter_arrays=parameters,
                )
                updated = model_to_ndarrays(model)
                metrics.update(update_norms(updated, parameters))
                metrics.update(bbb_posterior_summary(model))

            elif self.cfg.method == "fola":
                global_means, global_precs = unpack_fola(parameters, layout)
                ndarrays_to_model(model, global_means)
                local_precs, metrics = train_fola(
                    model,
                    loader,
                    self.cfg,
                    server_round=server_round,
                    device=device,
                    global_mean_arrays=global_means,
                    global_precision_arrays=global_precs,
                    client_size=client_size,
                    average_client_size=self.average_client_size,
                )
                local_means = model_to_ndarrays(model)
                updated = pack_fola(local_means, local_precs)
                metrics.update(update_norms(local_means, global_means))
                metrics.update(
                    fola_posterior_summary(
                        local_means,
                        local_precs,
                        precision_min=self.cfg.fola.precision_min,
                    )
                )
            else:  # pragma: no cover
                raise ValueError(self.cfg.method)

            metrics = {
                **metrics,
                "client_id": float(self.client_id),
                "num_examples": float(client_size),
                "server_round": float(server_round),
            }
            return updated, client_size, metrics
        finally:
            del model
            release_cuda_memory()

    def evaluate(self, parameters, config):
        # Centralized server evaluation is used for consistent ECE/uncertainty metrics.
        return 0.0, 0, {}
