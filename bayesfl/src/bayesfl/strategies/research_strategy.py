"""Strict full-participation Flower strategies for all three methods."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from bayesfl.config import ExperimentConfig, round_learning_rate
from bayesfl.logging_utils import CsvRecorder
from bayesfl.posterior.gaussian import gaussian_product, inverse_softplus_np, softplus_np
from bayesfl.posterior.packing import ParameterLayout, pack_fola, unpack_fola
from .common import normalized_example_weights, weighted_average_arrays, weighted_metrics


class ResearchStrategy(FedAvg):
    """One strategy class with method-specific aggregation math."""

    def __init__(
        self,
        *,
        cfg: ExperimentConfig,
        layout: ParameterLayout,
        initial_arrays: Sequence[np.ndarray],
        run_dir: Path,
        logger,
    ) -> None:
        fraction_fit = cfg.federation.clients_per_round / float(cfg.federation.num_clients)
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=0.0,
            min_fit_clients=cfg.federation.clients_per_round,
            min_evaluate_clients=1,
            min_available_clients=cfg.federation.num_clients,
            accept_failures=False,
            initial_parameters=ndarrays_to_parameters(list(initial_arrays)),
            on_fit_config_fn=lambda rnd: {
                "server_round": int(rnd),
                "learning_rate": float(round_learning_rate(cfg.training, rnd)),
            },
        )
        self.cfg = cfg
        self.layout = layout
        self.logger = logger
        self.client_metrics = CsvRecorder(run_dir / "metrics" / "client_metrics.csv")
        self.round_metrics = CsvRecorder(run_dir / "metrics" / "round_train_metrics.csv")

    def aggregate_fit(self, server_round, results, failures):
        if failures:
            first = failures[0]
            raise RuntimeError(
                f"Round {server_round}: {len(failures)} client job(s) failed; first={first!r}"
            )
        if len(results) != self.cfg.federation.clients_per_round:
            raise RuntimeError(
                f"Round {server_round}: expected {self.cfg.federation.clients_per_round} results, "
                f"received {len(results)}"
            )

        client_arrays = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        counts = [int(fit_res.num_examples) for _, fit_res in results]
        weights = normalized_example_weights(counts)
        if self.cfg.method == "fedavg":
            aggregated = weighted_average_arrays(client_arrays, weights)
        elif self.cfg.method == "bbb":
            aggregated = self._aggregate_bbb(client_arrays, weights)
        elif self.cfg.method == "fola":
            aggregated = self._aggregate_fola(client_arrays, weights)
        else:  # pragma: no cover
            raise ValueError(self.cfg.method)

        # Keep one row per client so heterogeneity and local behavior remain inspectable.
        for _, res in results:
            client_row = {
                "round": server_round,
                "num_examples": int(res.num_examples),
                **dict(res.metrics),
            }
            self.client_metrics.append(client_row)

        metrics = weighted_metrics([(int(res.num_examples), dict(res.metrics)) for _, res in results])
        row = {"round": server_round, "num_clients": len(results), "num_examples": sum(counts), **metrics}
        self.round_metrics.append(row)
        self.logger.info(
            "Round %d aggregation complete: clients=%d examples=%d train_loss=%s",
            server_round,
            len(results),
            sum(counts),
            f"{metrics.get('train_loss', float('nan')):.6f}",
        )
        return ndarrays_to_parameters(aggregated), metrics

    def _aggregate_bbb(
        self,
        clients: Sequence[Sequence[np.ndarray]],
        weights: Sequence[float],
    ) -> list[np.ndarray]:
        # Optional ablation: directly FedAvg all variational parameters.
        if self.cfg.bbb.aggregation == "fedavg_variational":
            return weighted_average_arrays(clients, weights)

        result = weighted_average_arrays(clients, weights)
        name_to_idx = {name: idx for idx, name in enumerate(self.layout.names)}
        processed: set[str] = set()
        for rho_name, rho_idx in name_to_idx.items():
            if "rho_" not in rho_name:
                continue
            mu_name = rho_name.replace("rho_", "mu_", 1)
            if mu_name not in name_to_idx or mu_name in processed:
                continue
            mu_idx = name_to_idx[mu_name]
            means = [np.asarray(c[mu_idx]) for c in clients]
            variances = [np.maximum(softplus_np(np.asarray(c[rho_idx])) ** 2, 1e-12) for c in clients]
            precisions = [1.0 / var for var in variances]
            mu_global, precision_global = gaussian_product(means, precisions, weights)
            sigma_global = np.sqrt(1.0 / np.maximum(precision_global, 1e-12))
            rho_global = inverse_softplus_np(sigma_global).astype(clients[0][rho_idx].dtype, copy=False)
            result[mu_idx] = mu_global.astype(clients[0][mu_idx].dtype, copy=False)
            result[rho_idx] = rho_global
            processed.add(mu_name)
        return result

    def _aggregate_fola(
        self,
        clients: Sequence[Sequence[np.ndarray]],
        weights: Sequence[float],
    ) -> list[np.ndarray]:
        means_by_client = []
        precs_by_client = []
        for client in clients:
            means, precs = unpack_fola(client, self.layout)
            means_by_client.append(means)
            precs_by_client.append(precs)

        global_means: list[np.ndarray] = []
        global_precs: list[np.ndarray] = []
        for param_idx in range(self.layout.size):
            means = [m[param_idx] for m in means_by_client]
            precs = [p[param_idx] for p in precs_by_client]
            mu, precision = gaussian_product(
                means,
                precs,
                weights,
                precision_min=self.cfg.fola.precision_min,
                precision_max=self.cfg.fola.precision_max,
            )
            global_means.append(mu)
            global_precs.append(precision)
        return pack_fola(global_means, global_precs)
