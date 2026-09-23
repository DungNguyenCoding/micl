"""The uploaded dense FOLA reducer, extracted without changing its arithmetic."""
from __future__ import annotations
from typing import Sequence
import numpy as np
from bayesfl.config import ExperimentConfig
from bayesfl.posterior.gaussian import gaussian_product
from bayesfl.posterior.packing import ParameterLayout, pack_fola, unpack_fola


def aggregate_fola(clients: Sequence[Sequence[np.ndarray]], weights: Sequence[float],
                   *, layout: ParameterLayout, cfg: ExperimentConfig) -> list[np.ndarray]:
    means_by_client = []
    precs_by_client = []
    for client in clients:
        means, precs = unpack_fola(client, layout)
        means_by_client.append(means)
        precs_by_client.append(precs)

    global_means: list[np.ndarray] = []
    global_precs: list[np.ndarray] = []
    for param_idx in range(layout.size):
        means = [m[param_idx] for m in means_by_client]
        precs = [p[param_idx] for p in precs_by_client]

        if cfg.fola.mode == "paper_reference":
            # Released CIFAR implementation:
            #   omega_S = sum pi_n omega_n
            #   mu_S = sum pi_n omega_n mu_n / (omega_S + eps)
            denom = np.zeros_like(precs[0], dtype=np.float64)
            numer = np.zeros_like(means[0], dtype=np.float64)
            for mu_n, omega_n, weight in zip(means, precs, weights):
                omega64 = np.asarray(omega_n, dtype=np.float64)
                w = float(weight)
                denom += w * omega64
                numer += w * omega64 * np.asarray(mu_n, dtype=np.float64)
            mu = numer / (denom + float(cfg.fola.aggregation_epsilon))
            precision = denom
            global_means.append(mu.astype(means[0].dtype, copy=False))
            global_precs.append(precision.astype(precs[0].dtype, copy=False))
        else:
            mu, precision = gaussian_product(
                means,
                precs,
                weights,
                precision_min=cfg.fola.precision_min,
                precision_max=cfg.fola.precision_max,
            )
            global_means.append(mu)
            global_precs.append(precision)
    return pack_fola(global_means, global_precs)
