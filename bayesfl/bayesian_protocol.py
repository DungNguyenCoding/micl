"""Pure helpers for the paper's server-separated rho/nu protocol.

The proposed method uses two physical client-fit phases for every logical
training round:

1. precision phase: clients optimize rho_{t,k}; the server aggregates rho and
   broadcasts rho_{t+1};
2. natural-mean phase: clients optimize nu_{t,k} using the newly broadcast
   covariance; the server aggregates nu and obtains mu_{t+1}.

Keeping the schedule and coordinate transforms in this dependency-light module
makes them easy to unit-test independently from Flower, Ray, Pyro, and CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PRECISION_PHASE = "precision"
NATURAL_MEAN_PHASE = "natural_mean"
MODEL_PHASE = "model"


@dataclass(frozen=True)
class PhaseContext:
    """Map one backend/Flower physical round to a logical FL round."""

    physical_round: int
    logical_round: int
    phase: str


def physical_round_count(method: str, logical_rounds: int) -> int:
    """Return the number of backend fit rounds required by ``method``."""
    logical_rounds = int(logical_rounds)
    if logical_rounds < 0:
        raise ValueError("logical_rounds cannot be negative")
    return 2 * logical_rounds if method.lower() == "proposed" else logical_rounds


def phase_context(method: str, physical_round: int) -> PhaseContext:
    """Resolve phase and logical-round index from a physical-round index."""
    physical_round = int(physical_round)
    if physical_round < 0:
        raise ValueError("physical_round cannot be negative")

    if method.lower() != "proposed":
        return PhaseContext(
            physical_round=physical_round,
            logical_round=physical_round,
            phase=MODEL_PHASE,
        )

    if physical_round == 0:
        return PhaseContext(0, 0, MODEL_PHASE)

    logical_round = (physical_round + 1) // 2
    phase = PRECISION_PHASE if physical_round % 2 == 1 else NATURAL_MEAN_PHASE
    return PhaseContext(physical_round, logical_round, phase)


def initialize_local_nu(
    global_mean: np.ndarray,
    local_precision: np.ndarray,
    next_global_precision: np.ndarray,
    *,
    minimum_precision: float = 1.0e-12,
) -> np.ndarray:
    """Implement Eq. (33) for diagonal covariance matrices.

    nu_{t,k} = Sigma_{t,k}^{-1} Sigma_{t+1} mu_t
             = (rho_{t,k} / rho_{t+1}) * mu_t.
    """
    mean = np.asarray(global_mean, dtype=np.float64).reshape(-1)
    local_rho = np.maximum(
        np.asarray(local_precision, dtype=np.float64).reshape(-1),
        float(minimum_precision),
    )
    global_rho = np.maximum(
        np.asarray(next_global_precision, dtype=np.float64).reshape(-1),
        float(minimum_precision),
    )
    if not (mean.size == local_rho.size == global_rho.size):
        raise ValueError("mean and precision vectors must have equal length")
    return (local_rho / global_rho * mean).astype(np.float32)


def implied_local_mean(
    local_nu: np.ndarray,
    local_precision: np.ndarray,
    next_global_precision: np.ndarray,
    *,
    minimum_precision: float = 1.0e-12,
) -> np.ndarray:
    """Map the phase-2 coordinate nu to the local posterior mean in Eq. (34).

    mu_{t,k} = Sigma_{t+1}^{-1} Sigma_{t,k} nu_{t,k}
             = (rho_{t+1} / rho_{t,k}) * nu_{t,k}.
    """
    nu = np.asarray(local_nu, dtype=np.float64).reshape(-1)
    local_rho = np.maximum(
        np.asarray(local_precision, dtype=np.float64).reshape(-1),
        float(minimum_precision),
    )
    global_rho = np.maximum(
        np.asarray(next_global_precision, dtype=np.float64).reshape(-1),
        float(minimum_precision),
    )
    if not (nu.size == local_rho.size == global_rho.size):
        raise ValueError("nu and precision vectors must have equal length")
    return (global_rho / local_rho * nu).astype(np.float32)


def ideal_conflated_precision(
    local_precisions: list[np.ndarray] | tuple[np.ndarray, ...],
    weights: np.ndarray,
) -> np.ndarray:
    """Compute Eq. (18) without communication distortion."""
    if not local_precisions:
        raise ValueError("At least one local precision is required")
    vectors = [np.asarray(value, dtype=np.float64).reshape(-1) for value in local_precisions]
    if any(value.size != vectors[0].size for value in vectors):
        raise ValueError("All local precision vectors must have equal length")
    pi = np.asarray(weights, dtype=np.float64).reshape(-1)
    if pi.size != len(vectors):
        raise ValueError("weights and local_precisions have inconsistent lengths")
    pi = pi / max(float(pi.sum()), 1.0e-30)
    return np.sum(np.stack([p * value for p, value in zip(pi, vectors)]), axis=0).astype(
        np.float32
    )


def ideal_conflated_mean_from_nu(
    local_nus: list[np.ndarray] | tuple[np.ndarray, ...],
    weights: np.ndarray,
) -> np.ndarray:
    """Compute Eq. (37) in the ideal channel: mu_{t+1}=sum_k pi_k nu_{t,k}."""
    if not local_nus:
        raise ValueError("At least one local nu vector is required")
    vectors = [np.asarray(value, dtype=np.float64).reshape(-1) for value in local_nus]
    if any(value.size != vectors[0].size for value in vectors):
        raise ValueError("All local nu vectors must have equal length")
    pi = np.asarray(weights, dtype=np.float64).reshape(-1)
    if pi.size != len(vectors):
        raise ValueError("weights and local_nus have inconsistent lengths")
    pi = pi / max(float(pi.sum()), 1.0e-30)
    return np.sum(np.stack([p * value for p, value in zip(pi, vectors)]), axis=0).astype(
        np.float32
    )
