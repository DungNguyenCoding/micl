"""Accuracy, NLL, ECE, and Bayesian posterior-predictive evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from bayesian_torch_backend import BayesianTorchStateAdapter
from config import VariationalConfig
from serialization import ParameterLayout


@dataclass
class EvaluationResult:
    accuracy: float
    nll: float
    ece: float
    bin_lower: np.ndarray
    bin_upper: np.ndarray
    bin_count: np.ndarray
    bin_confidence: np.ndarray
    bin_accuracy: np.ndarray


def expected_calibration_error(
    probabilities: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> EvaluationResult:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    accuracy = float(np.mean(predictions == targets))
    true_prob = np.maximum(probabilities[np.arange(len(targets)), targets], 1.0e-12)
    nll = float(-np.mean(np.log(true_prob)))

    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    counts = np.zeros(n_bins, dtype=np.int64)
    bin_conf = np.zeros(n_bins, dtype=np.float64)
    bin_acc = np.zeros(n_bins, dtype=np.float64)
    ece = 0.0
    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        count = int(np.count_nonzero(mask))
        counts[i] = count
        if count:
            bin_conf[i] = float(np.mean(confidence[mask]))
            bin_acc[i] = float(np.mean(predictions[mask] == targets[mask]))
            ece += (count / max(1, len(targets))) * abs(bin_acc[i] - bin_conf[i])

    return EvaluationResult(
        accuracy=accuracy,
        nll=nll,
        ece=float(ece),
        bin_lower=edges[:-1],
        bin_upper=edges[1:],
        bin_count=counts,
        bin_confidence=bin_conf,
        bin_accuracy=bin_acc,
    )


def _predict_deterministic(model, loader: DataLoader, device: torch.device):
    model.eval()
    probability_batches = []
    target_batches = []
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device, non_blocking=True)
            logits = model(features)
            probability_batches.append(torch.softmax(logits, dim=1).cpu().numpy())
            target_batches.append(targets.numpy())
    return np.concatenate(probability_batches), np.concatenate(target_batches)


def evaluate_deterministic(
    model: torch.nn.Module,
    layout: ParameterLayout,
    model_vector: np.ndarray,
    loader: DataLoader,
    device: torch.device,
    n_bins: int = 10,
) -> EvaluationResult:
    model = model.to(device)
    layout.load_model_vector(model, np.asarray(model_vector, dtype=np.float32))
    probabilities, targets = _predict_deterministic(model, loader, device)
    return expected_calibration_error(probabilities, targets, n_bins=n_bins)


def evaluate_bayesian_state(
    deterministic_model: torch.nn.Module,
    layout: ParameterLayout,
    state_vector: np.ndarray,
    variational_cfg: VariationalConfig,
    loader: DataLoader,
    device: torch.device,
    seed: int,
    n_bins: int = 10,
) -> tuple[EvaluationResult, EvaluationResult, tuple[float, float, float], np.ndarray]:
    """Return predictive eval, posterior-mean eval, sigma stats, mean model vector."""
    adapter = BayesianTorchStateAdapter(
        deterministic_model, layout, variational_cfg
    ).to(device)
    adapter.load_state(state_vector)
    adapter.model.eval()

    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    probability_batches = []
    target_batches = []
    mc = int(variational_cfg.mc_eval)
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device, non_blocking=True)
            accumulated = None
            for _ in range(mc):
                logits = adapter.model(features)
                if isinstance(logits, tuple):
                    logits = logits[0]
                probs = torch.softmax(logits, dim=1)
                accumulated = probs if accumulated is None else accumulated + probs
            assert accumulated is not None
            probability_batches.append((accumulated / float(mc)).cpu().numpy())
            target_batches.append(targets.numpy())

    probabilities = np.concatenate(probability_batches)
    targets_np = np.concatenate(target_batches)
    predictive = expected_calibration_error(probabilities, targets_np, n_bins=n_bins)

    mean_vector = adapter.mean_model_vector()
    mean_eval_model = deterministic_model.to(device)
    layout.load_model_vector(mean_eval_model, mean_vector)
    mean_probs, mean_targets = _predict_deterministic(mean_eval_model, loader, device)
    posterior_mean = expected_calibration_error(mean_probs, mean_targets, n_bins=n_bins)
    return predictive, posterior_mean, adapter.sigma_stats(), mean_vector
