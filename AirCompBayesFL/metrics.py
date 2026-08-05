"""Accuracy, NLL, ECE, and reliability-diagram evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

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
    confidences = probabilities.max(axis=1)
    correct = predictions == targets

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_count = np.zeros(n_bins, dtype=np.int64)
    bin_confidence = np.zeros(n_bins, dtype=np.float64)
    bin_accuracy = np.zeros(n_bins, dtype=np.float64)
    ece = 0.0

    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        if index == 0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences > lower) & (confidences <= upper)
        count = int(mask.sum())
        bin_count[index] = count
        if count > 0:
            bin_confidence[index] = float(confidences[mask].mean())
            bin_accuracy[index] = float(correct[mask].mean())
            ece += (count / len(targets)) * abs(
                bin_accuracy[index] - bin_confidence[index]
            )

    clipped = np.clip(probabilities[np.arange(len(targets)), targets], 1.0e-12, 1.0)
    nll = float(-np.log(clipped).mean())
    accuracy = float(correct.mean())
    return EvaluationResult(
        accuracy=accuracy,
        nll=nll,
        ece=float(ece),
        bin_lower=edges[:-1],
        bin_upper=edges[1:],
        bin_count=bin_count,
        bin_confidence=bin_confidence,
        bin_accuracy=bin_accuracy,
    )


def _predict_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probability_batches: List[np.ndarray] = []
    target_batches: List[np.ndarray] = []
    non_blocking = bool(device.type == "cuda" and loader.pin_memory)
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device, non_blocking=non_blocking)
            logits = model(features)
            probabilities = torch.softmax(logits, dim=1)
            probability_batches.append(probabilities.cpu().numpy())
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
    layout.load_model_vector(model, model_vector)
    probabilities, targets = _predict_model(model, loader, device)
    return expected_calibration_error(probabilities, targets, n_bins=n_bins)


def evaluate_bayesian(
    model: torch.nn.Module,
    layout: ParameterLayout,
    mean: np.ndarray,
    precision: np.ndarray,
    loader: DataLoader,
    device: torch.device,
    mc_samples: int,
    seed: int,
    n_bins: int = 10,
) -> EvaluationResult:
    model = model.to(device)
    mean = np.asarray(mean, dtype=np.float64)
    precision = np.maximum(np.asarray(precision, dtype=np.float64), 1.0e-12)
    std = np.sqrt(1.0 / precision)
    rng = np.random.default_rng(seed)

    accumulated: np.ndarray | None = None
    targets_reference: np.ndarray | None = None
    for _ in range(int(mc_samples)):
        sample = rng.normal(mean, std).astype(np.float32)
        layout.load_model_vector(model, sample)
        probabilities, targets = _predict_model(model, loader, device)
        if accumulated is None:
            accumulated = np.zeros_like(probabilities, dtype=np.float64)
            targets_reference = targets
        accumulated += probabilities

    assert accumulated is not None and targets_reference is not None
    posterior_predictive = accumulated / float(mc_samples)
    return expected_calibration_error(
        posterior_predictive, targets_reference, n_bins=n_bins
    )
