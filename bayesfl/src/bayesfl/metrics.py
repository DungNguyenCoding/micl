"""Predictive accuracy, calibration, and uncertainty metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class ECEOutput:
    ece: float
    mce: float
    bin_edges: np.ndarray
    bin_accuracy: np.ndarray
    bin_confidence: np.ndarray
    bin_count: np.ndarray


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 15,
) -> ECEOutput:
    probs = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2:
        raise ValueError("probabilities must have shape [N, C]")
    confidence = probs.max(axis=1)
    prediction = probs.argmax(axis=1)
    correct = prediction == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_acc = np.zeros(n_bins, dtype=np.float64)
    bin_conf = np.zeros(n_bins, dtype=np.float64)
    bin_count = np.zeros(n_bins, dtype=np.int64)
    ece = 0.0
    max_gap = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (confidence >= edges[i]) & (confidence <= edges[i + 1])
        else:
            mask = (confidence >= edges[i]) & (confidence < edges[i + 1])
        count = int(mask.sum())
        bin_count[i] = count
        if count == 0:
            continue
        acc = float(correct[mask].mean())
        conf = float(confidence[mask].mean())
        gap = abs(acc - conf)
        bin_acc[i] = acc
        bin_conf[i] = conf
        ece += (count / max(1, len(labels))) * gap
        max_gap = max(max_gap, gap)
    return ECEOutput(float(ece), float(max_gap), edges, bin_acc, bin_conf, bin_count)


def multiclass_brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probs = np.asarray(probabilities, dtype=np.float64)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), np.asarray(labels, dtype=np.int64)] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def entropy(probabilities: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    probs = np.clip(np.asarray(probabilities, dtype=np.float64), eps, 1.0)
    return -np.sum(probs * np.log(probs), axis=-1)


def predictive_metric_bundle(
    mean_probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    sample_probabilities: np.ndarray | None = None,
    n_bins: int = 15,
) -> tuple[Dict[str, float], ECEOutput]:
    probs = np.asarray(mean_probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pred = probs.argmax(axis=1)
    chosen = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    nll = float(-np.log(chosen).mean())
    ece = expected_calibration_error(probs, labels, n_bins=n_bins)
    pred_entropy = float(entropy(probs).mean())
    if sample_probabilities is None:
        expected_entropy = pred_entropy
        mutual_information = 0.0
    else:
        samples = np.asarray(sample_probabilities, dtype=np.float64)
        expected_entropy = float(entropy(samples).mean(axis=0).mean())
        mutual_information = max(0.0, pred_entropy - expected_entropy)
    metrics = {
        "accuracy": float((pred == labels).mean()),
        "nll": nll,
        "brier": multiclass_brier(probs, labels),
        "ece": ece.ece,
        "mce": ece.mce,
        "mean_confidence": float(probs.max(axis=1).mean()),
        "predictive_entropy": pred_entropy,
        "expected_entropy": expected_entropy,
        "mutual_information": mutual_information,
    }
    return metrics, ece
