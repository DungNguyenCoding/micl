import numpy as np

from bayesfl.metrics import expected_calibration_error, predictive_metric_bundle


def test_perfect_predictions_have_low_ece():
    probs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    labels = np.array([0, 1, 0])
    out = expected_calibration_error(probs, labels, n_bins=5)
    assert out.ece == 0.0
    metrics, _ = predictive_metric_bundle(probs, labels)
    assert metrics["accuracy"] == 1.0
    assert metrics["brier"] == 0.0
