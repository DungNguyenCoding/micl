from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import SimulationConfig
from metrics import evaluate_bayesian_mean, evaluate_deterministic
from serialization import ParameterLayout

ROOT = Path(__file__).resolve().parents[1]


def test_paper_reference_kkt_is_the_only_v150_power_control_mode():
    cfg = SimulationConfig()
    cfg.wireless.power_control_mode = "paper_reference_kkt"
    cfg.validate()
    cfg.wireless.power_control_mode = "unified_kkt"
    with pytest.raises(ValueError):
        cfg.validate()


def test_server_paths_use_source_specific_power_scaling_with_shared_kkt_core():
    source = (ROOT / "aggregation.py").read_text(encoding="utf-8")
    # FedAvg/FedProx + SCAFFOLD payloads use Hong-2023 reference scaling.
    assert source.count("aggregate_updates_hong2023(") >= 3
    # Both Bayesian phases retain target-2025 scaling.
    assert source.count("aggregate_updates_proposed(") >= 2

    aircomp_source = (ROOT / "aircomp.py").read_text(encoding="utf-8")
    # Both aggregation implementations call one common KKT magnitude helper.
    assert aircomp_source.count("_optimal_magnitude(") >= 3


def test_phase2_precision_loader_preserves_float64():
    source = (ROOT / "client.py").read_text(encoding="utf-8")
    segment = source.split("def _load_proposed_precision", 1)[1].split(
        "def _remove_proposed_precision", 1
    )[0]
    assert "astype(np.float64)" in segment
    assert "astype(np.float32)" not in segment


def test_posterior_mean_diagnostic_matches_deterministic_mean_evaluation():
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))
    layout = ParameterLayout(model)
    mean = np.linspace(-0.2, 0.2, layout.total_numel, dtype=np.float32)
    features = torch.tensor(
        [
            [[[0.0, 1.0], [1.0, 0.0]]],
            [[[1.0, 1.0], [0.0, 0.0]]],
            [[[0.5, 0.5], [0.5, 0.5]]],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1, 1], dtype=torch.long)
    loader = DataLoader(TensorDataset(features, targets), batch_size=3)
    device = torch.device("cpu")

    deterministic = evaluate_deterministic(model, layout, mean, loader, device)
    bayes_mean = evaluate_bayesian_mean(model, layout, mean, loader, device)
    assert bayes_mean.accuracy == deterministic.accuracy
    assert bayes_mean.nll == pytest.approx(deterministic.nll)
    assert bayes_mean.ece == pytest.approx(deterministic.ece)
