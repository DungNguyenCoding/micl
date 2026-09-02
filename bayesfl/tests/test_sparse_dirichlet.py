import numpy as np

from config import SimulationConfig
from dataset import _prepare_sparse_dirichlet


def test_seed0_sparse_dirichlet_acceptance_statistics():
    cfg = SimulationConfig.profile("cifar10")
    targets = np.repeat(np.arange(10, dtype=np.int64), 5000)
    payload = _prepare_sparse_dirichlet(cfg.data, 0, targets)

    assert payload["total_samples_used"] == 10046
    assert payload["mean_size"] == 100.46
    assert payload["min_size"] == 79
    assert payload["max_size"] == 127
    assert payload["mean_classes_per_client"] == 4.0
    assert payload["empty_client_backfills"] == 0
    assert payload["num_empty_clients_after_backfill"] == 0
    assert payload["class_draws_exhausted"] == 0
    assert payload["unique_samples_used"] == 10046

    all_indices = [
        i for client in payload["clients"].values() for i in client["indices"]
    ]
    assert len(all_indices) == len(set(all_indices)) == 10046
    assert all(len(client["labels"]) == 4 for client in payload["clients"].values())
