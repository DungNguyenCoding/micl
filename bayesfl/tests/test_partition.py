import numpy as np

from bayesfl.data.partition import build_sparse_dirichlet_indices


def test_cifar_requested_realized_size_statistics():
    # Synthetic balanced CIFAR-like labels; no dataset download required.
    labels = np.repeat(np.arange(10), 5000)
    result = build_sparse_dirichlet_indices(
        labels,
        num_clients=100,
        num_classes=10,
        alpha=0.1,
        avg_samples_per_client=100,
        classes_per_client=4,
        min_samples_per_client=1,
        seed=0,
        target_total_samples=10046,
    )
    md = result.metadata
    assert md["total_samples_used"] == 10046
    assert md["mean_size"] == 100.46
    assert md["min_size"] == 79
    assert md["max_size"] == 127
    assert md["mean_classes_per_client"] == 4.0
    assert md["num_empty_clients_after_backfill"] == 0
    assert md["class_draws_exhausted"] == 0

    flat = np.concatenate(result.indices)
    assert len(flat) == len(np.unique(flat))
