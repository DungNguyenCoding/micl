import json
import numpy as np

import dataset
from config import DataConfig


def test_dirichlet_partition_is_unique_deterministic_and_scarce(tmp_path, monkeypatch):
    targets = np.repeat(np.arange(10, dtype=np.int64), 5000)
    monkeypatch.setattr(dataset, "ensure_mnist", lambda root: None)
    monkeypatch.setattr(dataset, "_training_targets", lambda root: targets)

    cfg = DataConfig(
        root="unused",
        num_clients=40,
        mean_samples_per_client=50,
        min_samples_per_client=1,
        bs_radius_m=200.0,
        partition_mode="dirichlet",
        dirichlet_alpha=0.1,
    )

    p1 = dataset.prepare_partitions(cfg, 12025, tmp_path / "a")
    p2 = dataset.prepare_partitions(cfg, 12025, tmp_path / "b")
    a = json.loads(p1.read_text())
    b = json.loads(p2.read_text())

    assert a == b
    assert a["partition_mode"] == "dirichlet"
    assert a["dirichlet_alpha"] == 0.1

    all_indices = [
        int(index)
        for client in a["clients"].values()
        for index in client["indices"]
    ]
    assert len(all_indices) == len(set(all_indices))
    assert a["total_selected_examples"] == len(all_indices)
    assert a["unique_selected_examples"] == len(all_indices)
    assert 1500 <= len(all_indices) <= 2500

    active_classes = [len(c["labels"]) for c in a["clients"].values()]
    assert min(active_classes) >= 1
    assert float(np.mean(active_classes)) < 6.0

    for client in a["clients"].values():
        assert sum(int(v) for v in client["class_counts"].values()) == int(
            client["num_examples"]
        )
