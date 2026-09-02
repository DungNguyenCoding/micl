import json

import numpy as np

import dataset
from config import DataConfig


def test_random_nonadjacent_two_label_partition(
    tmp_path,
    monkeypatch,
):
    # Synthetic balanced 10-class training set.
    targets = np.concatenate(
        [
            np.full(
                1000,
                label,
                dtype=np.int64,
            )
            for label in range(10)
        ]
    )

    monkeypatch.setattr(
        dataset,
        "ensure_mnist",
        lambda root: None,
    )

    monkeypatch.setattr(
        dataset,
        "_training_targets",
        lambda root: targets,
    )

    cfg = DataConfig(
        root="unused",
        num_clients=40,
        labels_per_client=2,
        label_pairing_mode="random_nonadjacent",
        mean_samples_per_client=50,
        min_samples_per_client=1,
        bs_radius_m=200.0,
    )

    path = dataset.prepare_partitions(
        cfg,
        seed=12025,
        partition_dir=tmp_path,
    )

    with path.open() as f:
        payload = json.load(f)

    assert (
        payload["label_pairing_mode"]
        == "random_nonadjacent"
    )

    assert (
        payload["labels_per_client"]
        == 2
    )

    for client_id in range(40):
        labels = payload[
            "clients"
        ][str(client_id)]["labels"]

        assert len(labels) == 2

        a, b = labels

        assert a != b
        assert abs(a - b) > 1
