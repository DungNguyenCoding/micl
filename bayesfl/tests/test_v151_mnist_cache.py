from __future__ import annotations

import dataset as dataset_module


def test_mnist_split_is_loaded_once_per_process(monkeypatch, tmp_path):
    calls = []

    class FakeMNIST:
        def __init__(self, *, root, train, download, transform):
            calls.append((root, bool(train), bool(download), transform))
            self.root = root
            self.train = bool(train)

    dataset_module.clear_dataset_cache()
    monkeypatch.setattr(dataset_module.datasets, "MNIST", FakeMNIST)

    key = dataset_module._root_cache_key(tmp_path)
    first = dataset_module._cached_mnist(key, True)
    second = dataset_module._cached_mnist(key, True)
    test_split = dataset_module._cached_mnist(key, False)

    assert first is second
    assert first is not test_split
    assert len(calls) == 2

    dataset_module.clear_dataset_cache()
