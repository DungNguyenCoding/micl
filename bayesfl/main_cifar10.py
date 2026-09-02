from __future__ import annotations

"""CIFAR-10 entry point with Ray-safe dataset/model overrides."""

import os

os.environ["AIRCOMP_DATASET"] = "cifar10"

from cifar10_support import install_cifar10_overrides

install_cifar10_overrides()

import main as core_main  # noqa: E402


if __name__ == "__main__":
    print(
        "CIFAR-10 extension active: RGB 32x32. "
        "Model architecture is selected by model.name in the YAML."
    )
    core_main.main()
