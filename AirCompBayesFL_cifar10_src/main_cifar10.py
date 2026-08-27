from __future__ import annotations

"""CIFAR-10 entry point for AirCompBayesFL v1.6.2.

Use this file instead of main.py for CIFAR-10 experiments.  main.py remains
untouched and continues to reproduce the MNIST/paper experiments.
"""

import os

# Propagates CIFAR mode to Ray worker processes.
os.environ["AIRCOMP_DATASET"] = "cifar10"

from cifar10_support import (
    CIFAR10_PARAMETER_COUNT,
    cifar10_parameter_count,
    install_cifar10_overrides,
)

install_cifar10_overrides()

import main as core_main  # noqa: E402  (must be imported after overrides)


if __name__ == "__main__":
    count = cifar10_parameter_count()
    if count != CIFAR10_PARAMETER_COUNT:
        raise RuntimeError(f"Unexpected CIFAR-10 model dimension: {count}")
    print(
        "CIFAR-10 extension active: RGB 32x32, "
        f"model dimension={count:,}. Core MNIST source is unchanged."
    )
    core_main.main()
