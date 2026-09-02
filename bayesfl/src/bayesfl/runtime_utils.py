"""Small runtime helpers shared by server and Ray clients."""

from __future__ import annotations

import gc
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(preference: str = "auto") -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference.startswith("cuda") and torch.cuda.is_available():
        return torch.device(preference)
    if preference == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
