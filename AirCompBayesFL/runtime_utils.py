"""Runtime, CUDA, and Ray helpers shared by clients and the server.

The helpers in this module deliberately avoid initializing CUDA unless the
configuration explicitly requests a CUDA device.  This matters for native
Windows Flower/Ray simulations because Ray controls ``CUDA_VISIBLE_DEVICES``
per worker process.
"""

from __future__ import annotations

import gc
import os
import platform
from dataclasses import dataclass

import torch

from config import SimulationConfig


@dataclass(frozen=True)
class GPUStatus:
    requested: bool
    available: bool
    torch_version: str
    cuda_build: str | None
    device_name: str | None
    device_count: int


def resolve_device(requested: str) -> torch.device:
    """Resolve ``cpu``, ``cuda``, ``cuda:N``, or ``auto`` safely."""
    normalized = str(requested).strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda" or normalized.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch wheel and verify the NVIDIA driver."
            )
        device = torch.device(normalized)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {normalized}, but this process exposes only "
                f"{torch.cuda.device_count()} CUDA device(s)."
            )
        return device
    raise ValueError("Device must be one of: cpu, cuda, cuda:N, auto")


def should_pin_memory(configured: bool, device: torch.device) -> bool:
    """Return whether a DataLoader should pin host memory.

    Pinned memory is useful only for CPU-to-CUDA copies.  Never pin memory for
    a CPU evaluator.  This avoids ``cudaErrorDevicesUnavailable`` in a Ray
    ServerApp that has no GPU resource assigned.
    """
    return bool(configured and device.type == "cuda" and torch.cuda.is_available())


def release_cuda_memory(device: torch.device) -> None:
    """Best-effort cleanup after a virtual client finishes local training."""
    if device.type != "cuda":
        return
    gc.collect()
    try:
        torch.cuda.synchronize(device)
    except Exception:
        # Do not mask the original training result/error during cleanup.
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def inspect_gpu(requested: bool = True) -> GPUStatus:
    """Return a small, printable description of the current CUDA runtime."""
    available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count()) if available else 0
    name = torch.cuda.get_device_name(0) if available and count > 0 else None
    return GPUStatus(
        requested=bool(requested),
        available=available,
        torch_version=str(torch.__version__),
        cuda_build=torch.version.cuda,
        device_name=name,
        device_count=count,
    )


def configure_runtime_environment(config: SimulationConfig) -> None:
    """Set process environment knobs before Flower starts Ray workers."""
    os.environ.setdefault("RAY_DEDUP_LOGS", "1")
    # Ray 2.55 emits a warning when a zero-GPU process hides accelerators.  The
    # server stays on CPU by configuration, so keeping the user's accelerator
    # visibility unchanged is safe and prevents pin-memory probing failures.
    if platform.system() == "Windows":
        os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")


def validate_runtime(config: SimulationConfig) -> GPUStatus:
    """Fail early for contradictory GPU settings and return GPU information."""
    client_request = config.runtime.client_device.strip().lower()
    gpu_requested = client_request == "cuda" or client_request.startswith("cuda:")

    if gpu_requested and config.runtime.client_num_gpus <= 0:
        raise ValueError(
            "runtime.client_device requests CUDA, but runtime.client_num_gpus is 0. "
            "Use client_num_gpus: 1.0 for one client at a time."
        )
    if config.runtime.client_num_gpus > 0 and client_request == "cpu":
        raise ValueError(
            "runtime.client_num_gpus is positive, but runtime.client_device is cpu."
        )

    status = inspect_gpu(requested=gpu_requested)
    if gpu_requested and not status.available:
        raise RuntimeError(
            "The YAML requests CUDA clients, but CUDA is unavailable in the launcher "
            f"process. torch={status.torch_version}, CUDA build={status.cuda_build}."
        )

    server_request = config.runtime.server_device.strip().lower()
    if platform.system() == "Windows" and server_request.startswith("cuda"):
        raise ValueError(
            "Native-Windows Ray mode should keep runtime.server_device: cpu. "
            "GPU resources are assigned to ClientApps, not the ServerApp."
        )
    return status
