"""Standalone CUDA and execution-backend preflight check."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import SimulationConfig
from runtime_utils import (
    configure_runtime_environment,
    resolve_backend,
    validate_runtime,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke_gpu.yaml")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    path = Path(args.config)
    if not path.is_absolute():
        path = project_root / path

    config = SimulationConfig.from_yaml(path)
    configure_runtime_environment(config)
    status = validate_runtime(config)
    backend = resolve_backend(config)

    print(f"Configuration: {path}")
    print(f"Resolved backend: {backend}")
    print(f"Torch: {status.torch_version}")
    print(f"CUDA build: {status.cuda_build}")
    print(f"CUDA available: {status.available}")
    print(f"Visible CUDA devices: {status.device_count}")
    print(f"GPU: {status.device_name}")
    print(f"Client device: {config.runtime.client_device}")
    print(f"Ray GPUs per client: {config.runtime.client_num_gpus}")
    print(f"Server device: {config.runtime.server_device}")
    print(f"Configured pin memory: {config.data.pin_memory}")
    if backend == "local":
        print("Client concurrency: sequential (stable native-Windows CUDA mode)")
    else:
        print("Client concurrency: Flower/Ray resource scheduling")
    print("Preflight passed.")


if __name__ == "__main__":
    main()
