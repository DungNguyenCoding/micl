"""Run grouped Flower simulations for FedAvg, VI Bayesian FL, or OLA/FOLA."""

from __future__ import annotations

# Keep CPU libraries from oversubscribing before importing torch-heavy modules.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import csv
import logging
from pathlib import Path

import flwr as fl
import numpy as np
from flwr.common import NDArrays

import client
import dataset
import model
from config import RunConfig, parse_args
from strategy import GroupedBayesStrategy


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, mode="w", encoding="utf-8")],
    )


def save_config_csv(cfg: RunConfig, output_dir: Path) -> Path:
    path = output_dir / "config.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for key, value in cfg.as_rows():
            writer.writerow([key, value])
    return path


def initial_payload(cfg: RunConfig, input_shape: tuple[int, int, int], num_classes: int) -> NDArrays:
    base_model = model.build_model(cfg, input_shape, num_classes)
    flat = model.flatten_parameters(base_model)
    if cfg.method == "fedavg":
        return [flat]
    if cfg.method == "ola":
        precision = np.full_like(flat, fill_value=float(cfg.precision_init), dtype=np.float32)
        return [flat, precision]
    if cfg.method == "vi":
        scale = np.full_like(flat, fill_value=float(cfg.vi_prior_scale), dtype=np.float32)
        return [flat, scale]
    raise ValueError(f"Unknown method: {cfg.method}")


def main() -> None:
    cfg = parse_args()
    model.configure_torch_threads(cfg.torch_threads)
    model.set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    logging.info("Starting Bayesian FL source run")
    logging.info("Flower version: %s", getattr(fl, "__version__", "unknown"))
    logging.info("Method=%s dataset=%s model=%s", cfg.method, cfg.dataset, cfg.model)
    logging.info("Physical devices=%d Flower virtual clients=%d client_fraction=%.4f", cfg.num_devices, cfg.num_virtual_clients, cfg.client_fraction)
    logging.info("Output directory: %s", output_dir.resolve())
    save_config_csv(cfg, output_dir)

    bundle = dataset.load_federated_data(cfg)
    dataset.save_device_summary(output_dir / "device_summary.csv", bundle, cfg)
    dataset.save_client_data_summary(output_dir / "client_data_summary.csv", bundle, cfg)
    logging.info("Loaded dataset with input_shape=%s num_classes=%d", bundle.input_shape, bundle.num_classes)
    logging.info("Saved device/data summary to %s", output_dir / "device_summary.csv")

    payload = initial_payload(cfg, bundle.input_shape, bundle.num_classes)
    logging.info("Initial payload arrays: %s", [arr.shape for arr in payload])

    client_fn = client.gen_client_fn(
        device_groups=bundle.device_groups,
        trainsets=bundle.trainsets,
        valsets=bundle.valsets,
        cfg=cfg,
        input_shape=bundle.input_shape,
        num_classes=bundle.num_classes,
        initial_payload=payload,
    )
    strategy = GroupedBayesStrategy(
        cfg=cfg,
        initial_payload=payload,
        testloader=bundle.testloader,
        input_shape=bundle.input_shape,
        num_classes=bundle.num_classes,
        output_dir=output_dir,
        client_sizes=bundle.client_sizes,
        label_counts=bundle.label_counts,
        device_positions=bundle.device_positions,
        device_groups=bundle.device_groups,
    )

    client_resources = {"num_cpus": float(cfg.client_cpus), "num_gpus": float(cfg.client_gpus)}
    logging.info("Flower client_resources=%s", client_resources)

    if cfg.dry_run:
        logging.info("Dry run requested; skipping fl.simulation.start_simulation")
    else:
        try:
            fl.simulation.start_simulation(
                client_fn=client_fn,
                num_clients=int(cfg.num_virtual_clients),
                config=fl.server.ServerConfig(num_rounds=int(cfg.num_rounds)),
                strategy=strategy,
                client_resources=client_resources,
            )
        except AttributeError as exc:
            raise RuntimeError(
                "This scaffold uses Flower's legacy start_simulation API, matching your existing repos. "
                "Install the pinned requirements in requirements.txt or port main.py to Flower's newer ClientApp/ServerApp runtime."
            ) from exc

    model_path = strategy.save_model()
    metric_paths = strategy.save_all_metrics(final_model_path=model_path)
    for name, path in metric_paths.items():
        logging.info("Saved %s: %s", name, path)
    logging.info("Saved final model: %s", model_path)
    logging.info("Done")


if __name__ == "__main__":
    main()
