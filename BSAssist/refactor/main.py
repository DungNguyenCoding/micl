"""Run grouped Flower simulation for BS-assisted OTA-FL on CIFAR-10.

The structure follows your fedavg_mnist baseline:
    - Hydra config in docs/conf/config.yaml
    - client.gen_client_fn(...) creates Context-based Flower clients
    - a custom Flower strategy handles server-side/BS logic
    - fl.simulation.start_simulation(...) launches the simulation

The difference from normal FedAvg is that each Flower virtual client can simulate
several physical edge devices. This keeps the target K=300 OTA-FL system while
reducing Ray scheduling overhead.
"""

from __future__ import annotations

# Must be set before importing numpy/torch in the main process. Ray workers also
# inherit these values, preventing CPU oversubscription.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import math
from pathlib import Path

import flwr as fl
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Subset

import client
import dataset
import model
import utils
from config import build_sim_config, list_int
from strategy import BsDatasetAssistedOtaStrategy


@hydra.main(config_path="docs/conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    sim_cfg = build_sim_config(cfg)
    m0_values = list_int(cfg.m0_values)
    if not m0_values:
        raise ValueError("m0_values cannot be empty")

    model.set_seed(sim_cfg.runtime_seed)
    model.configure_torch_threads(sim_cfg.torch_threads)
    device = model.resolve_device(sim_cfg.device)

    output_dir = Path(to_absolute_path(str(cfg.output_dir)))
    data_dir = to_absolute_path(str(cfg.data_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model = model.Cifar10CNN().to(device)
    d_model = model.count_trainable_params(base_model)
    if d_model != 307_498:
        raise RuntimeError(f"Expected D=307,498 trainable parameters, got {d_model}")
    initial_parameters = model.get_parameters(base_model)

    num_groups = min(max(1, int(sim_cfg.num_flower_clients)), int(sim_cfg.num_devices))
    device_groups = dataset.make_device_groups(sim_cfg.num_devices, num_groups)

    print("Configuration:\n" + OmegaConf.to_yaml(cfg))
    print(f"Using device: {device}")
    print(f"Flower version: {fl.__version__}")
    print(f"Physical devices K={sim_cfg.num_devices}; Flower virtual clients={num_groups}")
    print(f"CIFAR-10 CNN trainable parameters D={d_model:,}")
    print(
        f"N=ceil(D/F)={math.ceil(d_model / sim_cfg.num_subchannels)} "
        f"OFDM symbols per update round with F={sim_cfg.num_subchannels}"
    )

    trainset, testset = dataset.load_cifar10(data_dir)
    max_m0 = max(m0_values)
    full_bs_dataset, client_datasets = dataset.partition_cifar10_non_iid(
        trainset=trainset,
        cfg=sim_cfg,
        max_m0=max_m0,
    )
    client_distances = dataset.sample_client_distances(sim_cfg)
    client_sizes = [len(ds) for ds in client_datasets]

    split_path = output_dir / "data_split_summary.csv"
    dataset.save_split_summary(split_path, full_bs_dataset, client_datasets)
    print(f"Saved data split summary: {split_path}")

    client_fn = client.gen_client_fn(
        device_groups=device_groups,
        client_datasets=client_datasets,
        client_distances_m=client_distances,
        cfg=sim_cfg,
    )

    client_resources = {
        "num_cpus": float(cfg.client_cpus),
        "num_gpus": float(cfg.client_gpus),
    }
    print(f"Using client_resources: {client_resources}")

    history_paths: list[Path] = []
    for m0 in m0_values:
        bs_subset = Subset(full_bs_dataset.dataset, list(full_bs_dataset.indices[: int(m0)]))
        experiment_name = f"m0_{int(m0)}"
        strategy = BsDatasetAssistedOtaStrategy(
            initial_parameters_ndarrays=initial_parameters,
            bs_dataset=bs_subset,
            testset=testset,
            client_sizes=client_sizes,
            cfg=sim_cfg,
            output_dir=output_dir,
            experiment_name=experiment_name,
            total_rounds=int(cfg.num_rounds),
            num_flower_clients=num_groups,
        )

        print(
            f"\n=== Grouped Flower OTA-FL / CIFAR-10 / |M0|={m0} / "
            f"K={sim_cfg.num_devices} / C_flower={num_groups} / "
            f"D={strategy.d_model} / F={sim_cfg.num_subchannels} / N={strategy.n_symbols} ==="
        )
        print(f"Global examples: {strategy.global_examples}, BS weight w0={strategy.w0:.6f}")

        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_groups,
            config=fl.server.ServerConfig(num_rounds=int(cfg.num_rounds)),
            strategy=strategy,
            client_resources=client_resources,
        )

        csv_path = strategy.save_history_csv()
        model_path = strategy.save_model()
        history_paths.append(csv_path)
        print(f"Saved history: {csv_path}")
        print(f"Saved model: {model_path}")

    if bool(cfg.plot):
        plot_path = utils.plot_histories(history_paths, output_dir)
        if plot_path is not None:
            print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
