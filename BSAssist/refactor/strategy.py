"""Custom Flower strategy implementing BS-assisted superimposed OTA aggregation."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common import FitIns, Metrics, NDArrays, Parameters, Scalar
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from torch.utils.data import DataLoader, Dataset, Subset

import model
from config import SimConfig


class BsDatasetAssistedOtaStrategy(FedAvg):
    """Algorithm 3: BS initial update + superimposed OTA update report.

    This strategy receives one aggregate noiseless OTA signal from each grouped
    Flower client. Each grouped Flower client can represent multiple simulated
    edge devices, so the physical simulation can still use K=300 while Flower
    schedules far fewer Ray tasks per round.
    """

    def __init__(
        self,
        initial_parameters_ndarrays: NDArrays,
        bs_dataset: Dataset,
        testset: Dataset,
        client_sizes: Sequence[int],
        cfg: SimConfig,
        output_dir: Path,
        experiment_name: str,
        total_rounds: int,
        num_flower_clients: int,
    ) -> None:
        self.cfg = cfg
        self.device = model.resolve_device(cfg.device)
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.total_rounds = int(total_rounds)
        self.num_flower_clients = int(num_flower_clients)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.initial_parameters_ndarrays = model.clone_parameters(initial_parameters_ndarrays)
        self.shapes = [tuple(arr.shape) for arr in initial_parameters_ndarrays]
        self.d_model = int(sum(int(np.prod(s)) for s in self.shapes))
        self.n_symbols = int(math.ceil(self.d_model / cfg.num_subchannels))
        self.padded_size = self.n_symbols * cfg.num_subchannels

        self.bs_dataset = bs_dataset
        self.bs_loader = DataLoader(
            bs_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )
        self.bs_examples = int(len(bs_dataset))
        self.client_sizes = [int(x) for x in client_sizes]
        self.global_examples = int(self.bs_examples + sum(self.client_sizes))
        self.w0 = self.bs_examples / max(self.global_examples, 1)

        self.bs_model = model.Cifar10CNN().to(self.device)
        self.eval_model = model.Cifar10CNN().to(self.device)
        self.testloader = DataLoader(
            testset,
            batch_size=256,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

        self.theta_prime_ndarrays: Optional[NDArrays] = None
        self.rho_ref: float = 0.0
        self.last_distortion: float = float("nan")
        self.history_rows: List[Dict[str, float]] = []
        self.rng = np.random.default_rng(cfg.seed + 77)

        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=self.num_flower_clients,
            min_available_clients=self.num_flower_clients,
            min_evaluate_clients=0,
            initial_parameters=ndarrays_to_parameters(self.initial_parameters_ndarrays),
            accept_failures=False,
        )

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Perform BS local update and send theta'_t, |M|, rho_ref,t to clients."""
        current_params = parameters_to_ndarrays(parameters)
        model.set_parameters(self.bs_model, current_params, self.device)
        bs_seed = self.cfg.seed + 3_000_003 * server_round
        model.train(self.bs_model, self.bs_loader, self.device, self.cfg, deterministic_seed=bs_seed)
        bs_updated = model.get_parameters(self.bs_model)

        current_flat = model.flatten_parameters(current_params)
        bs_updated_flat = model.flatten_parameters(bs_updated)
        delta0 = bs_updated_flat - current_flat
        theta_prime_flat = current_flat + self.w0 * delta0
        self.theta_prime_ndarrays = model.unflatten_parameters(theta_prime_flat, self.shapes)
        self.rho_ref = float(np.linalg.norm(delta0.astype(np.float64)) ** 2 / max(self.d_model, 1))

        fit_config: Dict[str, Scalar] = {
            "server_round": int(server_round),
            "rho_ref": float(self.rho_ref),
            "global_examples": int(self.global_examples),
            "d_model": int(self.d_model),
            "padded_size": int(self.padded_size),
            "num_subchannels": int(self.cfg.num_subchannels),
        }
        fit_ins = FitIns(ndarrays_to_parameters(self.theta_prime_ndarrays), fit_config)
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        return [(client_proxy, fit_ins) for client_proxy in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Metrics]:
        if failures:
            raise RuntimeError(f"Flower fit failures in round {server_round}: {failures}")
        if not results:
            return None, {}
        if self.theta_prime_ndarrays is None:
            raise RuntimeError("theta_prime was not initialized before aggregate_fit")

        rx_sum = np.zeros(self.padded_size, dtype=np.float64)
        ideal_sum = np.zeros(self.d_model, dtype=np.float64) if self.cfg.track_distortion else None

        for _, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            rx = arrays[0].reshape(-1)
            if rx.size != self.padded_size:
                raise ValueError(f"Client returned rx size {rx.size}, expected {self.padded_size}")
            rx_sum += rx.astype(np.float64, copy=False)

            if self.cfg.track_distortion:
                if len(arrays) < 2:
                    raise ValueError("track_distortion=True but client did not return ideal weighted delta")
                ideal = arrays[1].reshape(-1)
                if ideal.size != self.d_model:
                    raise ValueError(f"Client returned ideal size {ideal.size}, expected {self.d_model}")
                ideal_sum += ideal.astype(np.float64, copy=False)

        scale = math.sqrt(
            max(self.rho_ref, self.cfg.rho_eps)
            / (self.cfg.power_scaling_linear * self.cfg.noise_power_mw)
        )
        actual_noiseless = scale * rx_sum[: self.d_model]
        if ideal_sum is not None:
            self.last_distortion = float(np.linalg.norm(actual_noiseless - ideal_sum) ** 2)
        else:
            self.last_distortion = float("nan")

        # Receiver noise z_t is complex CN(0, sigma_z^2 I). We decode the real part.
        noise_std = math.sqrt(self.cfg.noise_power_mw / 2.0)
        noise_real = self.rng.normal(0.0, noise_std, size=self.padded_size)
        delta_hat_padded = scale * (rx_sum + noise_real)
        delta_hat_flat = delta_hat_padded[: self.d_model].astype(np.float32, copy=False)

        theta_prime_flat = model.flatten_parameters(self.theta_prime_ndarrays)
        next_flat = theta_prime_flat + delta_hat_flat
        next_params = model.unflatten_parameters(next_flat, self.shapes)

        metrics: Metrics = {
            "rho_ref": float(self.rho_ref),
            "distortion": float(self.last_distortion),
            "Nt": int(server_round * self.n_symbols),
            "N_symbols_per_round": int(self.n_symbols),
        }
        return ndarrays_to_parameters(next_params), metrics

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Metrics]]:
        if (
            server_round > 0
            and self.cfg.eval_every > 1
            and server_round % self.cfg.eval_every != 0
            and server_round != self.total_rounds
        ):
            return None

        params = parameters_to_ndarrays(parameters)
        model.set_parameters(self.eval_model, params, self.device)
        loss, accuracy = model.test(self.eval_model, self.testloader, self.device)
        nt = int(server_round * self.n_symbols)
        row = {
            "round": float(server_round),
            "Nt": float(nt),
            "accuracy": float(accuracy),
            "loss": float(loss),
            "distortion": float(self.last_distortion),
            "rho_ref": float(self.rho_ref),
            "m0": float(self.bs_examples),
        }
        self.history_rows.append(row)
        print(
            f"[{self.experiment_name}] round={server_round:04d} Nt={nt:07d} "
            f"acc={accuracy:.4f} loss={loss:.4f} distortion={self.last_distortion:.4e}"
        )
        return float(loss), {
            "accuracy": float(accuracy),
            "Nt": int(nt),
            "distortion": float(self.last_distortion),
            "rho_ref": float(self.rho_ref),
        }

    def save_history_csv(self) -> Path:
        csv_path = self.output_dir / f"{self.experiment_name}_history.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["round", "Nt", "accuracy", "loss", "distortion", "rho_ref", "m0"]
            )
            writer.writeheader()
            for row in self.history_rows:
                writer.writerow(row)
        return csv_path

    def save_model(self) -> Path:
        model_path = self.output_dir / f"{self.experiment_name}_model.pt"
        torch.save(self.eval_model.state_dict(), model_path)
        return model_path
