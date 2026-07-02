"""Grouped Flower clients for physical-device simulation with rich metrics."""

from __future__ import annotations

import json
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.client.client import Client
from flwr.common import Context
from flwr.common.typing import NDArrays, Scalar
from torch.utils.data import DataLoader, Dataset

import bayes_vi
import model
from config import RunConfig
from observability import SCHEMA_VERSION, array_stats, snr_values, vector_cosine
from selector import parse_selected_ids


class GroupedBayesClient(fl.client.NumPyClient):
    """A Flower virtual client that sequentially simulates many physical devices."""

    def __init__(
        self,
        group_id: int,
        device_ids: Sequence[int],
        trainsets: Sequence[Dataset],
        valsets: Sequence[Dataset],
        cfg: RunConfig,
        input_shape: Sequence[int],
        num_classes: int,
        initial_payload: NDArrays,
    ) -> None:
        model.configure_torch_threads(cfg.torch_threads)
        self.group_id = int(group_id)
        self.device_ids = [int(x) for x in device_ids]
        self.trainsets = trainsets
        self.valsets = valsets
        self.cfg = cfg
        self.input_shape = tuple(int(x) for x in input_shape)
        self.num_classes = int(num_classes)
        self.initial_payload = [np.asarray(x, dtype=np.float32) for x in initial_payload]
        self.device = model.resolve_device(cfg.device)
        self._train_loaders: Dict[int, DataLoader] = {}
        self._val_loaders: Dict[int, DataLoader] = {}

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return [arr.copy() for arr in self.initial_payload]

    def _loader_for(self, device_id: int) -> DataLoader:
        if device_id not in self._train_loaders:
            self._train_loaders[device_id] = DataLoader(
                self.trainsets[device_id],
                batch_size=int(self.cfg.batch_size),
                shuffle=True,
                num_workers=int(self.cfg.num_workers),
                pin_memory=(self.device.type == "cuda"),
            )
        return self._train_loaders[device_id]

    def _val_loader_for(self, device_id: int) -> DataLoader | None:
        if device_id >= len(self.valsets) or len(self.valsets[device_id]) == 0:
            return None
        if device_id not in self._val_loaders:
            self._val_loaders[device_id] = DataLoader(
                self.valsets[device_id],
                batch_size=int(self.cfg.batch_size),
                shuffle=False,
                num_workers=int(self.cfg.num_workers),
                pin_memory=(self.device.type == "cuda"),
            )
        return self._val_loaders[device_id]

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        server_round = int(config.get("server_round", 0))
        selected_ids = parse_selected_ids(str(config.get("selected_ids", "")))
        active_ids = [did for did in self.device_ids if did in selected_ids]

        if self.cfg.method == "fedavg":
            payload, examples, bundle = self._fit_fedavg(parameters, active_ids, server_round)
        elif self.cfg.method == "ola":
            payload, examples, bundle = self._fit_ola(parameters, active_ids, server_round)
        elif self.cfg.method == "vi":
            payload, examples, bundle = self._fit_vi(parameters, active_ids, server_round)
        else:
            raise ValueError(f"Unsupported method: {self.cfg.method}")

        train_rows = bundle.get("client_train_rows", [])
        eval_rows = bundle.get("client_eval_rows", [])
        avg_loss = float(bundle.get("train_loss", 0.0))
        metrics: Dict[str, Scalar] = {
            "group_id": int(self.group_id),
            "active_devices": int(len(active_ids)),
            "train_loss": float(avg_loss),
            "group_examples": int(examples),
            "client_train_json": json.dumps(train_rows),
            "client_eval_json": json.dumps(eval_rows),
        }
        # Include lightweight group summaries as scalars for Flower history.
        for key, value in bundle.get("group_metrics", {}).items():
            if isinstance(value, (int, float, str, bool)):
                metrics[str(key)] = value
        return payload, int(examples), metrics

    def _base_client_row(self, did: int, server_round: int, n: int) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": "",  # server fills this in when writing CSV files
            "round": int(server_round),
            "method": self.cfg.method,
            "physical_client_id": int(did),
            "virtual_client_id": int(self.group_id),
            "num_examples": int(n),
            "local_epochs": int(self.cfg.local_epochs),
            "batch_size": int(self.cfg.batch_size),
        }

    def _maybe_eval_local_model(self, did: int, server_round: int, local_model: torch.nn.Module) -> dict[str, object] | None:
        if int(self.cfg.local_eval_every) <= 0:
            return None
        if server_round > 0 and server_round % int(self.cfg.local_eval_every) != 0:
            return None
        loader = self._val_loader_for(did)
        if loader is None:
            return None
        loss, acc = model.evaluate(local_model, loader, self.device)
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": "",
            "round": int(server_round),
            "method": self.cfg.method,
            "physical_client_id": int(did),
            "virtual_client_id": int(self.group_id),
            "eval_scope": "local_val",
            "eval_dataset": "client_val",
            "num_eval_examples": int(len(self.valsets[did])),
            "local_accuracy": float(acc),
            "local_loss": float(loss),
            "local_nll": float(loss),
            "num_examples_train": int(len(self.trainsets[did])),
            "local_mc_samples": 1,
        }

    def _fit_fedavg(self, parameters: NDArrays, active_ids: Sequence[int], server_round: int) -> tuple[NDArrays, int, dict[str, object]]:
        global_mu = np.asarray(parameters[0], dtype=np.float32)
        if not active_ids:
            return [np.zeros_like(global_mu)], 0, {"train_loss": 0.0, "client_train_rows": [], "client_eval_rows": [], "group_metrics": {}}

        weighted = np.zeros_like(global_mu, dtype=np.float64)
        total_examples = 0
        loss_sum = 0.0
        train_rows: list[dict[str, object]] = []
        eval_rows: list[dict[str, object]] = []
        for did in active_ids:
            local_model = model.build_model(self.cfg, self.input_shape, self.num_classes).to(self.device)
            model.set_flat_parameters(local_model, global_mu, self.device)
            model.set_seed(int(self.cfg.seed + 1_000_003 * server_round + did))
            loader = self._loader_for(did)
            train_loss, _fisher, stats = model.train_deterministic(local_model, loader, self.device, self.cfg)
            flat = model.flatten_parameters(local_model)
            update = flat - global_mu
            n = len(self.trainsets[did])
            weighted += float(n) * flat.astype(np.float64, copy=False)
            total_examples += int(n)
            loss_sum += float(train_loss) * int(n)
            row = self._base_client_row(did, server_round, n)
            row.update(stats)
            row.update(
                {
                    "num_batches": int(stats.get("num_batches", 0)),
                    "update_l2_norm": float(np.linalg.norm(update.astype(np.float64))),
                    "update_linf_norm": float(np.max(np.abs(update))) if update.size else 0.0,
                    "update_cosine_to_global": vector_cosine(flat, global_mu),
                    "drift_from_global_before_l2": float(np.linalg.norm(update.astype(np.float64))),
                    "drift_from_global_after_l2": float("nan"),
                }
            )
            train_rows.append(row)
            erow = self._maybe_eval_local_model(did, server_round, local_model)
            if erow is not None:
                eval_rows.append(erow)
        return [(weighted / max(total_examples, 1)).astype(np.float32)], total_examples, {
            "train_loss": loss_sum / max(total_examples, 1),
            "client_train_rows": train_rows,
            "client_eval_rows": eval_rows,
            "group_metrics": {},
        }

    def _fit_ola(self, parameters: NDArrays, active_ids: Sequence[int], server_round: int) -> tuple[NDArrays, int, dict[str, object]]:
        global_mu = np.asarray(parameters[0], dtype=np.float32)
        global_precision = np.maximum(np.asarray(parameters[1], dtype=np.float32), float(self.cfg.precision_floor))
        if not active_ids:
            return [np.zeros_like(global_precision), np.zeros_like(global_precision)], 0, {"train_loss": 0.0, "client_train_rows": [], "client_eval_rows": [], "group_metrics": {}}

        precision_count_sum = np.zeros_like(global_precision, dtype=np.float64)
        precision_mu_count_sum = np.zeros_like(global_precision, dtype=np.float64)
        total_examples = 0
        loss_sum = 0.0
        train_rows: list[dict[str, object]] = []
        eval_rows: list[dict[str, object]] = []
        round_idx = max(int(server_round), 1)
        for did in active_ids:
            local_model = model.build_model(self.cfg, self.input_shape, self.num_classes).to(self.device)
            model.set_flat_parameters(local_model, global_mu, self.device)
            model.set_seed(int(self.cfg.seed + 2_000_003 * server_round + did))
            loader = self._loader_for(did)
            train_loss, fisher, stats = model.train_deterministic(
                local_model,
                loader,
                self.device,
                self.cfg,
                prior_mu=global_mu,
                prior_precision=global_precision,
                prior_lambda=float(self.cfg.ola_prior_lambda),
                collect_fisher=True,
            )
            if fisher is None:
                fisher = np.zeros_like(global_mu)
            local_mu = model.flatten_parameters(local_model)
            gamma = np.full_like(global_precision, fill_value=float(self.cfg.precision_init), dtype=np.float32)
            local_precision = (1.0 / round_idx) * fisher + ((round_idx - 1.0) / round_idx) * global_precision + (1.0 / round_idx) * gamma
            local_precision = np.maximum(local_precision.astype(np.float32, copy=False), float(self.cfg.precision_floor))
            local_sigma = np.sqrt(1.0 / local_precision)
            local_snr, _ = snr_values(local_mu, local_sigma)
            n = len(self.trainsets[did])
            precision_count_sum += float(n) * local_precision.astype(np.float64, copy=False)
            precision_mu_count_sum += float(n) * (local_precision * local_mu).astype(np.float64, copy=False)
            total_examples += int(n)
            loss_sum += float(train_loss) * int(n)
            update = local_mu - global_mu
            row = self._base_client_row(did, server_round, n)
            row.update(stats)
            row.update(
                {
                    "num_batches": int(stats.get("num_batches", 0)),
                    "update_l2_norm": float(np.linalg.norm(update.astype(np.float64))),
                    "update_linf_norm": float(np.max(np.abs(update))) if update.size else 0.0,
                    "update_cosine_to_global": vector_cosine(local_mu, global_mu),
                    "drift_from_global_before_l2": float(np.linalg.norm(update.astype(np.float64))),
                    "ola_task_loss": float(stats.get("task_loss", float("nan"))),
                    "ola_prior_loss": float(stats.get("prior_loss", float("nan"))),
                }
            )
            row.update({k.replace("posterior_precision_", "ola_precision_"): v for k, v in array_stats(local_precision, "posterior_precision").items()})
            row.update({k.replace("posterior_sigma_", "ola_sigma_"): v for k, v in array_stats(local_sigma, "posterior_sigma").items()})
            row.update({k.replace("posterior_snr_raw_", "ola_snr_raw_"): v for k, v in array_stats(local_snr, "posterior_snr_raw").items()})
            train_rows.append(row)
            erow = self._maybe_eval_local_model(did, server_round, local_model)
            if erow is not None:
                eval_rows.append(erow)
        return [precision_count_sum.astype(np.float32), precision_mu_count_sum.astype(np.float32)], total_examples, {
            "train_loss": loss_sum / max(total_examples, 1),
            "client_train_rows": train_rows,
            "client_eval_rows": eval_rows,
            "group_metrics": {},
        }

    def _fit_vi(self, parameters: NDArrays, active_ids: Sequence[int], server_round: int) -> tuple[NDArrays, int, dict[str, object]]:
        global_loc = np.asarray(parameters[0], dtype=np.float32)
        global_scale = np.maximum(np.asarray(parameters[1], dtype=np.float32), float(self.cfg.vi_min_scale))
        if not active_ids:
            zeros = np.zeros_like(global_loc)
            return [zeros, zeros], 0, {"train_loss": 0.0, "client_train_rows": [], "client_eval_rows": [], "group_metrics": {}}

        total_examples = 0
        loss_sum = 0.0
        train_rows: list[dict[str, object]] = []
        eval_rows: list[dict[str, object]] = []
        if self.cfg.bayes_aggregation == "product":
            first_sum = np.zeros_like(global_loc, dtype=np.float64)  # n * precision
            second_sum = np.zeros_like(global_loc, dtype=np.float64)  # n * precision * loc
        else:
            first_sum = np.zeros_like(global_loc, dtype=np.float64)  # n * loc
            second_sum = np.zeros_like(global_loc, dtype=np.float64)  # n * (var + loc^2)

        for did in active_ids:
            loader = self._loader_for(did)
            seed = int(self.cfg.seed + 3_000_003 * server_round + did)
            loc, scale, loss, vi_stats = bayes_vi.train_vi_local(
                trainloader=loader,
                input_shape=self.input_shape,
                num_classes=self.num_classes,
                hidden_dims=self.cfg.normalized_hidden(),
                global_loc=global_loc,
                global_scale=global_scale,
                device=self.device,
                cfg=self.cfg,
                seed=seed,
            )
            n = len(self.trainsets[did])
            if self.cfg.bayes_aggregation == "product":
                precision = 1.0 / np.maximum(scale * scale, float(self.cfg.vi_min_scale) ** 2)
                first_sum += float(n) * precision.astype(np.float64, copy=False)
                second_sum += float(n) * (precision * loc).astype(np.float64, copy=False)
            else:
                var = scale * scale
                first_sum += float(n) * loc.astype(np.float64, copy=False)
                second_sum += float(n) * (var + loc * loc).astype(np.float64, copy=False)
            total_examples += int(n)
            loss_sum += float(loss) * int(n)
            update = loc - global_loc
            row = self._base_client_row(did, server_round, n)
            row.update(
                {
                    "train_loss": float(loss),
                    "task_loss": float("nan"),
                    "prior_loss": float(vi_stats.get("vi_kl_loss", float("nan"))),
                    "regularization_loss": float(vi_stats.get("vi_kl_loss", float("nan"))),
                    "num_batches": float("nan"),
                    "update_l2_norm": float(np.linalg.norm(update.astype(np.float64))),
                    "update_linf_norm": float(np.max(np.abs(update))) if update.size else 0.0,
                    "update_cosine_to_global": vector_cosine(loc, global_loc),
                    "drift_from_global_before_l2": float(np.linalg.norm(update.astype(np.float64))),
                }
            )
            row.update(vi_stats)
            train_rows.append(row)
            # Evaluate VI mean parameters on local validation if requested.
            val_loader = self._val_loader_for(did)
            if int(self.cfg.local_eval_every) > 0 and (server_round == 0 or server_round % int(self.cfg.local_eval_every) == 0) and val_loader is not None:
                local_model = model.build_model(self.cfg, self.input_shape, self.num_classes).to(self.device)
                model.set_flat_parameters(local_model, loc, self.device)
                vloss, vacc = model.evaluate(local_model, val_loader, self.device)
                eval_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "run_id": "",
                        "round": int(server_round),
                        "method": self.cfg.method,
                        "physical_client_id": int(did),
                        "virtual_client_id": int(self.group_id),
                        "eval_scope": "local_val_mean_posterior",
                        "eval_dataset": "client_val",
                        "num_eval_examples": int(len(self.valsets[did])),
                        "local_accuracy": float(vacc),
                        "local_loss": float(vloss),
                        "local_nll": float(vloss),
                        "num_examples_train": int(n),
                        "local_mc_samples": 1,
                    }
                )
        return [first_sum.astype(np.float32), second_sum.astype(np.float32)], total_examples, {
            "train_loss": loss_sum / max(total_examples, 1),
            "client_train_rows": train_rows,
            "client_eval_rows": eval_rows,
            "group_metrics": {},
        }

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        del parameters, config
        return 0.0, 0, {"accuracy": 0.0}


def _partition_id_from_context(context: Context | str | int) -> int:
    """Support both current Context-based and older string-based client_fn APIs."""
    if isinstance(context, (str, int)):
        return int(context)
    node_config = getattr(context, "node_config", {})
    if "partition-id" in node_config:
        return int(node_config["partition-id"])
    if "partition_id" in node_config:
        return int(node_config["partition_id"])
    if "cid" in node_config:
        return int(node_config["cid"])
    raise KeyError("Could not infer partition id from Flower Context")


def gen_client_fn(
    device_groups: Sequence[Sequence[int]],
    trainsets: Sequence[Dataset],
    valsets: Sequence[Dataset],
    cfg: RunConfig,
    input_shape: Sequence[int],
    num_classes: int,
    initial_payload: NDArrays,
) -> Callable[[Context], Client]:
    """Generate Flower client factory for grouped physical-device simulation."""
    cache: Dict[int, Client] = {}

    def client_fn(context: Context) -> Client:
        group_id = _partition_id_from_context(context)
        if bool(cfg.cache_clients) and group_id in cache:
            return cache[group_id]
        fl_client = GroupedBayesClient(
            group_id=group_id,
            device_ids=device_groups[int(group_id)],
            trainsets=trainsets,
            valsets=valsets,
            cfg=cfg,
            input_shape=input_shape,
            num_classes=num_classes,
            initial_payload=initial_payload,
        ).to_client()
        if bool(cfg.cache_clients):
            cache[group_id] = fl_client
        return fl_client

    return client_fn
