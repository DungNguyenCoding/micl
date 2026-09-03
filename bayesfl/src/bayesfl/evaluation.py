"""Centralized evaluation and posterior diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from bayesfl.config import ExperimentConfig
from bayesfl.logging_utils import CsvRecorder
from bayesfl.metrics import predictive_metric_bundle
from bayesfl.models.factory import build_model
from bayesfl.posterior.gaussian import softplus_np
from bayesfl.posterior.packing import ParameterLayout, ndarrays_to_model, unpack_fola
from bayesfl.runtime_utils import resolve_device, seed_everything


class CentralEvaluator:
    def __init__(
        self,
        cfg: ExperimentConfig,
        test_loader: DataLoader,
        run_dir: Path,
        *,
        logger,
    ) -> None:
        self.cfg = cfg
        self.test_loader = test_loader
        self.run_dir = run_dir
        self.logger = logger
        self.device = resolve_device(cfg.runtime.central_eval_device)
        self.model = build_model(cfg).to(self.device)
        self.layout = ParameterLayout.from_model(self.model)
        self.metrics_recorder = CsvRecorder(run_dir / "metrics" / "global_metrics.csv")
        self.posterior_recorder = CsvRecorder(run_dir / "posterior" / "posterior_summary.csv")

    @torch.no_grad()
    def _collect_probabilities(
        self,
        parameters: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        method = self.cfg.method
        if method == "fola":
            mean_arrays, precision_arrays = unpack_fola(parameters, self.layout)
            ndarrays_to_model(self.model, mean_arrays)
        else:
            ndarrays_to_model(self.model, parameters)
            precision_arrays = None

        self.model.eval()
        all_primary_probs: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        all_sample_probs: list[np.ndarray] = []
        all_point_probs: list[np.ndarray] = []

        if method == "bbb":
            mc = self.cfg.bbb.mc_eval
        elif method == "fola":
            mc = 1 if self.cfg.fola.paper_mean_only_eval else self.cfg.fola.mc_eval
        else:
            mc = 1

        mean_tensors = None
        precision_tensors = None
        if method == "fola":
            params = [p for _, p in self.model.named_parameters()]
            mean_tensors = [
                torch.as_tensor(a, device=self.device, dtype=p.dtype)
                for p, a in zip(params, mean_arrays)
            ]
            precision_tensors = [
                torch.as_tensor(a, device=self.device, dtype=p.dtype).clamp_min(
                    self.cfg.fola.precision_min
                )
                for p, a in zip(params, precision_arrays)
            ]

        for x, y in self.test_loader:
            x = x.to(self.device, non_blocking=True)

            if method == "fola":
                assert mean_tensors is not None and precision_tensors is not None
                for (_, param), mean in zip(self.model.named_parameters(), mean_tensors):
                    param.copy_(mean)
                point_logits = self.model(x)
                point_probs = torch.softmax(point_logits, dim=1)
                all_point_probs.append(point_probs.cpu().numpy())

                if self.cfg.fola.paper_mean_only_eval:
                    # The paper reports global accuracy of the aggregated mode,
                    # not Monte Carlo samples from the Laplace covariance.
                    all_primary_probs.append(point_probs.cpu().numpy())
                    all_labels.append(y.numpy())
                    continue

            samples = []
            for _ in range(mc):
                if method == "fola":
                    assert mean_tensors is not None and precision_tensors is not None
                    for (_, param), mean, precision in zip(
                        self.model.named_parameters(), mean_tensors, precision_tensors
                    ):
                        param.copy_(mean + torch.randn_like(mean) * torch.rsqrt(precision))
                logits = self.model(x)
                samples.append(torch.softmax(logits, dim=1))

            stack = torch.stack(samples, dim=0)
            all_primary_probs.append(stack.mean(dim=0).cpu().numpy())
            all_labels.append(y.numpy())
            if mc > 1:
                all_sample_probs.append(stack.cpu().numpy())

        if method == "fola":
            ndarrays_to_model(self.model, mean_arrays)

        primary_probs = np.concatenate(all_primary_probs, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        sample_probs = (
            np.concatenate(all_sample_probs, axis=1) if all_sample_probs else None
        )
        point_probs = np.concatenate(all_point_probs, axis=0) if all_point_probs else None
        return primary_probs, labels, sample_probs, point_probs

    def evaluate(self, server_round: int, parameters: Sequence[np.ndarray]) -> tuple[float, dict[str, float]]:
        seed_everything(self.cfg.runtime.seed + 99_991 * int(server_round))
        primary_probs, labels, sample_probs, point_probs = self._collect_probabilities(parameters)

        if self.cfg.method != "fola":
            metrics, ece = predictive_metric_bundle(
                primary_probs,
                labels,
                sample_probabilities=sample_probs,
                n_bins=15,
            )
        else:
            if point_probs is None:  # pragma: no cover
                raise RuntimeError("FOLA point probabilities were not collected")
            point_metrics, ece = predictive_metric_bundle(
                point_probs,
                labels,
                sample_probabilities=None,
                n_bins=15,
            )
            metrics = dict(point_metrics)
            metrics["fola_mean_accuracy"] = point_metrics["accuracy"]
            metrics["fola_mean_nll"] = point_metrics["nll"]
            metrics["fola_mean_brier"] = point_metrics["brier"]
            metrics["fola_mean_ece"] = point_metrics["ece"]
            metrics["fola_mean_mce"] = point_metrics["mce"]
            metrics["fola_mean_mean_confidence"] = point_metrics["mean_confidence"]
            metrics["fola_mean_predictive_entropy"] = point_metrics["predictive_entropy"]
            metrics["fola_mean_expected_entropy"] = point_metrics["expected_entropy"]
            metrics["fola_mean_mutual_information"] = point_metrics["mutual_information"]

            if not self.cfg.fola.paper_mean_only_eval:
                mc_metrics, _ = predictive_metric_bundle(
                    primary_probs,
                    labels,
                    sample_probabilities=sample_probs,
                    n_bins=15,
                )
                for key, value in mc_metrics.items():
                    metrics[f"fola_mc_{key}"] = value

        metrics["round"] = float(server_round)
        self.metrics_recorder.append({"round": server_round, **metrics})

        np.savez_compressed(
            self.run_dir / "reliability" / f"round_{server_round:04d}.npz",
            bin_edges=ece.bin_edges,
            bin_accuracy=ece.bin_accuracy,
            bin_confidence=ece.bin_confidence,
            bin_count=ece.bin_count,
        )
        self._record_posterior(server_round, parameters)
        if server_round == 0 or (
            self.cfg.output.checkpoint_every > 0
            and server_round % self.cfg.output.checkpoint_every == 0
        ):
            self._save_checkpoint(server_round, parameters)

        if self.cfg.method == "fola":
            if self.cfg.fola.paper_mean_only_eval:
                self.logger.info(
                    "Round %d centralized eval: mean_loss=%.6f mean_acc=%.4f mean_ece=%.4f",
                    server_round,
                    metrics["nll"],
                    metrics["accuracy"],
                    metrics["ece"],
                )
            else:
                self.logger.info(
                    "Round %d centralized eval: mean_loss=%.6f mean_acc=%.4f mean_ece=%.4f "
                    "mc_acc=%.4f mc_nll=%.6f mc_ece=%.4f mi=%.6f",
                    server_round,
                    metrics["nll"],
                    metrics["accuracy"],
                    metrics["ece"],
                    metrics.get("fola_mc_accuracy", float("nan")),
                    metrics.get("fola_mc_nll", float("nan")),
                    metrics.get("fola_mc_ece", float("nan")),
                    metrics.get("fola_mc_mutual_information", float("nan")),
                )
        else:
            self.logger.info(
                "Round %d centralized eval: loss=%.6f acc=%.4f ece=%.4f mi=%.6f",
                server_round,
                metrics["nll"],
                metrics["accuracy"],
                metrics["ece"],
                metrics["mutual_information"],
            )
        flower_metrics = {k: float(v) for k, v in metrics.items() if k != "round"}
        return float(metrics["nll"]), flower_metrics

    def _record_posterior(self, server_round: int, parameters: Sequence[np.ndarray]) -> None:
        names = self.layout.names
        if self.cfg.method == "bbb":
            name_to_arr = {name: np.asarray(a) for name, a in zip(names, parameters)}
            for name in names:
                if "mu_" not in name:
                    continue
                rho_name = name.replace("mu_", "rho_", 1)
                if rho_name not in name_to_arr:
                    continue
                mu = name_to_arr[name]
                sigma = softplus_np(name_to_arr[rho_name])
                snr = np.abs(mu) / np.maximum(sigma, 1e-12)
                self.posterior_recorder.append(
                    {
                        "round": server_round,
                        "parameter": name,
                        "kind": "bbb",
                        "numel": mu.size,
                        "mean_abs": float(np.mean(np.abs(mu))),
                        "sigma_mean": float(np.mean(sigma)),
                        "sigma_min": float(np.min(sigma)),
                        "sigma_max": float(np.max(sigma)),
                        "snr_mean": float(np.mean(snr)),
                    }
                )
        elif self.cfg.method == "fola":
            means, precisions = unpack_fola(parameters, self.layout)
            for name, mean, precision in zip(names, means, precisions):
                raw_p = np.asarray(precision)
                p = np.maximum(raw_p, self.cfg.fola.precision_min)
                sigma = 1.0 / np.sqrt(p)
                self.posterior_recorder.append(
                    {
                        "round": server_round,
                        "parameter": name,
                        "kind": "fola",
                        "numel": mean.size,
                        "mean_abs": float(np.mean(np.abs(mean))),
                        "sigma_mean": float(np.mean(sigma)),
                        "precision_mean": float(np.mean(raw_p)),
                        "precision_min": float(np.min(raw_p)),
                        "precision_max": float(np.max(raw_p)),
                        "zero_precision_fraction": float(np.mean(raw_p <= 0)),
                    }
                )

    def _save_checkpoint(self, server_round: int, parameters: Sequence[np.ndarray]) -> None:
        payload: dict[str, np.ndarray] = {}
        if self.cfg.method == "fola":
            means, precisions = unpack_fola(parameters, self.layout)
            for name, arr in zip(self.layout.names, means):
                payload[f"mean__{name}"] = np.asarray(arr)
            for name, arr in zip(self.layout.names, precisions):
                payload[f"precision__{name}"] = np.asarray(arr)
        else:
            for name, arr in zip(self.layout.names, parameters):
                payload[name] = np.asarray(arr)
        np.savez_compressed(
            self.run_dir / "checkpoints" / f"global_round_{server_round:04d}.npz",
            **payload,
        )
