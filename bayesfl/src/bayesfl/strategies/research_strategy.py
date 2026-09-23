"""Strict full-participation Flower strategies for all three methods."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from flwr.common import Code, FitIns, GetPropertiesIns, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from bayesfl.config import ExperimentConfig, round_learning_rate
from bayesfl.logging_utils import CsvRecorder
from bayesfl.communication import payload_components
from bayesfl.experiment_state import RunState
from bayesfl.posterior.sparse import dedicated_rng
from bayesfl.posterior.aggregation import aggregate_fola
from bayesfl.posterior.gaussian import gaussian_product, inverse_softplus_np, softplus_np
from bayesfl.posterior.packing import ParameterLayout, pack_fola, unpack_fola
from .common import normalized_example_weights, weighted_average_arrays, weighted_metrics


class ResearchStrategy(FedAvg):
    """One strategy class with method-specific aggregation math."""

    def __init__(
        self,
        *,
        cfg: ExperimentConfig,
        layout: ParameterLayout,
        initial_arrays: Sequence[np.ndarray],
        run_dir: Path,
        logger,
        state: RunState | None = None,
    ) -> None:
        fraction_fit = cfg.federation.clients_per_round / float(cfg.federation.num_clients)
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=0.0,
            min_fit_clients=cfg.federation.clients_per_round,
            min_evaluate_clients=1,
            min_available_clients=cfg.federation.num_clients,
            accept_failures=False,
            initial_parameters=ndarrays_to_parameters(list(initial_arrays)),
            on_fit_config_fn=lambda rnd: {
                "server_round": int(rnd),
                "learning_rate": float(round_learning_rate(cfg.training, rnd)),
            },
        )
        self.state = state or RunState(cfg, layout, initial_arrays, run_dir)
        self._identity_by_proxy = {}
        self._recipient_by_proxy = {}
        self.cfg = cfg
        self.layout = layout
        self.logger = logger
        self.client_metrics = CsvRecorder(run_dir / "metrics" / "client_metrics.csv")
        self.round_metrics = CsvRecorder(run_dir / "metrics" / "round_train_metrics.csv")

        # Compression-study instrumentation.
        #
        # When disabled, no global/client posterior copies are retained
        # and no snapshot files are written.
        self._snapshot_enabled = bool(
            cfg.method == "fola"
            and cfg.output.save_full_client_posteriors
        )
        self._snapshot_rounds = {
            int(r)
            for r in cfg.output.full_client_posterior_rounds
        }
        self._snapshot_root = run_dir / "compression_snapshots"

        if self._snapshot_enabled:
            self._snapshot_root.mkdir(
                parents=True,
                exist_ok=True,
            )
            self._snapshot_current_global = [
                np.asarray(a).copy()
                for a in initial_arrays
            ]
        else:
            self._snapshot_current_global = None

    def _resolve_identities(self, client_manager):
        client_manager.wait_for(self.cfg.federation.num_clients)
        proxies = client_manager.all()
        if len(proxies) != self.cfg.federation.num_clients:
            raise RuntimeError("Expected the configured fixed client roster")
        for proxy_id, proxy in proxies.items():
            if proxy_id in self._identity_by_proxy:
                continue
            # These messages carry only scalar identity/control data. Arrays=0;
            # envelope bytes are explicitly unavailable rather than called free.
            attempt = self.state.ledger.event_count
            zero = payload_components([], method="control")
            self.state.ledger.record(round_id=0, client_id="identity_" + str(proxy_id),
                                     phase="initialize", direction="downlink", components=zero,
                                     attempt_id=attempt, serialized_tensor_bytes=0)
            res = proxy.get_properties(GetPropertiesIns(config={}), timeout=60.0, group_id=0)
            self.state.ledger.record(round_id=0, client_id="identity_" + str(proxy_id),
                                     phase="initialize", direction="uplink", components=zero,
                                     attempt_id=attempt, serialized_tensor_bytes=0)
            cid = res.properties.get("client_id")
            if res.status.code != Code.OK or isinstance(cid, bool) or not isinstance(cid, int):
                raise RuntimeError("Client get_properties must return an integer client_id")
            self._identity_by_proxy[proxy_id] = cid
        if sorted(self._identity_by_proxy.values()) != list(range(self.cfg.federation.num_clients)):
            raise RuntimeError("Logical client IDs must be unique and cover the partition manifest")
        return proxies

    def configure_fit(self, server_round, parameters, client_manager):
        if self.cfg.communication.deterministic_client_schedule or self.cfg.sparse_enabled:
            proxies = self._resolve_identities(client_manager)
        if self.cfg.communication.deterministic_client_schedule:
            logical_to_proxy = {self._identity_by_proxy[cid]: proxy for cid, proxy in proxies.items()}
            rng = dedicated_rng(self.cfg.runtime.seed, 0, server_round, domain="client_schedule")
            selected = sorted(rng.choice(self.cfg.federation.num_clients,
                                         size=self.cfg.federation.clients_per_round, replace=False).tolist())
            fit_config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
            instructions = [(logical_to_proxy[cid], FitIns(parameters, dict(fit_config))) for cid in selected]
        else:
            # Preserve the supplied sampler on legacy configurations.
            instructions = super().configure_fit(server_round, parameters, client_manager)
        self._recipient_by_proxy = {
            proxy.cid: str(self._identity_by_proxy.get(proxy.cid, proxy.cid))
            for proxy, _ in instructions
        }
        arrays = parameters_to_ndarrays(parameters)
        extra = self.state.prepare_round(
            server_round, list(self._recipient_by_proxy.values()), arrays,
            serialized_tensor_bytes=sum(len(t) for t in parameters.tensors),
        )
        return [(proxy, FitIns(ins.parameters, {**ins.config, **extra})) for proxy, ins in instructions]

    def _decode_arrival(self, server_round, proxy, fit_res):
        try:
            return parameters_to_ndarrays(fit_res.parameters)
        except Exception as exc:
            self.state.ledger.record_undecodable(
                round_id=server_round,
                client_id=self._recipient_by_proxy.get(proxy.cid, proxy.cid),
                serialized_tensor_bytes=sum(len(t) for t in fit_res.parameters.tensors),
                reason=f"Tensor deserialization failed: {exc}",
            )
            raise

    def aggregate_fit(self, server_round, results, failures):
        client_arrays, masks, counts, valid_results, errors = [], [], [], [], []
        # Charge every arrived reply, including a later-rejected packet. Continue
        # processing arrivals after an error, then fail the round as the baseline did.
        arrivals = list(results)
        for failure in failures:
            if isinstance(failure, tuple) and len(failure) == 2 and hasattr(failure[1], "parameters"):
                proxy, res = failure
                try:
                    arrays = self._decode_arrival(server_round, proxy, res)
                except Exception as exc:
                    errors.append(exc)
                    continue
                self.state.ledger.record(
                    round_id=server_round, client_id=self._recipient_by_proxy.get(proxy.cid, proxy.cid),
                    phase="fit", direction="uplink", status="rejected",
                    components=payload_components(arrays, method=self.cfg.method,
                                                  sparse_upload=self.cfg.sparse_enabled, tensor_count=self.layout.size),
                    serialized_tensor_bytes=sum(len(t) for t in res.parameters.tensors),
                )
        # Stable aggregation order is used for matched new experiments. Legacy
        # dense configurations retain their supplied result-order policy.
        if self.cfg.communication.deterministic_client_schedule:
            arrivals.sort(key=lambda pair: self._identity_by_proxy[pair[0].cid])
        for proxy, fit_res in arrivals:
            try:
                arrays = self._decode_arrival(server_round, proxy, fit_res)
                reconstructed, mask = self.state.receive_packet(
                    round_id=server_round, recipient_id=self._recipient_by_proxy.get(proxy.cid, proxy.cid),
                    arrays=arrays, metrics=dict(fit_res.metrics), num_examples=fit_res.num_examples,
                    expected_client_id=self._identity_by_proxy.get(proxy.cid),
                    serialized_tensor_bytes=sum(len(t) for t in fit_res.parameters.tensors),
                )
                client_arrays.append(reconstructed)
                masks.append(mask)
                counts.append(int(fit_res.num_examples))
                valid_results.append((proxy, fit_res))
            except Exception as exc:
                errors.append(exc)
        if failures or errors or len(valid_results) != self.cfg.federation.clients_per_round:
            first = (errors or failures or ["result-count mismatch"])[0]
            raise RuntimeError(f"Round {server_round}: rejected/failed/absent client results; first={first!r}")
        results = valid_results
        weights = normalized_example_weights(counts)
        if self.cfg.method == "fedavg":
            aggregated = weighted_average_arrays(client_arrays, weights)
        elif self.cfg.method == "bbb":
            aggregated = self._aggregate_bbb(client_arrays, weights)
        elif self.cfg.method == "fola":
            aggregated = self._aggregate_fola(client_arrays, weights)
        else:  # pragma: no cover
            raise ValueError(self.cfg.method)
        aggregated = self.state.finish_round(
            aggregated, masks=masks, counts=counts, client_metrics=[dict(r.metrics) for _, r in results],
        )

        # Snapshot only COPIES of the FOLA states. The actual aggregated
        # arrays returned to Flower are never reconstructed/compressed here,
        # so enabling this instrumentation cannot alter the subsequent
        # training trajectory.
        if self._snapshot_enabled:
            if self._snapshot_current_global is None:
                raise RuntimeError(
                    "FOLA snapshot global state was not initialized"
                )

            if server_round in self._snapshot_rounds:
                self._save_fola_compression_snapshot(
                    server_round=server_round,
                    incoming_global=self._snapshot_current_global,
                    client_arrays=client_arrays,
                    counts=counts,
                    weights=weights,
                    results=results,
                    outgoing_global=aggregated,
                )

            self._snapshot_current_global = [
                np.asarray(a).copy()
                for a in aggregated
            ]

        # Keep one row per client so heterogeneity and local behavior remain inspectable.
        for _, res in results:
            client_row = {
                "round": server_round,
                "num_examples": int(res.num_examples),
                **dict(res.metrics),
            }
            self.client_metrics.append(client_row)

        metrics = weighted_metrics([(int(res.num_examples), dict(res.metrics)) for _, res in results])
        row = {**metrics, "round": server_round, "num_clients": len(results), "num_examples": sum(counts), **self.state.evaluation_fields()}
        self.round_metrics.append(row)
        self.logger.info(
            "Round %d aggregation complete: clients=%d examples=%d train_loss=%s",
            server_round,
            len(results),
            sum(counts),
            f"{metrics.get('train_loss', float('nan')):.6f}",
        )
        return ndarrays_to_parameters(aggregated), metrics

    def _save_fola_state_npz(
        self,
        path: Path,
        arrays: Sequence[np.ndarray],
        *,
        extra: dict[str, np.ndarray] | None = None,
    ) -> None:
        means, precisions = unpack_fola(
            arrays,
            self.layout,
        )

        payload: dict[str, np.ndarray] = {}

        for name, arr in zip(
            self.layout.names,
            means,
        ):
            payload[f"mean__{name}"] = (
                np.asarray(arr).copy()
            )

        for name, arr in zip(
            self.layout.names,
            precisions,
        ):
            payload[f"precision__{name}"] = (
                np.asarray(arr).copy()
            )

        if extra:
            payload.update(extra)

        np.savez_compressed(
            path,
            **payload,
        )

    def _save_fola_compression_snapshot(
        self,
        *,
        server_round: int,
        incoming_global: Sequence[np.ndarray],
        client_arrays: Sequence[Sequence[np.ndarray]],
        counts: Sequence[int],
        weights: Sequence[float],
        results,
        outgoing_global: Sequence[np.ndarray],
    ) -> None:
        """Save trajectory-independent compression-study inputs.

        Nothing reconstructed here is fed back into training.
        """

        round_dir = (
            self._snapshot_root
            / f"round_{server_round:04d}"
        )

        round_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._save_fola_state_npz(
            round_dir / "incoming_global.npz",
            incoming_global,
        )

        self._save_fola_state_npz(
            round_dir
            / "outgoing_global_uncompressed.npz",
            outgoing_global,
        )

        clients_dir = round_dir / "clients"
        clients_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        manifest_clients = []

        for (
            (_, fit_res),
            arrays,
            count,
            weight,
        ) in zip(
            results,
            client_arrays,
            counts,
            weights,
        ):
            if "client_id" not in fit_res.metrics:
                raise RuntimeError(
                    "client_id missing from FOLA FitRes metrics"
                )

            client_id = int(
                round(
                    float(
                        fit_res.metrics[
                            "client_id"
                        ]
                    )
                )
            )

            filename = (
                f"client_{client_id:04d}.npz"
            )

            self._save_fola_state_npz(
                clients_dir / filename,
                arrays,
                extra={
                    "client_id": np.asarray(
                        client_id,
                        dtype=np.int64,
                    ),
                    "num_examples": np.asarray(
                        int(count),
                        dtype=np.int64,
                    ),
                    "aggregation_weight": np.asarray(
                        float(weight),
                        dtype=np.float64,
                    ),
                },
            )

            manifest_clients.append(
                {
                    "client_id": client_id,
                    "num_examples": int(count),
                    "aggregation_weight": float(
                        weight
                    ),
                    "file": (
                        "clients/"
                        + filename
                    ),
                }
            )

        manifest_clients.sort(
            key=lambda x: x["client_id"]
        )

        manifest = {
            "server_round": int(server_round),
            "method": "fola",
            "mode": self.cfg.fola.mode,
            "num_clients": len(
                manifest_clients
            ),
            "num_examples": int(
                sum(counts)
            ),
            "parameter_names": list(
                self.layout.names
            ),
            "aggregation_epsilon": float(
                self.cfg.fola.aggregation_epsilon
            ),
            "training_trajectory_modified": False,
            "incoming_global_file": (
                "incoming_global.npz"
            ),
            "outgoing_global_reference_file": (
                "outgoing_global_uncompressed.npz"
            ),
            "clients": manifest_clients,
        }

        with (
            round_dir / "manifest.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                manifest,
                handle,
                indent=2,
                sort_keys=True,
            )

        self.logger.info(
            "Saved FOLA compression snapshot: "
            "round=%d clients=%d path=%s",
            server_round,
            len(manifest_clients),
            round_dir,
        )

    def _aggregate_bbb(
        self,
        clients: Sequence[Sequence[np.ndarray]],
        weights: Sequence[float],
    ) -> list[np.ndarray]:
        # Optional ablation: directly FedAvg all variational parameters.
        if self.cfg.bbb.aggregation == "fedavg_variational":
            return weighted_average_arrays(clients, weights)

        result = weighted_average_arrays(clients, weights)
        name_to_idx = {name: idx for idx, name in enumerate(self.layout.names)}
        processed: set[str] = set()
        for rho_name, rho_idx in name_to_idx.items():
            if "rho_" not in rho_name:
                continue
            mu_name = rho_name.replace("rho_", "mu_", 1)
            if mu_name not in name_to_idx or mu_name in processed:
                continue
            mu_idx = name_to_idx[mu_name]
            means = [np.asarray(c[mu_idx]) for c in clients]
            variances = [np.maximum(softplus_np(np.asarray(c[rho_idx])) ** 2, 1e-12) for c in clients]
            precisions = [1.0 / var for var in variances]
            mu_global, precision_global = gaussian_product(means, precisions, weights)
            sigma_global = np.sqrt(1.0 / np.maximum(precision_global, 1e-12))
            rho_global = inverse_softplus_np(sigma_global).astype(clients[0][rho_idx].dtype, copy=False)
            result[mu_idx] = mu_global.astype(clients[0][mu_idx].dtype, copy=False)
            result[rho_idx] = rho_global
            processed.add(mu_name)
        return result

    def _aggregate_fola(self, clients, weights):
        return aggregate_fola(clients, weights, layout=self.layout, cfg=self.cfg)
