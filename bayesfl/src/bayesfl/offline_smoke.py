"""Small SYNTHETIC CPU integration check; not a dataset/Flower benchmark.

Uses unchanged train_fola/train_fedavg and the production codec, reducer,
ledger and resume logic. Replaces only the data/model/transport environment.
"""
from __future__ import annotations

import argparse
import copy
import io
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bayesfl.config import (CommunicationConfig, CompressionConfig, DataConfig, ExperimentConfig,
                            FederationConfig, FOLAConfig, ModelConfig, RuntimeConfig, TrainingConfig)
from bayesfl.experiment_state import RunState
from bayesfl.logging_utils import CsvRecorder
from bayesfl.posterior.aggregation import aggregate_fola
from bayesfl.posterior.packing import (ParameterLayout, initial_fola_state, model_to_ndarrays,
                                      ndarrays_to_model, pack_fola, unpack_fola)
from bayesfl.posterior.sparse import compress_update, dedicated_rng
from bayesfl.runtime_utils import seed_everything
from bayesfl.strategies.common import normalized_example_weights, weighted_average_arrays
from bayesfl.training.deterministic import train_fedavg
from bayesfl.training.fola import train_fola


def smoke_config(method: str, *, rounds=3, keep_ratio=0.5, initial_precision=1.0, budget=None):
    cfg = ExperimentConfig(
        run_name="synthetic_protocol_check", method=method, data=DataConfig(),
        federation=FederationConfig(num_clients=3, clients_per_round=2), model=ModelConfig(),
        training=TrainingConfig(lr=0.01, lr_schedule="constant", local_epochs=2, batch_size=4,
                                rounds=rounds, momentum=0.0, grad_clip_norm=10.0),
        fola=FOLAConfig(mode="paper_reference", prior_lambda=0.01, initial_precision=initial_precision,
                        lambda_scale_by_size=False, paper_mean_only_eval=True),
        runtime=RuntimeConfig(seed=7, torch_num_threads=1, central_eval_device="cpu"),
        compression=CompressionConfig(keep_ratio=keep_ratio,
                                     precision_policy="floor_for_score" if initial_precision == 0 else "strict"),
        communication=CommunicationConfig(max_communication_bytes=budget, deterministic_client_schedule=True),
    )
    cfg.validate()
    return cfg


def tiny_model():
    return nn.Sequential(nn.Linear(4, 5), nn.Tanh(), nn.Linear(5, 2))


def synthetic_dataset(client_id: int):
    gen = torch.Generator().manual_seed(8309 + client_id)
    x = torch.randn(5 + 2 * client_id, 4, generator=gen)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).long()
    return TensorDataset(x, y)


def npy_roundtrip(arrays):
    payloads = []
    decoded = []
    for array in arrays:
        f = io.BytesIO()
        np.save(f, array, allow_pickle=False)
        payloads.append(f.getvalue())
        decoded.append(np.load(io.BytesIO(payloads[-1]), allow_pickle=False))
    return decoded, sum(len(p) for p in payloads)


def run_offline(cfg: ExperimentConfig, run_dir: Path, *, resume: bool = False) -> RunState:
    torch.set_num_threads(1)
    seed_everything(cfg.runtime.seed)
    blueprint = tiny_model()
    layout = ParameterLayout.from_model(blueprint)
    initial = initial_fola_state(blueprint, cfg.fola.initial_precision) if cfg.method == "fola" else model_to_ndarrays(blueprint)
    state = RunState(cfg, layout, initial, run_dir, partition_sha256="synthetic-fixed-v1", resume=resume)
    recorder = CsvRecorder(run_dir / "metrics" / "global_metrics.csv")

    def evaluate():
        model = copy.deepcopy(blueprint)
        means = unpack_fola(state.current, layout)[0] if cfg.method == "fola" else state.current
        ndarrays_to_model(model, means)
        x, y = synthetic_dataset(20).tensors
        with torch.no_grad():
            logits = model(x)
            accuracy = float((logits.argmax(1) == y).float().mean())
            loss = float(torch.nn.functional.cross_entropy(logits, y))
        row = dict(round=state.round_id, accuracy=accuracy, nll=loss,
                   global_accuracy=accuracy, global_loss=loss, **state.evaluation_fields())
        recorder.append(row)
        state.save_resume(dict(accuracy=accuracy, nll=loss))

    if not resume:
        evaluate()
    while state.stop_reason() == "running":
        rnd = state.round_id + 1
        rng = dedicated_rng(cfg.runtime.seed, 0, rnd, domain="client_schedule")
        ids = sorted(rng.choice(cfg.federation.num_clients, size=cfg.federation.clients_per_round, replace=False).tolist())
        down, down_size = npy_roundtrip(state.current)
        instruction = state.prepare_round(rnd, ids, down, serialized_tensor_bytes=down_size)
        contributions, masks, counts, all_metrics = [], [], [], []
        for cid in ids:
            seed = cfg.runtime.seed + 1_000_003 * cid + 10_007 * rnd
            seed_everything(seed)
            dataset = synthetic_dataset(cid)
            loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True,
                                generator=torch.Generator().manual_seed(seed))
            model = copy.deepcopy(blueprint)
            broadcast = [a.copy() for a in down]
            if cfg.method == "fola":
                means, precs = unpack_fola(broadcast, layout)
                ndarrays_to_model(model, means)
                local_p, metrics = train_fola(model, loader, cfg, server_round=rnd, device=torch.device("cpu"),
                                              global_mean_arrays=means, global_precision_arrays=precs,
                                              client_size=len(dataset), average_client_size=7.0)
                upload = pack_fola(model_to_ndarrays(model), local_p)
                if cfg.sparse_enabled:
                    upload, metadata = compress_update(broadcast, upload, layout, cfg, round_id=rnd,
                                                        client_id=cid, base_snapshot_id=instruction["base_snapshot_id"])
                    metrics.update(metadata)
            else:
                ndarrays_to_model(model, broadcast)
                metrics = train_fedavg(model, loader, cfg, server_round=rnd, device=torch.device("cpu"))
                upload = model_to_ndarrays(model)
            metrics["client_id"] = cid
            # Only these serialized arrays reach the production receive path.
            # Full local models and omitted posterior values are not passed in.
            del model
            wire, size = npy_roundtrip(upload)
            contribution, mask = state.receive_packet(round_id=rnd, recipient_id=cid, arrays=wire,
                                                        metrics=metrics, num_examples=len(dataset),
                                                        expected_client_id=cid, serialized_tensor_bytes=size)
            contributions.append(contribution)
            masks.append(mask)
            counts.append(len(dataset))
            all_metrics.append(metrics)
        weights = normalized_example_weights(counts)
        aggregate = (aggregate_fola(contributions, weights, layout=layout, cfg=cfg)
                     if cfg.method == "fola" else weighted_average_arrays(contributions, weights))
        state.finish_round(aggregate, masks=masks, counts=counts, client_metrics=all_metrics)
        evaluate()
    state.finalize()
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    report = []
    for method in ("fedavg_dense", "fola_dense", "fola_sparse_kl_global_local",
                   "fola_sparse_kl_local_global", "fola_sparse_random"):
        state = run_offline(smoke_config(method, rounds=args.rounds), args.output_dir / method)
        report.append(dict(method=method, rounds=state.round_id,
                           array_bytes=state.ledger.totals["all_array_bytes"],
                           stop_reason=state.stop_reason()))
    print("SYNTHETIC CPU PROTOCOL CHECK — not MNIST/CIFAR accuracy, not Flower/Ray")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
