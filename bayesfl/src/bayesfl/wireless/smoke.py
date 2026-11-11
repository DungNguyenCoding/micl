"""Synthetic CPU integration using the original trainers, not MNIST/CIFAR evidence."""
from __future__ import annotations
import argparse
import copy
import logging
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from bayesfl.config import (DataConfig, ExperimentConfig, FederationConfig, ModelConfig, TrainingConfig,
                           FOLAConfig, RuntimeConfig, CompressionConfig, CommunicationConfig)
from bayesfl.offline_smoke import tiny_model, synthetic_dataset
from bayesfl.logging_utils import CsvRecorder, save_resolved_config
from bayesfl.posterior.packing import (ParameterLayout, initial_fola_state, model_to_ndarrays,
                                      unpack_fola, pack_fola, ndarrays_to_model)
from bayesfl.runtime_utils import seed_everything
from bayesfl.training.fola import train_fola
from bayesfl.training.deterministic import train_fedavg
from bayesfl.wireless.config import WirelessConfig, ChannelConfig, ComputeConfig, SolverConfig
from bayesfl.wireless.state import WirelessState
from bayesfl.wireless.channel import Channel
from bayesfl.wireless.transport import LocalTransport
from bayesfl.wireless.controller import Controller


def smoke_config(policy="joint", selection="kl_global_local", method="fola", rounds=3):
    w = WirelessConfig(enabled=True, backend="local", policy=policy,
        uplink_bandwidth_hz=50_000.0 if policy == "dense" else 3000.0,
        downlink_bandwidth_hz=1e6, round_target_s=1.0,
        control_reserve_s=0.1, control_rate_bps=1e9, control_exchange_latency_s=1e-5,
        channel=ChannelConfig(mode="fixed", seed=23, path_gains=[0.5, 1.0, 2.0],
                              uplink_psd_w_hz=[1.0]*3, downlink_psd_w_hz=1.0, noise_psd_w_hz=1.0),
        compute=ComputeConfig(seconds_per_sample_epoch=[.001]*3, fixed_seconds=[0.0]*3, score_seconds=[.001]*3),
        solver=SolverConfig(max_bracket_queries=32, max_bisection_queries=40))
    cfg = ExperimentConfig(run_name="smoke_wireless_"+method+"_"+policy+"_"+selection,
        method=method, data=DataConfig(), model=ModelConfig(), federation=FederationConfig(3, 3),
        training=TrainingConfig(lr=.01, momentum=0, weight_decay=0, local_epochs=2,
                                batch_size=4, lr_schedule="constant", rounds=rounds, grad_clip_norm=10.0),
        fola=FOLAConfig(mode="paper_reference", prior_lambda=.01, initial_precision=1.0,
                        lambda_scale_by_size=False, paper_mean_only_eval=True),
        runtime=RuntimeConfig(seed=7, central_eval_device="cpu", torch_num_threads=1),
        compression=CompressionConfig(selection_rule="dense" if policy == "dense" else selection, keep_ratio=1.0),
        communication=CommunicationConfig(downlink_mode="multicast", deterministic_client_schedule=True),
        wireless=w)
    cfg.validate()
    return cfg


def synthetic_trainer(cfg, cid, partition_path, average_size, parameters, instruction):
    r = instruction["round_id"]
    seed = cfg.runtime.seed + 1_000_003*cid + 10_007*r
    seed_everything(seed)
    model = tiny_model()
    layout = ParameterLayout.from_model(model)
    ds = synthetic_dataset(cid)
    loader = DataLoader(ds, batch_size=cfg.training.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    if cfg.method == "fola":
        means, ps = unpack_fola(parameters, layout)
        ndarrays_to_model(model, means)
        p, metrics = train_fola(model, loader, cfg, server_round=r, device=torch.device("cpu"),
                               global_mean_arrays=means, global_precision_arrays=ps,
                               client_size=len(ds), average_client_size=average_size)
        out = pack_fola(model_to_ndarrays(model), p)
    else:
        ndarrays_to_model(model, parameters)
        metrics = train_fedavg(model, loader, cfg, server_round=r, device=torch.device("cpu"))
        out = model_to_ndarrays(model)
    return out, len(ds), metrics, layout


def run_smoke(cfg, directory: Path, *, resume=False, transport=None):
    torch.set_num_threads(1)
    directory.mkdir(parents=True, exist_ok=resume)
    seed_everything(cfg.runtime.seed)
    model = tiny_model()
    layout = ParameterLayout.from_model(model)
    initial = initial_fola_state(model, cfg.fola.initial_precision) if cfg.method == "fola" else model_to_ndarrays(model)
    channel = Channel(cfg.wireless, 3)
    state = WirelessState(cfg, layout, initial, directory, partition_sha256="synthetic-wireless-v2", resume=resume)
    save_resolved_config(cfg, directory)
    samples = [len(synthetic_dataset(i)) for i in range(3)]
    transport = transport or LocalTransport(cfg, "synthetic", sum(samples)/3, trainer=synthetic_trainer)
    controller = Controller(cfg, state, channel, transport, samples, logging.getLogger("wireless-smoke"))
    recorder = CsvRecorder(directory/"metrics/global_metrics.csv")
    def evaluate(r, parameters):
        ndarrays_to_model(model, unpack_fola(parameters, layout)[0] if cfg.method == "fola" else parameters)
        model.eval()
        with torch.no_grad():
            x, y = synthetic_dataset(20).tensors
            logits = model(x)
            loss = float(torch.nn.functional.cross_entropy(logits, y))
            acc = float((logits.argmax(1) == y).float().mean())
        metrics = dict(round=r, accuracy=acc, nll=loss, global_accuracy=acc, global_loss=loss)
        if cfg.method == "fola":
            metrics["fola_mean_accuracy"] = acc
        recorder.append(dict(metrics, **state.evaluation_fields()))
        state.save_resume(metrics)
        return loss, {"accuracy":acc}
    controller.run(evaluate)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("rounds must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for policy, selection, method in (("joint","kl_global_local","fola"), ("joint","random","fola"),
            ("equal_bandwidth","kl_global_local","fola"), ("uniform","kl_global_local","fola"),
            ("dense","dense","fola"), ("dense","dense","fedavg")):
        cfg = smoke_config(policy, selection, method, args.rounds)
        state = run_smoke(cfg, args.output_dir/cfg.run_name)
        results.append(dict(method=cfg.method_id, rounds=state.round_id, modeled_seconds=state.elapsed_s,
                            array_bytes=state.total_bytes, control_bytes=state.control_bytes))
    print("PASS: SYNTHETIC CPU protocol/trainer check; not dataset performance or Flower/Ray certification")
    print("Dense cases use a larger bandwidth solely to exercise dense-format code paths.")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
