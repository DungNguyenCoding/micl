"""Multi-phase client. Only PREPARE calls the original local trainer.

The cache is client-owned. Neither scores nor unselected posterior values are
returned by PREPARE/PRICE/TAIL. FlowerStore persists across ephemeral ClientApps.
"""
from __future__ import annotations
import json
import math
import time
from pathlib import Path
from typing import Protocol
import numpy as np
import torch
from bayesfl.posterior.packing import ParameterLayout, model_to_ndarrays, ndarrays_to_model, pack_fola, unpack_fola
from bayesfl.posterior.sparse import (posterior_vectors, flatten, layout_manifest, dedicated_rng,
    diagonal_kl, score_variance, snapshot_id)
from bayesfl.wireless.oracle import ScoreState
from bayesfl.wireless.payload import integer
from bayesfl.wireless.protocol import encode, validate_grant, canonical, binding


class Store(Protocol):
    def load(self): ...
    def save(self, meta: dict, arrays: dict[str, np.ndarray]): ...
    def clear(self): ...


class MemoryStore:
    """Local emulation only; not used to assume persistent Flower Python objects."""
    def __init__(self):
        self.value = None

    def load(self):
        return self.value

    def save(self, meta, arrays):
        self.value = (json.loads(canonical(meta)), {k: v.copy() for k, v in arrays.items()})

    def clear(self):
        self.value = None


class FlowerStore:
    """Keep state in the node's Context.state, never on the server strategy."""
    def __init__(self, state):
        self.state = state

    def load(self):
        if "wireless_meta" not in self.state:
            return None
        meta = json.loads(self.state["wireless_meta"]["json"])
        keys = meta.pop("_array_keys")
        arrays = self.state["wireless_arrays"].to_numpy_ndarrays(keep_input=True)
        if len(keys) != len(arrays):
            raise ValueError("Corrupt client Context state")
        return meta, dict(zip(keys, arrays))

    def save(self, meta, arrays):
        from flwr.app import ConfigRecord, ArrayRecord
        keys = sorted(arrays)
        self.state["wireless_arrays"] = ArrayRecord(numpy_ndarrays=[arrays[k] for k in keys])
        self.state["wireless_meta"] = ConfigRecord({"json": canonical(dict(meta, _array_keys=keys)).decode()})

    def clear(self):
        for name in ("wireless_meta", "wireless_arrays"):
            if name in self.state:
                del self.state[name]


def train_local(cfg, client_id, partition_path, average_client_size, parameters, instruction):
    """Identical trainer calls and per-client/round seeds to the baseline fit."""
    from bayesfl.models.factory import build_model
    from bayesfl.runtime_utils import seed_everything, resolve_device, release_cuda_memory
    from bayesfl.data.datasets import load_client_loader
    from bayesfl.training.fola import train_fola
    from bayesfl.training.deterministic import train_fedavg
    r = integer(instruction["round_id"], "round_id")
    seed = cfg.runtime.seed + 1_000_003 * client_id + 10_007 * r
    seed_everything(seed)
    torch.set_num_threads(max(1, int(cfg.runtime.torch_num_threads)))
    device = resolve_device("auto")
    loader, n = load_client_loader(cfg, partition_path, client_id, shuffle_seed=seed)
    model = build_model(cfg).to(device)
    layout = ParameterLayout.from_model(model)
    try:
        base = binding(parameters, layout, round_id=r, client_id=client_id,
                       run_token=instruction["run_token"], posterior=cfg.method == "fola")
        for key, value in base.items():
            if instruction.get(key) != value:
                raise ValueError(f"PREPARE binding mismatch before training: {key}")
        if cfg.method == "fola":
            means, precs = unpack_fola(parameters, layout)
            ndarrays_to_model(model, means)
            local_p, metrics = train_fola(model, loader, cfg, server_round=r, device=device,
                global_mean_arrays=means, global_precision_arrays=precs,
                client_size=n, average_client_size=average_client_size)
            arrays = pack_fola(model_to_ndarrays(model), local_p)
        elif cfg.method == "fedavg":
            ndarrays_to_model(model, parameters)
            metrics = train_fedavg(model, loader, cfg, server_round=r, device=device)
            arrays = model_to_ndarrays(model)
        else:
            raise ValueError("Unsupported wireless local trainer")
        return arrays, n, metrics, layout
    finally:
        del model
        release_cuda_memory()


def _score_from_cache(arrays):
    if "scores" not in arrays:
        raise ValueError("This dense-only client has no score oracle")
    state = ScoreState.__new__(ScoreState)
    state.scores = arrays["scores"]
    state.order = arrays["order"]
    state.tails = arrays["tails"]
    state.negative_sorted = arrays["negative_sorted"]
    state.sorted_scores = -state.negative_sorted
    return state


class ClientEngine:
    def __init__(self, cfg, client_id, partition_path, average_client_size, store: Store, *, trainer=None):
        self.cfg, self.client_id, self.store = cfg, int(client_id), store
        self.partition_path, self.average_client_size = str(partition_path), float(average_client_size)
        self.trainer = trainer or train_local

    def handle(self, parameters, instruction):
        phase = instruction.get("phase")
        if phase == "prepare":
            return self._prepare(parameters, instruction)
        if parameters:
            raise ValueError("Only PREPARE may carry a model downlink")
        cached = self.store.load()
        if cached is None:
            raise ValueError("No prepared client state for this request")
        meta, arrays = cached
        for key in ("round_id", "client_id", "run_token", "base_snapshot_id", "layout_id", "protocol"):
            if instruction.get(key) != meta["binding"][key]:
                raise ValueError(f"Stale/mismatched client request: {key}")
        if phase == "release":
            if not meta["uploaded"]:
                raise ValueError("Cannot release before upload")
            self.store.clear()
            return [], meta["n"], dict(meta["binding"], released=True)
        if meta["uploaded"]:
            raise ValueError("Already uploaded; repeated query/upload is not allowed")
        if phase == "price":
            s = _score_from_cache(arrays)
            response = s.at_price(instruction["price"], instruction["weight"], instruction["window_s"], instruction["efficiency"])
            return [], meta["n"], dict(meta["binding"], count=response.count, discarded_kl=response.discarded_kl)
        if phase == "tail":
            response = _score_from_cache(arrays).at_count(integer(instruction["count"], "count"))
            return [], meta["n"], dict(meta["binding"], count=response.count, discarded_kl=response.discarded_kl)
        if phase != "upload":
            raise ValueError(f"Unknown wireless phase: {phase}")
        grant = json.loads(instruction["grant_json"])
        validate_grant(grant)
        for key, value in meta["binding"].items():
            if grant.get(key) != value:
                raise ValueError(f"Grant binding mismatch: {key}")
        m, d = grant["count"], meta["d"]
        if grant["d"] != d or grant["posterior"] != (self.cfg.method == "fola"):
            raise ValueError("Grant representation/dimension mismatch")
        if m == d:
            mask = np.ones(d, dtype=bool)
        elif self.cfg.compression.selection_rule == "random":
            mask = np.zeros(d, dtype=bool)
            rng = dedicated_rng(self.cfg.runtime.seed, self.client_id, meta["binding"]["round_id"], domain="wireless-mask-v2")
            mask[rng.choice(d, m, replace=False)] = True
        else:
            mask = _score_from_cache(arrays).mask(m)
        packet, packet_meta = encode(arrays["mu"], arrays.get("precision"), mask, grant)
        actual_tail = float(math.fsum(arrays["scores"][~mask])) if "scores" in arrays else 0.0
        packet_meta.update(actual_discarded_kl=actual_tail,
                           local_steps=meta["local_steps"],
                           score_global_floored_fraction=meta["global_floor_fraction"],
                           score_local_floored_fraction=meta["local_floor_fraction"])
        # Mark before returning: no uncharged retry or duplicate acceptance.
        meta["uploaded"] = True
        self.store.save(meta, arrays)
        return packet, meta["n"], packet_meta

    def _prepare(self, parameters, instruction):
        if self.store.load() is not None:
            raise ValueError("Unreleased client state exists; do not silently retrain or discard it")
        if integer(instruction["client_id"], "client_id", 0) != self.client_id:
            raise ValueError("Wrong logical client")
        started = time.perf_counter()
        updated, n, metrics, layout = self.trainer(self.cfg, self.client_id, self.partition_path,
                                                  self.average_client_size, parameters, instruction)
        n = integer(n, "client sample count")
        base = binding(parameters, layout, round_id=instruction["round_id"], client_id=self.client_id,
                       run_token=instruction["run_token"], posterior=self.cfg.method == "fola")
        for key, value in base.items():
            if instruction.get(key) != value:
                raise ValueError(f"PREPARE binding mismatch: {key}")
        arrays, gf, lf = {}, 0.0, 0.0
        if self.cfg.method == "fola":
            mu, precision = posterior_vectors(updated, layout)
            arrays.update(mu=mu, precision=precision)
            if self.cfg.wireless.policy != "dense":
                global_mu, global_p = posterior_vectors(parameters, layout)
                gv, gf = score_variance(global_p, self.cfg.compression.precision_policy, self.cfg.score_precision_floor)
                lv, lf = score_variance(precision, self.cfg.compression.precision_policy, self.cfg.score_precision_floor)
                s = ScoreState(diagonal_kl(global_mu, gv, mu, lv))
                arrays.update(scores=s.scores, order=s.order, tails=s.tails, negative_sorted=s.negative_sorted)
        else:
            arrays["mu"] = flatten(updated, layout)
            if not np.isfinite(arrays["mu"]).all():
                raise ValueError("Nonfinite local FedAvg state")
        meta = dict(binding=base, n=n, d=arrays["mu"].size, uploaded=False,
                    local_steps=int(metrics.get("local_steps", 0)),
                    global_floor_fraction=gf, local_floor_fraction=lf)
        self.store.save(meta, arrays)
        return [], n, dict(base, d=meta["d"], local_steps=meta["local_steps"],
                           preparation_wall_seconds=time.perf_counter() - started,
                           score_global_floored_fraction=gf, score_local_floored_fraction=lf)
