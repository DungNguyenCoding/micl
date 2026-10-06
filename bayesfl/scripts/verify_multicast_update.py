#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json

from bayesfl.communication import core_round_bytes
from bayesfl.config import (
    CommunicationConfig,
    DataConfig,
    ExperimentConfig,
    FederationConfig,
    ModelConfig,
    TrainingConfig,
    round_learning_rate,
)
from bayesfl.experiment_state import config_fingerprint
from bayesfl.models.factory import build_model


def main() -> None:
    # New MNIST model.
    cfg = ExperimentConfig(
        run_name="verify_mnist",
        method="fola",
        data=DataConfig(dataset="mnist"),
        federation=FederationConfig(num_clients=100, clients_per_round=100),
        model=ModelConfig(name="mlp_784_256x5_10"),
        training=TrainingConfig(rounds=120),
        communication=CommunicationConfig(downlink_mode="multicast"),
    )
    cfg.validate()
    model = build_model(cfg)
    d = sum(p.numel() for p in model.parameters())
    assert d == 466_698, d
    print(f"PASS new MNIST parameter count: {d:,}")

    # Legacy architecture remains selectable.
    cfg.model.name = "mlp_784_500_300_10"
    cfg.validate()
    legacy = build_model(cfg)
    old_d = sum(p.numel() for p in legacy.parameters())
    assert old_d == 545_810, old_d
    print(f"PASS legacy MNIST model preserved: {old_d:,}")

    # CIFAR cosine schedule clamps at the R400 minimum from R400 onward.
    t = TrainingConfig(lr=0.02, lr_schedule="cosine", lr_min=0.0001, lr_decay_rounds=400)
    assert round_learning_rate(t, 399) > 0.0001
    for r in (400, 401, 500, 760, 1089):
        assert round_learning_rate(t, r) == 0.0001
    print("PASS CIFAR LR clamp: R400 onward = 0.0001")

    # Exact multicast formulas.
    cifar_d, cifar_k = 878_538, 20
    fola = core_round_bytes(cifar_d, cifar_k, method="fola", downlink_mode="multicast")
    fedavg = core_round_bytes(cifar_d, cifar_k, method="fedavg", downlink_mode="multicast")
    assert fola["total_bytes"] == 8 * cifar_d * (cifar_k + 1)
    assert fedavg["total_bytes"] == 4 * cifar_d * (cifar_k + 1)
    print(f"PASS dense FOLA multicast/round: {fola['total_bytes']:,} bytes")
    print(f"PASS dense FedAvg multicast/round: {fedavg['total_bytes']:,} bytes")
    assert 120 * fola["total_bytes"] == 17_711_326_080
    print("PASS CIFAR dense-FOLA R120 budget: 17,711,326,080 bytes")

    # New MNIST budget reference.
    mnist = core_round_bytes(466_698, 100, method="fola", downlink_mode="multicast")
    assert mnist["total_bytes"] == 377_091_984
    assert 120 * mnist["total_bytes"] == 45_251_038_080
    print("PASS MNIST dense-FOLA R120 budget: 45,251,038,080 bytes")

    # Fingerprint compatibility: the newly explicit accounting mode is excluded
    # from the optimization fingerprint so a migrated old CIFAR checkpoint can resume.
    c = ExperimentConfig(
        run_name="verify_fingerprint",
        method="fola",
        data=DataConfig(dataset="cifar10"),
        federation=FederationConfig(num_clients=20, clients_per_round=10),
        model=ModelConfig(name="paper_basiccnn"),
        training=TrainingConfig(rounds=150),
        communication=CommunicationConfig(deterministic_client_schedule=True, downlink_mode="unicast"),
    )
    c.validate()
    historical = c.to_dict()
    historical["training"].pop("rounds")
    historical["communication"].pop("max_communication_bytes")
    historical["communication"].pop("downlink_mode")
    historical_hash = hashlib.sha256(json.dumps(historical, sort_keys=True).encode()).hexdigest()
    assert config_fingerprint(c) == historical_hash
    c.communication.downlink_mode = "multicast"
    c.validate()
    assert config_fingerprint(c) == historical_hash
    print("PASS accounting-mode change is fingerprint-neutral; ledger migration remains mandatory")

    print("\nALL SOURCE-LEVEL PRECHECKS PASSED")


if __name__ == "__main__":
    main()
