from pathlib import Path
import copy
import yaml


ROOT = Path(".").resolve()
OUT = ROOT / "scripts/configs/mnist_dense_multicast_seed0"
OUT.mkdir(parents=True, exist_ok=True)

OUTPUTS = ROOT / "outputs"

BUDGET = 45_251_038_080

OLD_FOLA_PREFIX = (
    "run_confirm_newsrc_"
    "mnist_fixed1c_s10_fola_n100_seed0_"
    "e10_b32_lr001_lam0p01_r150"
)

OLD_FEDAVG_PREFIX = (
    "run_confirm_newsrc_"
    "mnist_fixed1c_s10_fedavg_n100_seed0_"
    "e10_b32_lr001_r150"
)

NEW_FOLA_NAME = (
    "run_multicast_mnist_fixed1c_s10_"
    "fola_mlp256x5_n100_seed0_"
    "e10_b32_lr001_lam0p01_r120"
)

NEW_FEDAVG_NAME = (
    "run_multicast_mnist_fixed1c_s10_"
    "fedavg_mlp256x5_n100_seed0_"
    "e10_b32_lr001_r240_budget120densefola"
)


def resolve_old(prefix: str) -> Path:
    matches = [
        p for p in OUTPUTS.glob(prefix + "*")
        if p.is_dir()
    ]

    if not matches:
        raise RuntimeError(
            f"No previous confirmed run found: {prefix}"
        )

    return max(
        matches,
        key=lambda p: p.stat().st_mtime,
    )


def read_yaml(path: Path):
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, obj):
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            obj,
            f,
            sort_keys=False,
        )


old_fola_dir = resolve_old(OLD_FOLA_PREFIX)
old_fedavg_dir = resolve_old(OLD_FEDAVG_PREFIX)

print("Source FOLA  :", old_fola_dir)
print("Source FedAvg:", old_fedavg_dir)

fola = read_yaml(
    old_fola_dir / "resolved_config.yaml"
)

fedavg = read_yaml(
    old_fedavg_dir / "resolved_config.yaml"
)


# ================================================================
# Common new settings
# ================================================================

for cfg in (fola, fedavg):

    cfg["model"]["name"] = "mlp_784_256x5_10"

    cfg.setdefault(
        "communication",
        {},
    )

    cfg["communication"]["downlink_mode"] = "multicast"
    cfg["communication"]["budget_metric"] = (
        "cumulative_all_array_bytes"
    )

    # Full participation means client order does not alter the set of
    # participants, but enabling this also makes future resume possible.
    cfg["communication"][
        "deterministic_client_schedule"
    ] = True

    cfg.setdefault(
        "compression",
        {},
    )

    cfg["compression"]["selection_rule"] = "dense"
    cfg["compression"]["keep_ratio"] = 1.0


# ================================================================
# Dense FOLA R120
# ================================================================

fola["run_name"] = NEW_FOLA_NAME
fola["training"]["rounds"] = 120
fola["communication"]["max_communication_bytes"] = BUDGET


# ================================================================
# Dense FedAvg R240
#
# FedAvg multicast round payload is exactly half dense FOLA:
#
#     4*d*(K+1)
#
# Thus 240 FedAvg rounds == 120 dense-FOLA rounds in bytes.
# ================================================================

fedavg["run_name"] = NEW_FEDAVG_NAME
fedavg["training"]["rounds"] = 240
fedavg["communication"]["max_communication_bytes"] = BUDGET


fola_path = OUT / f"{NEW_FOLA_NAME}.yaml"
fedavg_path = OUT / f"{NEW_FEDAVG_NAME}.yaml"

write_yaml(
    fola_path,
    fola,
)

write_yaml(
    fedavg_path,
    fedavg,
)

print()
print("Generated:")
print(" ", fola_path)
print(" ", fedavg_path)

print()
print("Common modeled multicast budget:")
print(f"  {BUDGET:,} bytes")
print(f"  {BUDGET / 1e9:.9f} GB")

print()
print("Preserved MNIST partition definition:")
print(
    yaml.safe_dump(
        fola["data"].get("partition", {}),
        sort_keys=False,
    )
)
