from pathlib import Path
import copy
import yaml
import torch

from bayesfl.config import load_config
from bayesfl.models.factory import initialize_model
from bayesfl.communication import core_round_bytes
from bayesfl.data.datasets import partition_stem

ROOT = Path.cwd().resolve()

OUT = ROOT / (
    "scripts/configs/mnist_stage15_cnn_multiseed"
)

QDIR = ROOT / "scripts/queues"

# samples/client, learning rate, source experiment stage
GROUPS = (
    (10, 0.10, 12),
    (5,  0.10, 14),
    (5,  0.03, 14),
)

SEEDS = (1, 2, 3, 4)

ROUND_CAP = {
    "fola": 150,
    "fedavg": 300,
}

SOURCE_CAP = {
    "fola": 200,
    "fedavg": 400,
}

GPU_SEEDS = {
    0: (1, 3),
    1: (2, 4),
}

MODEL = "mnist_cnn_16_32_128"
NPARAMS = 80202


def old_template(samples, lr, source_stage, method):

    lr_tag = f"{lr:.2f}".replace(".", "p")
    lam = "_lam0p01" if method == "fola" else ""

    source_dir = ROOT / (
        "scripts/configs/" + (
            "mnist_stage12_arch_fixed_s10_n20_seed0"
            if source_stage == 12
            else "mnist_stage14_arch_fixed_s5_n20_seed0"
        )
    )

    name = (
        f"debug_mnist_stage{source_stage}_cnn16_32_"
        f"fixed1c_s{samples}_n20_{method}_"
        f"seed0_e5_b{samples}_lr{lr_tag}"
        f"{lam}_r{SOURCE_CAP[method]}.yaml"
    )

    path = source_dir / name

    assert path.is_file(), (
        f"Missing completed seed-0 template: {path}"
    )

    return yaml.safe_load(path.read_text())


OUT.mkdir(parents=True, exist_ok=True)
QDIR.mkdir(parents=True, exist_ok=True)

configs = {}
paths = {}

# ============================================================
# Generate the 24 configs
# ============================================================

for samples, lr, stage in GROUPS:

    for seed in SEEDS:

        for method in ("fola", "fedavg"):

            cfg = copy.deepcopy(
                old_template(samples, lr, stage, method)
            )

            # Verify the original experiment settings.
            assert cfg["method"] == method
            assert cfg["model"]["name"] == MODEL
            assert cfg["runtime"]["seed"] == 0

            assert (
                cfg["training"]["rounds"]
                == SOURCE_CAP[method]
            )

            assert cfg["training"]["lr"] == lr
            assert cfg["training"]["batch_size"] == samples
            assert cfg["training"]["local_epochs"] == 5
            assert cfg["training"]["lr_schedule"] == "constant"

            assert cfg["data"]["partition"] == {
                "type": "fixed_labels",
                "samples_per_client": samples,
                "labels_per_client": 1,
            }

            assert cfg["federation"]["num_clients"] == 20
            assert cfg["federation"]["clients_per_round"] == 20

            assert cfg["compression"]["selection_rule"] == "dense"
            assert cfg["compression"]["keep_ratio"] == 1.0
            assert cfg["communication"]["downlink_mode"] == "multicast"

            assert cfg["fola"]["mode"] == "paper_reference"
            assert cfg["fola"]["prior_lambda"] == 0.01

            # Only change run name, seed, and round cap.
            lr_tag = f"{lr:.2f}".replace(".", "p")
            lam = "_lam0p01" if method == "fola" else ""

            cfg["run_name"] = (
                f"run_mnist_stage15_cnn16_32_"
                f"fixed1c_s{samples}_n20_{method}_"
                f"seed{seed}_e5_b{samples}_lr{lr_tag}"
                f"{lam}_r{ROUND_CAP[method]}"
            )

            cfg["runtime"]["seed"] = seed
            cfg["training"]["rounds"] = ROUND_CAP[method]

            path = OUT / f"{cfg['run_name']}.yaml"

            if path.is_file():

                assert yaml.safe_load(
                    path.read_text()
                ) == cfg, (
                    f"Existing configuration differs: {path}"
                )

            else:

                assert not list(
                    (ROOT / "outputs").glob(
                        cfg["run_name"] + "_*"
                    )
                ), (
                    f"Existing output with same name: "
                    f"{cfg['run_name']}"
                )

                path.write_text(
                    yaml.safe_dump(cfg, sort_keys=False)
                )

            loaded = load_config(path)

            assert loaded.method == method
            assert loaded.runtime.seed == seed
            assert loaded.training.rounds == ROUND_CAP[method]
            assert loaded.training.lr == lr
            assert loaded.model.name == MODEL

            key = (samples, lr, seed, method)

            configs[key] = loaded
            paths[key] = path.relative_to(ROOT).as_posix()

assert len(configs) == 24


# ============================================================
# Validate every FOLA/FedAvg pair
# ============================================================

for samples, lr, _ in GROUPS:

    for seed in SEEDS:

        f = configs[(samples, lr, seed, "fola")]
        a = configs[(samples, lr, seed, "fedavg")]

        # Same data partition for each paired comparison.
        assert partition_stem(f) == partition_stem(a)

        # Same model initialization for each paired comparison.
        mf = initialize_model(f)
        ma = initialize_model(a)

        assert sum(
            p.numel() for p in mf.parameters()
        ) == NPARAMS

        assert sum(
            p.numel() for p in ma.parameters()
        ) == NPARAMS

        sf = mf.state_dict()
        sa = ma.state_dict()

        assert sf.keys() == sa.keys()

        assert all(
            torch.equal(sf[k], sa[k])
            for k in sf
        ), f"Initialization mismatch: s{samples}, lr={lr}, seed={seed}"

        # Verify equal modeled communication cost.
        fb = ROUND_CAP["fola"] * core_round_bytes(
            NPARAMS,
            20,
            method="fola",
            downlink_mode="multicast",
        )["total_bytes"]

        ab = ROUND_CAP["fedavg"] * core_round_bytes(
            NPARAMS,
            20,
            method="fedavg",
            downlink_mode="multicast",
        )["total_bytes"]

        assert fb == ab

        print(
            f"PAIR PASS | "
            f"samples={samples:2d} | "
            f"LR={lr:.2f} | "
            f"seed={seed} | "
            f"budget={fb/1e9:.3f} GB"
        )


# ============================================================
# Verify that both learning rates share the same s5 partition
# ============================================================

for seed in SEEDS:

    assert (
        partition_stem(
            configs[(5, 0.03, seed, "fola")]
        )
        ==
        partition_stem(
            configs[(5, 0.10, seed, "fola")]
        )
    )

# ============================================================
# Split into two GPU queues
# ============================================================

for gpu, seeds in GPU_SEEDS.items():

    jobs = [
        paths[(samples, lr, seed, method)]
        for samples, lr, _ in GROUPS
        for seed in seeds
        for method in ("fola", "fedavg")
    ]

    assert len(jobs) == 12
    assert len(set(jobs)) == 12

    qpath = QDIR / (
        f"mnist_stage15_cnn_multiseed_gpu{gpu}.txt"
    )

    content = "\n".join(jobs) + "\n"

    if qpath.exists():
        assert qpath.read_text() == content, (
            f"Queue conflict: {qpath}"
        )
    else:
        qpath.write_text(content)

    print()
    print(f"GPU {gpu}: {len(jobs)} jobs")

    for job in jobs:
        print(" ", Path(job).stem)


# ============================================================
# Final integrity check
# ============================================================

all_jobs = [
    paths[(samples, lr, seed, method)]
    for seeds in GPU_SEEDS.values()
    for samples, lr, _ in GROUPS
    for seed in seeds
    for method in ("fola", "fedavg")
]

assert len(all_jobs) == len(set(all_jobs)) == 24

print()
print("=" * 65)
print("STAGE-15 CNN MULTISEED QUEUES READY")
print("=" * 65)
print("Architecture   : cnn16_32")
print("Client count   : 20")
print("Local epochs   : 5")
print("Samples/client : 5 and 10")
print("Learning rates : 0.03 and 0.10")
print("New seeds      : 1, 2, 3, 4")
print("Existing seed  : 0")
print("FOLA           : R150")
print("FedAvg         : R300")
print("GPU0 jobs      : 12")
print("GPU1 jobs      : 12")
print("Total new jobs : 24")
print("PASS")
