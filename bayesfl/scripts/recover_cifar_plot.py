from pathlib import Path
import csv, math, yaml
from bayesfl.config import load_config

ROOT = Path.cwd().resolve()
D, N = 878_538, 20
BUDGET = 120 * 8 * D * (N + 1)
assert BUDGET == 17_711_326_080

CFG_DIR = ROOT / "scripts/configs/cifar_plot_recovery"
QUEUE_DIR = ROOT / "scripts/queues"
CFG_DIR.mkdir(parents=True, exist_ok=True)
QUEUE_DIR.mkdir(parents=True, exist_ok=True)

JOBS = {
    "fola": {
        "source": (
            "scripts/configs/confirm_newsrc_dense/"
            "run_confirm_newsrc_cifar_basiccnn_a0p01_fola_"
            "cos400_n20_f1_seed0_e10_b32_lr002_lam0p01_r150.yaml"
        ),
        "name": (
            "run_confirm_newsrc_cifar_basiccnn_a0p01_fola_"
            "cos400_n20_f1_seed0_e10_b32_lr002_lam0p01_r150"
        ),
        "cap": 150,
        "plot_round": 120,
        "gpu": 0,
    },
    "fedavg": {
        "source": (
            "scripts/configs/comm_budget_extension_seed0/"
            "run_budgetmatch_cifar_basiccnn_a0p01_fedavg_"
            "n20_f1_seed0_e10_b32_lr001_budget150densefola.yaml"
        ),
        "name": (
            "run_budgetmatch_cifar_basiccnn_a0p01_fedavg_"
            "n20_f1_seed0_e10_b32_lr001_budget150densefola"
        ),
        "cap": 240,
        "plot_round": 240,
        "gpu": 1,
    },
}

def profile(cfg):
    d = cfg.to_dict()
    d.pop("run_name", None)
    d.pop("output", None)
    d["training"].pop("rounds", None)
    d["communication"].pop("max_communication_bytes", None)
    return d

queues = {0: [], 1: []}
blocked = []

for method, job in JOBS.items():
    source = ROOT / job["source"]
    assert source.is_file(), f"Missing original config: {source}"

    raw = yaml.safe_load(source.read_text())
    assert raw["method"] == method
    assert raw["data"]["partition"] == {
        "type": "paper_dirichlet",
        "dirichlet_alpha": 0.01,
        "local_data_fraction": 1.0,
    }
    assert raw["federation"]["num_clients"] == 20
    assert raw["federation"]["clients_per_round"] == 20
    assert raw["runtime"]["seed"] == 0
    assert raw["model"]["name"] == "paper_basiccnn"
    assert raw["training"]["local_epochs"] == 10
    assert raw["training"]["batch_size"] == 32
    assert raw["training"]["grad_clip_norm"] == 10.0
    assert raw["training"]["lr"] == (
        0.02 if method == "fola" else 0.01
    )
    assert raw["training"]["lr_schedule"] == (
        "cosine" if method == "fola" else "constant"
    )

    if method == "fola":
        assert raw["training"]["lr_decay_rounds"] == 400
        assert raw["fola"]["prior_lambda"] == 0.01
        assert raw["fola"]["mode"] == "paper_reference"

    raw["run_name"] = job["name"]
    raw["training"]["rounds"] = job["cap"]
    raw["communication"]["downlink_mode"] = "multicast"
    raw["communication"]["max_communication_bytes"] = None

    dest = CFG_DIR / (job["name"] + ".yaml")
    if dest.exists():
        assert yaml.safe_load(dest.read_text()) == raw, (
            f"Config conflict: {dest}"
        )
    else:
        dest.write_text(yaml.safe_dump(raw, sort_keys=False))

    wanted = load_config(dest)
    assert wanted.communication.downlink_mode == "multicast"

    previous = [
        p for p in [
            ROOT / "outputs" / job["name"],
            *(ROOT / "outputs").glob(job["name"] + "_*"),
        ]
        if p.is_dir()
    ]

    if previous:
        valid, invalid = [], []

        for run in previous:
            try:
                cfg = load_config(run / "resolved_config.yaml")
                assert profile(cfg) == profile(wanted)

                with (
                    run / "metrics/global_metrics.csv"
                ).open(newline="") as f:
                    rows = {
                        int(float(row["round"])): row
                        for row in csv.DictReader(f)
                        if row.get("round")
                    }

                assert all(
                    r in rows
                    for r in range(1, job["plot_round"] + 1)
                )
                assert math.isclose(
                    float(
                        rows[job["plot_round"]][
                            "cumulative_all_array_bytes"
                        ]
                    ),
                    BUDGET,
                    rel_tol=0,
                    abs_tol=1,
                )
                valid.append(run)

            except Exception as exc:
                invalid.append(f"{run}: {exc}")

        if invalid:
            blocked.extend(invalid)
        elif valid:
            print(f"SKIP COMPLETE: {method} | {valid[0]}")
        else:
            blocked.append(f"Incomplete existing run: {method}")

    else:
        queues[job["gpu"]].append(
            dest.relative_to(ROOT).as_posix()
        )
        print(
            f"QUEUE GPU{job['gpu']}: "
            f"{method.upper()} R{job['cap']}"
        )

if blocked:
    raise RuntimeError(
        "Existing incomplete or incompatible runs:\n"
        + "\n".join(blocked)
    )

for gpu, jobs in queues.items():
    queue = QUEUE_DIR / f"cifar_plot_recovery_gpu{gpu}.txt"
    queue.write_text("".join(job + "\n" for job in jobs))
    print(f"GPU{gpu}: {len(jobs)} job(s) | {queue}")

print(f"Equal-byte budget: {BUDGET:,} bytes")
print("PASS")
