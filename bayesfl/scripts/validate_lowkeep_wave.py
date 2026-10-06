from __future__ import annotations

import argparse
import math
from pathlib import Path
from collections import Counter

from bayesfl.config import load_config


Q0 = Path(
    "scripts/queues/"
    "comm_budget_extension_seed0_gpu0.txt"
)

Q1 = Path(
    "scripts/queues/"
    "comm_budget_extension_seed0_gpu1.txt"
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--keep",
        type=float,
        required=True,
    )

    args = parser.parse_args()

    keep = float(
        args.keep
    )

    paths = []

    for q in (
        Q0,
        Q1,
    ):
        assert q.exists(), (
            f"Missing queue: {q}"
        )

        entries = [
            Path(x.strip())
            for x in q.read_text(
                encoding="utf-8"
            ).splitlines()
            if x.strip()
        ]

        paths.extend(
            entries
        )

    assert len(paths) == 9, (
        f"Expected 9 jobs, found {len(paths)}"
    )

    assert len(
        set(paths)
    ) == 9

    profiles = Counter()
    rules = Counter()

    for path in paths:

        assert path.exists(), (
            f"Missing config: {path}"
        )

        cfg = load_config(
            path
        )

        assert cfg.method == "fola"
        assert cfg.sparse_enabled

        assert math.isclose(
            cfg.compression.keep_ratio,
            keep,
        )

        assert (
            cfg.compression.selection_rule
            in {
                "kl_global_local",
                "kl_local_global",
                "random",
            }
        )

        assert (
            cfg.training.local_epochs
            == 10
        )

        assert (
            cfg.training.local_epochs
            <= 10
        )

        assert (
            cfg.training.batch_size
            == 32
        )

        assert (
            cfg.training.rounds
            == 1000
        )

        assert (
            cfg.fola.mode
            == "paper_reference"
        )

        assert (
            cfg.fola.lambda_scale_by_size
            is False
        )

        assert (
            cfg.fola.paper_mean_only_eval
            is True
        )

        assert (
            cfg.communication.max_communication_bytes
            is not None
        )

        assert (
            cfg.communication.budget_metric
            == "cumulative_all_array_bytes"
        )

        if (
            cfg.data.dataset
            == "mnist"
        ):
            profile = (
                "mnist_fixed"
            )

            assert (
                cfg.data.partition["type"]
                == "fixed_labels"
            )

            assert (
                cfg.data.partition[
                    "samples_per_client"
                ]
                == 10
            )

            assert (
                cfg.data.partition[
                    "labels_per_client"
                ]
                == 1
            )

        else:

            fraction = float(
                cfg.data.partition[
                    "local_data_fraction"
                ]
            )

            profile = (
                "cifar_f1"
                if math.isclose(
                    fraction,
                    1.0,
                )
                else "cifar_f0p5"
            )

            assert math.isclose(
                float(
                    cfg.data.partition[
                        "dirichlet_alpha"
                    ]
                ),
                0.01,
            )

        profiles[
            profile
        ] += 1

        rules[
            cfg.compression.selection_rule
        ] += 1

        print(
            f"OK | "
            f"{profile:12s} | "
            f"{cfg.compression.selection_rule:20s} | "
            f"keep={cfg.compression.keep_ratio:.2f} | "
            f"budget="
            f"{cfg.communication.max_communication_bytes/1e9:.6f} GB"
        )

    assert profiles == Counter({
        "cifar_f1": 3,
        "cifar_f0p5": 3,
        "mnist_fixed": 3,
    })

    assert rules == Counter({
        "kl_global_local": 3,
        "kl_local_global": 3,
        "random": 3,
    })

    print()
    print(
        f"PASS: keep={keep:.2f} wave contains exactly 9 valid runs"
    )


if __name__ == "__main__":
    main()
