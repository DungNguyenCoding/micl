from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


MANIFEST = Path(
    "scripts/configs/"
    "comm_budget_extension_seed0/"
    "manifest.json"
)

Q0 = Path(
    "scripts/queues/"
    "comm_budget_extension_seed0_gpu0.txt"
)

Q1 = Path(
    "scripts/queues/"
    "comm_budget_extension_seed0_gpu1.txt"
)

ALLOWED = {
    0.05,
    0.02,
    0.01,
}


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

    if not any(
        math.isclose(
            keep,
            x,
        )
        for x in ALLOWED
    ):
        raise ValueError(
            "Allowed keep ratios: "
            "0.05, 0.02, 0.01"
        )

    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Missing manifest: {MANIFEST}"
        )

    m = json.loads(
        MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    jobs = [
        job
        for job in m["sparse_jobs"]
        if math.isclose(
            float(
                job["keep_ratio"]
            ),
            keep,
        )
    ]

    assert len(jobs) == 9, (
        f"Expected 9 jobs for keep={keep}, "
        f"found {len(jobs)}"
    )

    assert len({
        job["path"]
        for job in jobs
    }) == 9

    assert {
        job["profile"]
        for job in jobs
    } == {
        "cifar_f1",
        "cifar_f0p5",
        "mnist_fixed",
    }

    assert {
        job["rule"]
        for job in jobs
    } == {
        "kl_global_local",
        "kl_local_global",
        "random",
    }

    # ============================================================
    # Greedy compute balancing between exactly TWO GPUs.
    # ============================================================
    bins = [
        {
            "load": 0.0,
            "jobs": [],
        },
        {
            "load": 0.0,
            "jobs": [],
        },
    ]

    jobs = sorted(
        jobs,
        key=lambda x: float(
            x.get(
                "work",
                x["expected_rounds"],
            )
        ),
        reverse=True,
    )

    for job in jobs:

        target = min(
            bins,
            key=lambda b: b["load"],
        )

        target["jobs"].append(
            job
        )

        target["load"] += float(
            job.get(
                "work",
                job["expected_rounds"],
            )
        )

    gpu0 = bins[0]["jobs"]
    gpu1 = bins[1]["jobs"]

    assert (
        len(gpu0)
        + len(gpu1)
        == 9
    )

    assert len({
        job["path"]
        for job in (
            gpu0 + gpu1
        )
    }) == 9

    Q0.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Q0.write_text(
        "\n".join(
            job["path"]
            for job in gpu0
        )
        + "\n",
        encoding="utf-8",
    )

    Q1.write_text(
        "\n".join(
            job["path"]
            for job in gpu1
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 84)
    print(
        f"LOW-KEEP WAVE PREPARED: keep={keep:.2f}"
    )
    print("=" * 84)

    for gpu, group in (
        (0, gpu0),
        (1, gpu1),
    ):

        print()
        print(
            f"GPU{gpu}: "
            f"{len(group)} jobs"
        )

        for i, job in enumerate(
            group,
            start=1,
        ):
            print(
                f"  {i:2d}. "
                f"{job['profile']:12s} | "
                f"{job['rule']:20s} | "
                f"keep={job['keep_ratio']:.2f} | "
                f"expected R={job['expected_rounds']}"
            )

    print()
    print(
        f"TOTAL = {len(gpu0) + len(gpu1)} jobs"
    )

    print(
        f"Queue 0: {Q0}"
    )

    print(
        f"Queue 1: {Q1}"
    )


if __name__ == "__main__":
    main()
