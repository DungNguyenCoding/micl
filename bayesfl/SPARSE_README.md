# FedKLS sparse communication — current entry point

The existing fixed-ratio KL(global||local)/random benchmark and its paper figures remain
unchanged. Read [docs/LOCKED_EXPERIMENTS.md](docs/LOCKED_EXPERIMENTS.md),
[docs/RUN_REGISTRY.md](docs/RUN_REGISTRY.md) and
[docs/COMMUNICATION_ACCOUNTING.md](docs/COMMUNICATION_ACCOUNTING.md).

The new **opt-in wireless** path is described in [docs/wireless/README.md](docs/wireless/README.md).
It selects current-round integer counts and bandwidth without changing local FOLA
training. Its dense-endpoint packet rule and time/resume controller are separate from
legacy sparse v1. Do not apply its count/byte rules to old runs.

Historical complete sparse-readme versions remain in docs/history. The current code
stores raw precision, not variance; zero precision flooring is score-only when enabled.
Neither sparsity mode implies smaller local models or guaranteed accuracy gains.
