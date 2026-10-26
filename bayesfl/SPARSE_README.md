# FedKLS sparse posterior communication — current entry point

Updated 2026-10-26. The original sparse-release README described an earlier MLP/N100 and CIFAR alpha=.1/half-data/unicast study. It is retained unchanged in [history](docs/history/pre_refresh_20261026/SPARSE_README.md), **not** the current experiment lock.

Current study: full local FOLA training; model-wide `KL(global || local)` or uniform without-replacement selection; exact `ceil(r*d)` selected float32 mean–precision pairs and packed bitmap; old-global reconstruction; dense multicast downlink. No subnet pruning or local compute saving is implied.

Read [START_HERE](docs/START_HERE.md), [locked settings](docs/LOCKED_EXPERIMENTS.md), [algorithm contract](docs/ALGORITHM_NOTES.md), [communication budget](docs/COMMUNICATION_ACCOUNTING.md), and [runbook](docs/RUNBOOK.md). Current primary runs are MNIST s10/seed0 and CIFAR alpha=.01/f1/seed0, both at dense FOLA R150 budgets.

Do not run the old `scripts/configs/bayesian_sparse/...` defaults as a substitute for Stage-22 or final CIFAR configs. Do not apply old test counts or source-preservation claims to the later live source without new validation.
