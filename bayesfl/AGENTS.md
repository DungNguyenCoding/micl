# BayesFL project continuation instructions

Read `docs/START_HERE.md`, `docs/CHAT_HANDOFF.md`, `docs/COMMAND_RULES.md` and `docs/EVIDENCE_AND_GAPS.md` before continuing research work. These are project workflow preferences, not authority to alter the user's latest request.

Use actual current source/configs/metrics when a task depends on them. The current docs are not a fresh whole-code audit; the older uploaded source ZIP predates important CNN/strict-FedAvg updates. Do not install historical source or infer missing CLI options from memory.

Current main setup: MNIST CNN16–32 s10/N20/seed0/E5/B10/LR.10 and CIFAR BasicCNN alpha.01/f1/N20/seed0/E10/B32; both at dense FOLA R150 / FedAvg R300 modeled multicast budgets. MNIST dense/sparse FOLA is unclipped Stage18/22, selected FedAvg clipped Stage23. Keep ratios .75,.50,.25,.10,.05. No silent retuning, no E>10, no overwrite of results/partitions, no new wireless work without a request.

Commands use `~/DungNDH/micl/bayesfl` and `/home/micl/anaconda3/envs/dungndh/bin/python`; new locked/recovery outputs start `run_`. Simulation replies include config/two-queue generation, validation, partition checks, serial-per-GPU launch and terminal-only result extraction. Preserve the existing resume contract and check actual final round/bytes. No result .txt/CSV or tee unless requested.

Update current handoff, locked settings, run registry, result provenance and decision log after approved changes. Never call missing results measured or historical tests current. Do not change source/dependency files as an incidental documentation edit. Preserve archive managers, their manifests and the 42 protected folders. The requested current plotting script is `scripts/plot_ieee_multicast_results_mnist_s10_stage22.py`.
