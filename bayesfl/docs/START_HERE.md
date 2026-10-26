# Start here — current BayesFL/FedKLS state

**As of 2026-10-26.** This documentation supersedes older experiment defaults while preserving the supplied originals unchanged under [history](history/README.md).

For a new chat, first read [CHAT_HANDOFF.md](CHAT_HANDOFF.md). Before changing code or writing execution commands, also read [COMMAND_RULES.md](COMMAND_RULES.md), [LOCKED_EXPERIMENTS.md](LOCKED_EXPERIMENTS.md) and [EVIDENCE_AND_GAPS.md](EVIDENCE_AND_GAPS.md). The [full index](README.md) gives the remaining module/run/result maps.

## Current choice

MNIST CNN16–32: **s10, N20, seed0, unclipped dense/sparse FOLA, clipped recovered FedAvg**. CIFAR-10 BasicCNN: **alpha .01, full local-data fraction, N20, seed0**. Both panels use **dense FOLA R150 / FedAvg R300** with modeled multicast budgets. Do not silently revert to old MLP/N100, ResNet, CIFAR half-data or R120.

Primary MNIST dense is Stage18, not Stage24. Primary sparse is Stage22, not Stage21 s5. Current figures are Acc_Com and Acc_KeepRatio under `plots/`. The user reports plotting normally after cleanup. No new simulations are needed simply to install or use these docs.

## What “current” means here

This bundle reconciles the latest user decisions with uploaded documentation and recorded runtime/cleanup evidence. It is not a freshly audited full working repository. The exact current source and raw metric rows should be captured/read for code changes and for quoting the still-unsupplied CIFAR R150 accuracies. Existing snapshots and inferred facts are labeled, not silently promoted to measurements.

`CHANGE_MANIFEST.json` and `PACKAGE_MANIFEST.txt` at root remain historical release artifacts. `pyproject.toml` and `requirements.txt` remain byte-unchanged. `DOCS_UPDATE_MANIFEST.json` describes only this documentation refresh. No `.gitignore` change is applied.

## Minimal continuation message

> Continue this BayesFL/FedKLS project using the attached current CHAT_HANDOFF.md and documentation bundle. Read the command rules, locked profiles, registry and evidence limits first. Preserve final experiments and archive state. Do not retune, rerun completed jobs or modify source until my requested task requires it. Use current code/config/metric files rather than historical source snapshots when implementing changes.

Attach the lightweight live export as well for a code/metrics task. Docs alone cannot contain the entire latest source or guarantee access to another chat's attachments.
