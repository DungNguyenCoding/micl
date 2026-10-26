# Decisions and next-work boundary

## Current state

The user considers simulation results sufficient for the paper. The final chosen plotting setup is MNIST CNN s10, N20, seed0 and CIFAR BasicCNN alpha=.01, full local-data fraction, seed0; both have dense FOLA R150 budgets. Preserve the unclipped Stage-18 MNIST dense baseline. Stage-23 recovered clipped FedAvg is selected for the primary MNIST curve. Stage-24 clipped FOLA is not selected.

The pipeline is full local FOLA training followed by KL(global || local) or equal-cardinality random sparse upload. Five tested keep ratios are .75,.50,.25,.10,.05. Dense downlink remains modeled multicast. All ten CIFAR R150 sparse targets were already complete; empty continuation queues were therefore correct. No new simulations were launched by the status check.

## Relevant history to avoid repeating

Earlier MLP and CNN architecture/LR sweeps, clipping ablations and five-seed overviews informed the current selected conditions. They are not default requests to retune now. N100 work was stopped with two resumable FOLA runs retained. Stage 2 archived 121 folders, Stage 3 built the registry, and Stage 3B archived 13 historical script/queue files (user-reported complete). Main plots subsequently worked normally. Do not undo cleanup or recreate historical experiments unless asked.

MNIST s5/seed3 remains a preserved alternative, not the current primary panel. R120 figures remain an alternative budget cut, not the current final choice. CIFAR alpha .1, half-data, ResNet56 and reverse KL are distinct historical studies.

## Immediate continuation after documentation installation

Verify the documentation installation, then optionally run the bundled primary-endpoint check/export to capture current CIFAR R150 results, exact live configs and source for the next chat. Do not run a new training queue simply to produce a handoff.

If the next request is paper writing, use the latest exact result tables and disclose clipping/LR differences and single-seed selection. If the next request is code modification, inspect the current source snapshot first. The docs are navigation and decisions, not an executable replacement for missing code.

## Proposed research direction, not yet implemented

The project discussed a wireless environment with over-the-air uplink aggregation and different client keep ratios based on wireless conditions. This remains prospective. Do not label current modeled multicast payload savings as a wireless simulation. A future design must explicitly define channel/noise/power assumptions, coordinate alignment/masks, mean–precision aggregation compatibility, and a new resource metric (e.g. symbols/time/energy). Preserve the finished noiseless/discrete-payload FedKLS benchmark as a separate baseline.

The project contains source papers 00–07 and the sparse survey. A future technical comparison should read the actual relevant sources, especially 06 and 07, rather than treating this handoff as a new paper review. No new theorem, wireless implementation, convergence guarantee or optimal keep-ratio rule is asserted here.

## Update discipline

After each approved change, record its date, changed files, commands, tests actually run, new exact run paths/checkpoints and any changed lock. Update `EXPERIMENT_REGISTRY.json`, `LOCKED_EXPERIMENTS.md`, `RESULTS_AND_PLOTTING.md` and `CHAT_HANDOFF.md` together when applicable. Never rewrite historical numbers without labeling why a source/run changed. Keep proposed work distinct from completed work.
