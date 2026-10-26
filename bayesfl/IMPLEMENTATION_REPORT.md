# Implementation status — current documentation summary

Documentation refreshed 2026-10-26. This update changes documentation and adds documentation-only check/export tooling; it does not change the FedKLS trainer, optimizer, dataset partitioning, packet protocol, communication implementation or plotter.

The original implementation report is preserved verbatim at [docs/history/pre_refresh_20261026/IMPLEMENTATION_REPORT.md](docs/history/pre_refresh_20261026/IMPLEMENTATION_REPORT.md). Its “trainers unchanged”, “398 original YAMLs”, unicast formulas and “no dataset runs performed here” describe that earlier delivery, not the current project state.

Current documented functionality includes full-local-training sparse posterior communication, model-wide KL/global-first and random selectors, mean–precision bitmap packets, broadcast-value reconstruction, dense multicast accounting, equal-byte comparisons, and completed-round resume. Later work added MNIST CNN variants, strict/unclipped FedAvg controls, clipping ablations and final sparse simulations. Source paths: [REPOSITORY_MAP](docs/REPOSITORY_MAP.md); exact runs: [RUN_REGISTRY](docs/RUN_REGISTRY.md).

Current source bodies for later patches and raw current CSVs are not contained in this documentation refresh. Inspect them before modifying algorithms. [EVIDENCE_AND_GAPS](docs/EVIDENCE_AND_GAPS.md) states which details were supplied, inspected historically, reported by the user or still need live verification. No fresh full-source or full-GPU correctness certification is claimed.
