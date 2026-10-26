# Sparse validation — historical results and current obligations

The original sparse-delivery checks are preserved at [history/VALIDATION_SPARSE.md](docs/history/pre_refresh_20261026/VALIDATION_SPARSE.md). They reported baseline 23 passed/4 skipped and modified 110 passed/5 skip reports in a CPU authoring environment, plus synthetic protocol/accounting/resume checks. They explicitly did not exercise the full target CUDA/Flower/Ray stack.

Those counts are historical evidence, not a fresh result for the later CNN, strict-FedAvg, multicast and cleanup state. User-supplied dataset trajectories establish that specific experiments ran; they do not replace regression tests or prove a universally faithful paper reproduction.

After an actual implementation change, inspect current tests and check: KL direction and numerical policy; exact-cardinality random/top-score selection; bitmap ordering/padding; packet identity/round/layout checks; no hidden local state; omission reconstruction; all-omitted exact preservation; dense/full-retention numerical consistency; multicast byte limits; checkpoint/ledger/partition consistency; and resume versus uninterrupted behavior. Preserve existing regression fixtures rather than silently updating expected results to make tests pass.

A source change that affects clipping, precision, initialization, aggregation, mask RNG or schedule requires explicit experiment provenance. Do not reuse a published accuracy as if it came from the modified code. Current runbook and evidence limits are in [docs/RUNBOOK.md](docs/RUNBOOK.md) and [docs/EVIDENCE_AND_GAPS.md](docs/EVIDENCE_AND_GAPS.md).
