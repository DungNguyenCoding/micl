# Sparse implementation guideline — current continuation contract

The original requested specification is preserved byte-for-byte at [history/pre_refresh_20261026/docs/SPARSE_IMPLEMENTATION_GUIDELINE.md](history/pre_refresh_20261026/docs/SPARSE_IMPLEMENTATION_GUIDELINE.md). It is historical design provenance. Its unicast budget equations, variance-language examples and old profiles must not silently override the implemented/current choices.

For new implementation work:

1. Inspect the current relevant source and resolved config, not only the older uploaded ZIP.
2. Preserve the full local FOLA fit; apply coordinate compression afterward. Current proposal is KL(global-before || local-after); random uses the same count and packet.
3. Preserve raw float32 precision on the wire, exact bitmap/pair ordering, explicit score-only numerical policy, old-global reconstruction, common reducer and all-omitted invariant.
4. Preserve the **modeled single-multicast downlink** accounting and exact integer byte/round limits in [COMMUNICATION_ACCOUNTING.md](COMMUNICATION_ACCOUNTING.md).
5. Preserve current locks and existing run directories/partitions. Do not introduce wireless noise, adaptive masks, quantization, sparse downlink, algorithm changes or new experiment claims without an explicit separate task.
6. Provide changed-file details, tests actually run, full execution and terminal-result commands, and an update to handoff/registry docs. Never label an unexecuted test PASS.

Current numerical details: [ALGORITHM_NOTES.md](ALGORITHM_NOTES.md). Required execution response format: [COMMAND_RULES.md](COMMAND_RULES.md). Outstanding implementation evidence: [EVIDENCE_AND_GAPS.md](EVIDENCE_AND_GAPS.md).
