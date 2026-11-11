# Start here — two workflows, one preserved learner

**Current source release: wireless v1, 2026-11-09.** The earlier documentation-only
snapshot is preserved in `history/pre_wireless_v1/`; older historical docs remain in
`history/pre_refresh_20261026/`.

1. Read [CHAT_HANDOFF.md](CHAT_HANDOFF.md) for continuity and exact legacy output paths.
2. Read [COMMAND_RULES.md](COMMAND_RULES.md) and [LOCKED_EXPERIMENTS.md](LOCKED_EXPERIMENTS.md)
   before executing or changing a study.
3. Read [wireless/README.md](wireless/README.md), [wireless/STATUS.md](wireless/STATUS.md)
   and [wireless/VALIDATION.md](wireless/VALIDATION.md) for implementation and validation scope.

The current primary figures still use MNIST CNN s10/N20/seed0 and CIFAR BasicCNN
alpha=.01/f1/N20/seed0. Both use legacy dense FOLA R150 budgets and FedAvg R300.
The new opt-in wireless controller does not replace these finished experiments.

The supplied current `baseline_src.zip` was inspected and extended. The source audit,
legacy configuration-serialization comparison and primary CSV checks replace earlier
statements that only an old MLP source snapshot was available. This does not certify
runtime behavior on the user's GPU server. Physical wireless scenario values remain
explicit experimental choices; an example YAML is not an approved paper setting.

For a new chat, export the current tree. Default export includes the 24 primary legacy
runs; wireless runs require explicit `--wireless-run-dir` flags. See
[tools/README.md](tools/README.md). The exporter copies decisions already written in
docs; it cannot learn decisions from chat history automatically.
