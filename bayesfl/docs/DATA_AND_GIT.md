# Artifacts for plots, new training, resume and handoff

Code/configs/docs/tests and selected raw metrics are enough for source inspection and
plots when their dependencies/expected paths are present. Training needs datasets and
partition indices; **resuming** additionally needs intact global state, partition content,
configuration identity and communication/event history. A model weights file alone is
not sufficient evidence for a clean resume.

The wireless release does not modify `.gitignore`, dependency declarations, tracked-file
status, Git history, existing datasets/partitions or cleanup archives. Do not globally
ignore .npz without considering partition indices and small test fixtures. Keep any
large-file backup separate from source version control decisions.

Legacy v1 resume needs its original resume/communication artifacts. Wireless v2 needs
`wireless_resume/`, `wireless/events.jsonl`, exact tracked metric files, pinned trace (if
used) and partition/config identities. Preserve these on the research machine. Do not
truncate failed ledgers or discard mismatches to force a continuation.

Default `docs/tools/export_handoff.py --export` includes code/docs/configs/tests and
24 primary legacy trajectories. New wireless outputs require explicit repeated
`--wireless-run-dir` flags. `--wireless-only` bypasses the legacy-result requirement,
not the current source-tree requirement. New exports include dynamic client/round metrics
and small protocol/config/trace metadata, not checkpoint arrays or full ledgers.
They support review/plots, not training resumption or complete disaster recovery.

Refresh docs with new decisions before exporting; the tool does not read conversation
memory. Reuse an old ZIP only when relevant project state has not changed. A sibling
same-SSD archive does not protect against SSD failure. Review all exported source,
configuration paths and metadata before sharing externally.
