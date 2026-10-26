# Documentation change log

## 2026-10-26 — current-state / new-chat handoff refresh

Replaced misleading current-entry prose in README/SPARSE_README/implementation/validation pages. Reorganized docs into current locks, command rules, runbook, source/script/run maps, numerical accounting, plotting/results, Git/data portability, cleanup state, evidence gaps and a self-contained new-chat handoff. Familiar old doc paths now link to the current contract and immutable originals.

Recorded selected Stage18 unclipped MNIST FOLA, Stage23 recovered clipped FedAvg, Stage22 s10 sparse runs and CIFAR alpha.01/f1, all at R150 budgets. Corrected stale unicast, MLP/N100, ResNet/half-data and R120 defaults. Preserved the recovered88.23 versus historical88.12 distinction. Kept absent CIFAR R150 accuracies null. Recorded the MNIST498 versus CIFAR499 sparse target distinction.

Preserved original uploaded documentation bytes and relevant audit evidence. Added documentation-only `docs/tools/export_handoff.py` for terminal endpoint checks and lightweight source/metrics export. Added an external preview/apply/verify installer with live-document backups and source/dependency non-modification checks.

No changes to `src/`, existing `scripts/`, tests, simulation YAMLs, queues, experiment metrics, plotter, partitions, .gitignore, pyproject.toml or requirements.txt. Original CHANGE_MANIFEST.json/PACKAGE_MANIFEST.txt remain historical and unchanged. No new training, source-audit certification, wireless implementation or new scientific result is claimed.
