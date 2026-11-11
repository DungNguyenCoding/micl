# Wireless FedKLS — pre-implementation review

**Status: design reviewed; production wireless implementation not yet added.**

This directory is an additive planning package. It does not change `src/`, existing
YAMLs, queues, partition files, results, plotting scripts, or dependency versions.
Do not describe it as a completed wireless feature or launch a wireless YAML using
the old loader: the inspected loader silently ignores an unknown top-level
`wireless` section.

## Scope and reading order

1. [DESIGN_REVIEW.md](DESIGN_REVIEW.md): source-derived model, mathematical checks,
   baseline mismatches, and explicitly proposed implementation resolutions.
2. [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md): module boundaries,
   round protocol, configuration/resume contract, experiment gates and documentation work.
3. [VALIDATION_PLAN.md](VALIDATION_PLAN.md): checks actually run and required gates.
4. [DECISIONS_AND_OPEN_ITEMS.md](DECISIONS_AND_OPEN_ITEMS.md): assumptions inherited
   from the proposal versus recommendations not yet approved as experimental locks.
5. [SOURCE_AUDIT.md](SOURCE_AUDIT.md): exact archive/member/line locators.

The intended first extension is **synchronous, reliable orthogonal digital upload,
all clients participating, fixed PSD, current-round KL-aware integer retention and
bandwidth allocation**. It is not AirComp, client scheduling, a variance-wire
migration, or a new local learning algorithm.

## Evidence and limits

The audited inputs are `baseline_src.zip` and `FedKLS_Wireless_Develop.zip` supplied
in the project. Their SHA-256 digests and inspected source digests are recorded in
[evidence/source_manifest.json](evidence/source_manifest.json). All 11 proposal
members are reproduced as numbered text in
[evidence/proposal_sources.txt](evidence/proposal_sources.txt).

The uploaded baseline's available CPU tests yielded **128 passed, 5 skipped**;
Flower, Ray, Bayesian-Torch and CUDA execution were not validated. All 24 primary
endpoint/byte checks passed against the actual CSVs inside the uploaded archive.
These are checks of this uploaded snapshot, not a connection to the user's live server.
See [VALIDATION_PLAN.md](VALIDATION_PLAN.md).

## Current non-wireless paper workflow is protected

The existing `scripts/plot_ieee_multicast_results_mnist_s10_stage22.py`,
`plots/Acc_Com.*`, `plots/Acc_KeepRatio.*`, and their registered 24 trajectories remain
unchanged. The locked baseline is MNIST s10/N20/seed0 and CIFAR alpha=.01/f1/N20/seed0,
with dense FOLA R150 / FedAvg R300 multicast-byte endpoints. Wireless time-budget
comparisons are new experiments and do not inherit that 2:1 round relation.

No wireless runtime command is provided here because that feature does not exist
in the audited baseline. The next production unit is an opt-in protocol/allocator
core, followed by persistence/RPC validation, not a long GPU queue.

## New-chat handoff

Attach the current source snapshot plus this directory. Ask the new assistant to
read the main `docs/CHAT_HANDOFF.md`, this README, the design review and open items.
The main handoff has historical statements that this review supersedes for the
audited snapshot: current code is now supplied, the CIFAR R150 numbers are now read,
and the uploaded `.gitignore` contains the discussed archive/checkpoint rules.
Git tracked-file status is still unknown.

The current exporter includes this directory's Markdown, TXT and JSON files. It
does not automatically include new wireless run outputs, CSVs or JSONL traces.
Updating the exporter and its README is an explicit production milestone, not an
assumed consequence of adding `src/bayesfl/wireless/`.
