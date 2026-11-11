# FedKLS digital wireless extension — v1 implementation report

Built 2026-11-09 from the supplied current `baseline_src.zip`, with design basis
`FedKLS_Wireless_Develop.zip`. This is executable source plus corresponding documentation,
not another design-only package. The full source snapshot retains the uploaded original
files; a separate verified overlay installs only changed/new source, docs, helpers/tests
and validation evidence, never experimental outputs or datasets.

## Delivered

* Strict opt-in config/main routing preserving disabled legacy serialization/fingerprints.
* Reliable digital orthogonal fixed-PSD channels, fixed/seeded Rayleigh/explicit trace
  modes, slowest-client multicast, real-partition preparation barrier, explicit control
  reservation, modeled time and data-plane energy distinct from emulator walltime.
* Integer-count local score oracle and Method1 price coordinator, dense-format exception,
  padded packet cost, feasible incumbent and numerical gap diagnostics; no universal
  integer-optimality claim or fixed-ratio action menu.
* Client-owned PREPARE/cache/scalar-query/UPLOAD/RELEASE lifecycle, local emulator and
  actual Flower/Ray adapters using Context.state across ephemeral client objects.
* Exact-count packet/grant validation, original prior fill/aggregation/epsilon behavior,
  dynamic actual-byte and modeled resource logs, strict evaluated-round resume and locks.
* Joint KL/random, equal bandwidth, uniform count, dense FOLA and dense FedAvg policies.
* New-config/two-queue generation from the saved primary profiles, terminal summaries,
  separate wireless plots, synthetic CPU/Flower smoke entry points and optional real
  adapter tests; no automatic long queue.
* Updated README/AGENTS/handoff, source/script/settings/runbook/resource/result/status
  documentation and explicit wireless-run support in the lightweight exporter.

## Preserved

Original local trainers, model definitions, partition code, legacy packets/accounting/
resume/reducer, selected figure script, old YAMLs/queues, dependencies, `.gitignore`,
protected output files and cleanup state are not rewritten. Existing source edits are
`config.py`, `main.py`, `server.py` and `docs/tools/export_handoff.py`; isolated new
wireless code carries the new behavior. Old manifest files remain historical.

Mean/precision representation is retained (not the proposal's mean/variance notation).
Byte-padded bitmaps and reserved canonical control costs are explicit implementation
adaptations. Successful messages have no injected weight noise. No AirComp, drop-client
scheduler, sparse local training, fixed-total-power solver, retransmission controller,
mid-round recovery or simultaneous legacy strict-byte cap is implemented.

## Observed checks / runtime boundary

**186 passed, 6 skipped** in the available CPU suite. **891 original YAMLs** retain
identical serialized configuration and legacy fingerprint. **All24 primary endpoints**
remain valid. Six synthetic policy cases run; local clean-resume equivalence, real
MNISTCNN/CentralEvaluator on synthetic images, generator preservation and dynamic export
checks pass. Details and exact logs: `docs/wireless/VALIDATION.md`, `validation/wireless/`.

Flower/Ray/Bayesian-Torch/CUDA were absent. Actual pinned Flower/Ray and CUDA dataset
runtime remain validation gates, not claimed successful tests. No new dataset wireless
accuracy result is fabricated. Example physical values are not locked paper settings.

## Entry points

Read `docs/wireless/README.md`, `CONFIGURATION.md`, `RUNBOOK.md`, `STATUS.md` and the
current `docs/CHAT_HANDOFF.md`. The installer manifest lists exact changed paths,
payload hashes and accepted previous versions; it refuses unreviewed local conflicts.
Stop training and source writers before preview/apply. Keep the generated backup/receipt.
The full source archive is for a separate extraction/reference, not a force-overwrite
of a live tree containing outputs.
