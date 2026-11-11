# Evidence and remaining gaps — wireless v1, 2026-11-09

## Evidence precedence

For results: raw CSV at the exact run and its saved configuration. For intended future
work: the user's latest explicit decision. Resolve disagreements explicitly rather than
silently rewriting an experiment or promoting old reports to current measurements.

This release used the actual uploaded `baseline_src.zip` with later CNN/clipping/resume
code and `FedKLS_Wireless_Develop.zip` with the proposed method. Their SHA-256 identities
are recorded in the wireless change manifest. Current resolved primary configurations
and all 24 raw endpoints were inspected/checked. Earlier docs saying only an old MLP ZIP
was available or that CIFAR R150 results were unavailable are superseded.

## What was actually checked

Read `wireless/VALIDATION.md` and `validation/wireless/` at repository root. Executed
checks include available baseline/new unit tests, local synthetic trainer integration,
real MNIST CNN/CentralEvaluator on synthetic images, clean resume, dynamic reporting and
lightweight exports, integer/codec guards, 891 original YAML serialization/fingerprint
comparisons, and all24 existing primary byte totals/endpoints. These are not experiments
on newly trained wireless MNIST/CIFAR models.

## Remaining boundaries

* Flower1.29/Ray, Bayesian-Torch and CUDA were unavailable in the authoring runtime.
  Optional real Flower tests are present but skipped here; GPU/Ray correctness needs
  the actual target-environment gates. No dependency files were changed to suit the author.
* Wireless physical settings are not yet locked. The source accepts explicit profiles
  and channel traces; the included example is labeled nonfinal.
* Ideal successful digital transmission, fixed PSD and fixed slots are modeling choices.
  Control counts exclude radio/network framing and startup/CSI costs. Energy is not total
  device energy. No real radio measurement, AirComp or packet-loss recovery is implemented.
* Source preservation and tests are evidence, not a proof of absence of all bugs.
  No universally certified integer optimality or future accuracy guarantee is asserted.
* The existing `.gitignore` was supplied and retained; live Git tracked/history status was
  not inspected. A same-SSD archive is not an independent backup.
* Stage3B completion remains user-reported unless separately verified on the server.
  The earlier recovered FedAvg difference is recorded, but its cause is not established.

The handoff exporter reads current files. Its default mode checks legacy24 and its
explicit wireless mode checks variable-count metrics. It does not automatically update
docs from conversations or create complete resume backups. Stop concurrent writers
before exports; per-file checks do not make the entire filesystem snapshot atomic.
