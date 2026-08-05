# Changelog

## 1.3.0

- Replaced the legacy single-client-call Bayesian update with the exact
  server-separated precision/natural-mean schedule from Algorithm 1.
- Added `bayesian_protocol.py` with physical/logical round mapping and the
  diagonal `rho <-> nu <-> mu` coordinate transforms.
- Added separate Pyro SVI entry points:
  `train_precision_phase` and `train_natural_mean_phase`.
- Proposed clients now return one `d`-value vector per phase and persist their
  phase-1 precision until the matching phase-2 call.
- Added separate AirComp aggregation functions for `Delta rho` and `Delta nu`.
- Both proposed phases reuse the same block-fading channel realization within a
  logical round, while additive noise samples remain independent.
- Flower/Ray runs now use two physical rounds for every logical proposed round;
  evaluation is skipped between the phases.
- Added phase-aware CSV fields, per-phase AirComp diagnostics, posterior
  precision summaries, and client `rho/nu` diagnostics.
- Added tests for phase order, the Eq. (33)/(34) coordinate round trip, ideal
  two-phase conflation, logging fields, AirComp, wireless normalization, and
  model size.

## 1.2.1

- Added missing wireless diagnostic fields to the CSV schema.

## 1.2.0

- Added explicit path-loss reference-distance normalization.

## 1.1.0

- Added stable native-Windows sequential CUDA backend.
