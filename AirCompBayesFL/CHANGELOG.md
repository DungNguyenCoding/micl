# Changelog

## 1.3.2

- Fixed the numerically frozen precision coordinate observed at rho=400.
- Keeps rho as a float64 master variable during Pyro VI, client state, server
  aggregation, and AirComp reconstruction.
- Casts sampled Bayesian weights to float32 only for the CNN forward pass.
- Preserves sub-float32-ULP precision updates in both ideal and wireless modes.
- Adds per-client precision delta and applied-gradient diagnostics.
- Adds a regression test for 1e-6 precision updates at rho=400.

## 1.3.1

- Optimizes the Bayesian precision coordinate `rho` directly with Pyro's
  differentiable ELBO and PyTorch SGD, matching Eq. (25).
- Keeps the phase-2 guide covariance at `Sigma_{t+1}` while regularizing
  against the round-start global prior `q_{theta_t}` from Eqs. (13) and (15).
- Scales mini-batch likelihoods to unbiased full-local-dataset estimates.
- Temporarily broadcasts `[mu_t, rho_{t+1}, rho_t]` during the natural-mean
  phase, then returns to the normal `[mu_{t+1}, rho_{t+1}]` server state.
- Adds source-contract tests for the corrected coordinates and phase-2 prior.

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
