# Implementation contract and proposal adaptations

## 1. Basis and explicit adaptations

The basis is `proposal/wireless_model.tex`, `joint_problem.tex`, `method1.tex` and
`implementation.tex` in the supplied `FedKLS_Wireless_Develop.zip`. The following changes
are implementation choices, not silently attributed to the manuscript:

* Means and **raw precision**, not variance, remain float32 on the wire so the baseline
  trainer/reducer arithmetic is preserved. For CIFAR, zero raw omega is permitted and
  floored only for KL scoring; the objective is then a positive-surrogate Gaussian KL.
* The bitmap has `8*ceil(d/8)` bits, including byte padding, not idealized `d` bits.
  Full retention has no bitmap **in v2 only**; the old sparse v1 format is untouched.
* Control signaling has an explicit reserved interval and a canonical-JSON byte model.
  Its time is subtracted before allocating model uploads. This is not measured Flower,
  TCP, MAC, CSI-estimation or PHY framing traffic.
* All policies use fixed-length slots: early completion is logged as idle time, not
  automatically used to start another round. Joint equality recovery is preserved.
* Binary64 lower-bound/gap values are numerical diagnostics. `certified_optimal` is false;
  no interval-arithmetic proof or general integer optimality claim is made.
* `uniform` is an additional baseline using one common maximal equal-share feasible count.
  Random control uses joint allocation from its own current KL scores, then independent
  random coordinates. Same-state count matching does not imply identical future trajectories.

## 2. Learner and reconstruction

`training/fola.py`, `training/deterministic.py`, model definitions, data generators and
`posterior/aggregation.py` are unchanged. PREPARE reproduces the baseline client/round
seeds and trainer calls; the optimizer is freshly created for each fit as before.
FOLA's estimator remains squared minibatch-mean task gradients, not a newly implemented
per-example Hessian/Fisher. The regularization lambda is distinct from wireless price.

Compute `KL(global-before || local-after)` model-wide with existing stable Gaussian
score functions. Strict mode requires positive scoring precision. `floor_for_score`
uses the configured score-only floor and does not alter trained/uploaded/aggregated omega.
Sort by descending score with ascending coordinate-index ties. Build reverse cumulative
tails `D(m)` rather than subtracting nearly equal total/prefix sums.

Selected values are sent in ascending coordinate order. The server fills omissions from
the exact immutable broadcast, normalizes dataset weights once across all clients and
uses the original FOLA reducer. Its epsilon denominator remains; coordinates omitted by
all clients are restored exactly. No per-coordinate weight renormalization or 1/r scaling.

## 3. Allocator

For K clients, normalized weights pi, efficiencies c and remaining upload time U:

```
L(m) = 64*m + 8*ceil(d/8)  if 1 <= m < d
L(d) = 64*d
w_k(m) = L(m)/(U*c_k)
F(m_vector) = sum(pi_k * D_k(m_k))
```

The server exposes only a count-query and price-query interface. At price p, the client
threshold is `64*p/(pi_k*U*c_k)` with strict greater-than; compare the sparse candidate
with dense separately, then use required bandwidth/discarded KL to break ties. Every
integer count 1..d is available. No discretized keep-ratio menu and no ratio round-trip.

The coordinator validates feasibility, retains a minimum-payload fallback, considers an
equal-share incumbent, tests full dense/price zero, brackets by finite doubling and
bisects. Every feasible improvement is retained. Iteration/price-resolution limits return
the best feasible incumbent, not an unqualified optimum. A numerical inconsistency
between diagnostic dual value and incumbent causes explicit failure. No greedy local
refinement or second optimization method is claimed.

Version A assigns B/K, chooses maximal feasible per-client count and waits for slot end.
Dense FOLA/FedAvg require the all-client dense payload to fit; an infeasible comparison
is reported, never fixed by changing its scenario. Minimum feasibility can permit joint
allocation when equal bandwidth cannot support every client.

## 4. Time, power and channels

Fixed PSD: `snr_k = psd_k * gain_k / noise_psd`, `c_k=log2(1+snr_k)`.
A fixed-total-power model is not implemented; changing to it invalidates this allocator's
bandwidth-independent-efficiency derivation. Optional client caps conservatively require
`psd_k * total_UL_bandwidth <= cap_k`.

Fixed gains, independent counter-seeded block-Rayleigh power gains, or explicit per-round
trace gains are supported. Reciprocal fading is an explicit option. No redraw on poor
channels. DL is one dense multicast limited by the worst downlink spectral efficiency.
Local modeled preparation uses actual partition sizes:

```
prep_k = n_k * E * seconds_per_sample_epoch_k + fixed_seconds_k + score_seconds_k
U = target - DL_time - max(prep_k) - control_reserve
```

`score_seconds` represents declared extra preparation, not measured sorting time.
Emulator wall times are logged separately. Optional compute profiles are not estimated
from the server's GPU runtime. Gains are explicit inputs; no undocumented path-loss or
cell-topology law is silently chosen.

The shared control model charges canonical instruction/response UTF-8 JSON at the stated
aggregate rate plus one declared latency per exchange. Startup identity discovery, CSI
acquisition, retransmissions and protocol framing are excluded. Reserve/describe those
separately in any physical interpretation. Data-plane modeled energy includes UL and DL
PSD*bandwidth*airtime; control/compute/circuit/idle energy is not included.

## 5. Protocol lifecycle and failure scope

Bindings cover protocol, run token, logical client, true round, parameter layout and
broadcast snapshot. Grants additionally hash the integer count, representation, payload
size, bandwidth, upload interval and efficiency. Validate counts, dtypes, finite values,
precision validity, bitmap padding and binding before aggregation.

PREPARE sends the only model downlink. TAIL/PRICE requests and replies carry no arrays.
UPLOAD carries only granted arrays. RELEASE clears client state before evaluation/commit.
A second prepare without release, repeated upload or stale grant is rejected. Flower
uses Context.state instead of assuming a Python NumPyClient instance persists.

Model arrays actually received are charged before validating packets. Undecodable/failed
RPC replies mark accounting incomplete, and no failed trajectory is accepted as a clean
comparison. Known infeasibility before dispatch spends no modeled data-plane resources;
failures after dispatch retain consumed resources, with a conservatively charged modeled
preparation barrier. No in-flight rollback/retry or interrupted-round resume is supplied.
Local backend object separation is an emulation contract, not cryptographic isolation.

## 6. Separation from legacy code

Existing source edits are limited to opt-in configuration/CLI/server routing and the
handoff exporter. The new implementation lives in `src/bayesfl/wireless/`. No old YAML,
trainer, reducer, partition, plotting script, queue runner, dependency or Git-ignore
policy is rewritten. Wireless `keep_ratio=1.0` is a required placeholder; **the grant's
integer count is the only operative upload cardinality**.
