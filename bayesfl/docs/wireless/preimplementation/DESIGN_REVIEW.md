# Wireless FedKLS design review

## 1. What the supplied proposal actually specifies

Sources: `FedKLS_Wireless_Develop.zip/README.md`, `wireless_model.tex`,
`joint_problem.tex`, `method1.tex`, `implementation.tex`, and `IMPLEMENTATION.md`.
The archive contains a LaTeX manuscript and an integration plan, **not Python
wireless training code**. Its interfaces are suggestions, not installed APIs.

All participating clients train the full model once. Uplink messages select
posterior coordinates after training. Clients report current-round information,
the server coordinates a common price, and counts and bandwidths are chosen
before posterior upload. All K clients upload; there is no whole-client exclusion,
no menu of allowed retention percentages, and no AirComp sum.

The proposed physical abstraction is reliable digital transmission using
orthogonal frequency resources. Successfully decoded posterior values are exact
copies of their finite-precision transmitted representation. Fading affects rate,
feasibility, bandwidth and retention. The initial proposal does not add Gaussian
noise to decoded means/precisions, packet loss, retransmissions, quantization,
client dropping, error feedback, PVI, or local model pruning.

### Timing and rates

Use Hz, seconds, bits, watts and joules consistently. Let beta be linear path gain,
h a block-fading coefficient and p_PSD the fixed uplink transmit PSD in W/Hz:

```text
gain[k,t] = beta[k] * abs(h[k,t])**2
snr[k,t] = p_PSD[k] * gain[k,t] / N0
c[k,t] = log2(1 + snr[k,t])
R_UL[k,t] = B[k,t] * c[k,t]
P_UL[k,t] = p_PSD[k] * B[k,t]
```

The proposal suggests fixed positions and independent round-wise Rayleigh fading.
Keep current channel efficiency fixed during each allocation/upload interval;
refresh the uplink information for that phase rather than assuming an arbitrarily
old measurement survives local training. Downlink uses the worst intended receiver
and a single dense multicast payload. Its SNR convention must also be explicit.

```text
U[t] = T_target[t] - T_DL[t] - max_k(T_local[k,t])
B[k,t] = L(m[k,t]) / (U[t] * c[k,t])
```

For a modeled control allowance, subtract it before allocating. This is a proposed
engineering extension supported by the proposal's overhead discussion; keep the
zero-overhead analytical convention available and label it honestly.

Equality recovery makes all idealized joint-policy uploads last U seconds. It
may leave bandwidth unused. Spending all spare bandwidth to end earlier is a
different completion policy. Version A in the proposal instead grants B/K and
can finish uploading early. A fixed-slot comparison must state whether such
policies wait until the slot ends; do not mix slotted and work-conserving clocks.

With fixed total power, c depends on allocated bandwidth. The supplied closed-form
threshold is not derived for that model. Reject that combination rather than
quietly computing SNR at an arbitrary reference bandwidth. Under fixed PSD,
`p_PSD[k] * B_total <= P_max[k]` is a conservative sufficient per-client power
condition; it is not an additional power-control optimization.

### Optimization and meaning of the objective

For valid diagonal Gaussians, the global-to-local marginal scores are
`s[k,i] = KL(q_global_i || q_local_ki)`. Sort descending with deterministic index
ties and define `D[k,m] = sum(sorted_scores[k,m:])`, so `D[k,d] = 0`.
Prior-fill reconstruction implies

```text
KL(q_reconstructed_k || q_local_k) = D[k,m]
F(m) = sum_k pi[k] * D[k,m[k]]
w[k,m] = L(m) / (U * c[k])
minimize F(m) subject to sum_k w[k,m[k]] <= B_total
m[k] in {1,...,d}
```

This is local reconstruction KL, not KL to the true posterior or a guaranteed
accuracy/loss/calibration improvement. Global aggregation is a separate step.

For price lambda >= 0, the best sparse count is

```text
theta[k] = 2*b*lambda / (pi[k]*U*c[k])
m_sparse = min(d-1, max(1, count(scores > theta[k])))
```

Compare its complete priced cost, including its bitmap, against the dense option.
Choose the lower priced objective; use smaller bandwidth for exact ties. The
constant bitmap cancels in sparse marginal differences, not in dense comparisons.
The implementation must distinguish `price_lambda` from FOLA's `prior_lambda`.

Exact local minimizers give nonincreasing **bandwidth demand**, including dense
format switches. Counts alone are not a safe demand proxy. Maintain the best
feasible incumbent, optionally initialized with feasible Version A counts, and
track the maximum queried dual lower bound. Finite price tolerance is not integer
optimality. Ordinary floating-point gaps are numerical diagnostics unless local
minimization errors have a rigorous bound.

The independent exact-rational checks in this review tested both the proposal's
ideal bitmap formula and the proposed byte-padded implementation formula. They
support the stated local threshold, demand monotonicity and weak-duality bounds
on the tested finite instances; they do not certify a future implementation or
all possible instances.

## 2. Baseline-to-proposal mismatches and proposed resolutions

### 2.1 Preserve raw precision on the wire

The baseline sends float32 mean and raw precision/omega, not variance:
`config.py:223-224`, `posterior/sparse.py:163-205`, and `posterior/packing.py`.
The proposal speaks of mean/variance pairs. Preserve the actual representation
and explicitly revise implementation-facing terminology; do not silently invert
precision arrays on upload. Two float32 values still occupy 8 bytes/coordinate.

The saved MNIST FOLA configs use initial precision 1, `strict`, floor parameter
1e-8, and no gradient clipping. CIFAR FOLA starts at raw precision 0 and sparse
configs use `floor_for_score`, resolving to 1e-12. This adapter computes a positive
**scoring surrogate** and leaves trained/transmitted/aggregated omega untouched.

For CIFAR, the proposal's exact reconstruction-KL identity is therefore an identity
for the declared surrogate Gaussians when flooring is active, not for an improper
raw zero-precision Gaussian. Do not add optimizer damping or clip posterior values
to satisfy a prose assumption. Record floored fractions alongside objective values.

### 2.2 Add a versioned mixed sparse/dense packet contract

The baseline v1 sparse encoder sends a packed bitmap even at m=d; its decoder
requires `bitmap_policy="always"` and exactly three arrays. The proposal's full
retention endpoint omits the bitmap. Preserve v1 and introduce an explicit opt-in
v2 packet format supporting a mix of sparse and dense clients within a round.

For the existing float32 representation, the proposed implementation cost is:

```text
L_v2_bits(m) = 64*m + 8*ceil(d/8),   1 <= m < d
              64*d,                 m == d
```

The manuscript uses d bitmap bits and explicitly neglects byte padding. Here both
model dimensions require six padding bits: 80,202 -> 80,208 bits; 878,538 -> 878,544
bits. With 20 sparse uploaders the difference is 120 bits = 15 bytes/round. It is
small but must not be hidden in exact byte assertions. One shared payload helper
must serve prediction, allocation, codec checks and logging.

Near-dense sparse messages can cost more than dense messages. Dense weakly
dominates those counts when comparing nonnegative reconstruction costs. Preserve
this separate endpoint in feasibility tests and optional refinement.

### 2.3 Retained counts must stay integers end to end

The baseline computes `ceil(keep_ratio*d)` in the encoder, decoder and budget
predictor. A wireless allocation must carry **integer m**, not set a global ratio
and recover m by multiplication. Actual probes found:

```text
d=80,202:  m=25  -> ratio=m/d -> keep_count(d,ratio)=26
d=878,538: m=111 -> ratio=m/d -> keep_count(d,ratio)=112
```

Across the model dimensions, 4,643 and 34,400 respective counts changed under that
binary64 round trip in the review environment. Use ratios only as derived display
values. Validate the received mask cardinality against the stored per-client grant,
not against a client-reported number alone.

### 2.4 Split the one-call training/upload lifecycle

`BayesFLNumPyClient.fit()` currently trains, compresses and returns the upload in
one call, then releases the model. `configure_fit()` reserves a known constant
round payload before dispatch. The new decision cannot happen correctly inside
that unchanged lifecycle: current-round local scores do not exist at dispatch.

Introduce explicit prepare -> price-query -> grant/upload -> commit phases.
Preparation trains exactly once and stores the full local posterior and scores
locally. The server receives only the specified scalar responses and the final
packet, never a full score vector/full unselected posterior disguised as metrics.
Price queries must not retrain, re-upload the global posterior or count as model
rounds. Reuse the original trainer functions and RNG reseeding.

`server.py:91-98` reconstructs a client from `context` but does not pass persistent
state. A plain `self.cached_posterior` is insufficient for the planned multi-call
protocol. Use a tested `Context.state` adapter or an explicitly client-owned cache
with run/client/round/snapshot keys. Confirm the exact API on the pinned Flower
1.29 environment; the generic design is supported by Flower's stateful-client
reference, not a substitute for that integration test.

### 2.5 Dynamic budgets need a new reservation lifecycle

`RunState.next_round_cost()` assumes one cfg.keep_ratio for all clients, and
`finish_round()` asserts observed bytes equal that pre-reserved amount. Changing
only the selector would violate this contract.

For the first wireless time-budget study, keep the legacy byte-budget controller
unchanged and use a separate wireless round controller. Log bytes while stopping
on explicit modeled time/round limits. Do not automatically combine the old byte
cap with the new optimization constraint.

If simultaneous strict total-byte and time limits are later required, either
reserve a conservative full round before dispatch, or formulate the remaining-byte
constraint explicitly and account for work already performed. Do not reject an
expensive post-training decision and pretend its downlink/control/training cost
never occurred. A minimum-only reservation does not guarantee the selected upload
fits a remaining total byte budget.

### 2.6 Config support must be explicit

The current `load_config()` builds only recognized sections and ignores unknown
root sections. A probe with `wireless: {enabled: true}` loaded normally and lost
the entire section. Add a real dataclass/parser/validator, reject unsupported radio
modes and unknown wireless fields, and serialize every physical/protocol choice.
Do not advertise runtime flags before implementing their parser.

When wireless is disabled, exclude the default wireless section from the legacy
configuration fingerprint. Merely adding a dataclass changes `asdict()` and would
otherwise invalidate old resume hashes. Enabled wireless runs need a versioned
fingerprint covering channel/trace, time model, protocol, allocator, tolerances and
control policy. Radio changes are not accounting-only migrations.

### 2.7 Preserve aggregation and distinguish clocks

The inspected FOLA reducer uses `sum(pi*omega*mu)/(sum(pi*omega)+epsilon)`;
`RunState.finish_round()` additionally restores globally unselected coordinates
from the broadcast. Preserve both. An ideal unregularized Gaussian-product formula
must not silently replace this numerical baseline.

Normalize client weights once over the full participating roster, never per
coordinate. When a client is dense, its mask is all true for the omission invariant.
Existing mean-point evaluation remains unchanged; ECE/NLL of that predictor is not
MC Bayesian calibration certification.

Existing `round_wall_time_seconds` is emulator runtime. Add modeled wireless time
as separate fields. GPU/Ray execution duration is not the parallel device latency
or a wireless transmission measurement. CIFAR's supplied partition metadata has
min/mean/max sizes 1 / 2,499.4 / 8,706; do not set every client's training time from
2,500 examples or ignore FOLA curvature work.

## 3. Feasibility, baselines and claims

Require finite U>0 and c[k]>0. Minimum bandwidth is
`sum_k min(w[k,1],w[k,d])`. If it exceeds B, return infeasible. Do not clamp an
unaffordable count to one, floor a bad channel to force success, discard a client,
redraw fading until feasible, or stretch the deadline without recording a changed
policy. Rayleigh power has no positive lower support bound, so fixed finite
all-client deadlines are not guaranteed feasible for every realization.

Initially make infeasibility explicit and stop the attempted run rather than
silently changing the experiment. Record any already-consumed time/bytes. Continuing
through failed slots needs a separate slot index, a rule for unchanged global
state, channel evolution and time charging; that is a later explicit design.

Primary allocator controls should share local FOLA, data, initialization, numerical
policy, topology, channel trace, target time and comparison clock:

- Version A: equal bandwidth, largest feasible count with dense checked first.
- Uniform retention: one feasible count across clients, with a declared bandwidth
  recovery rule. This is not necessarily equal bandwidth.
- Version B: common-price joint allocation, feasible incumbent and diagnostic gap.
- Random-selector control: matched **actual integer counts and bandwidths** on
  frozen identical local posteriors. Across independent training, specify whether
  the allocation trace is replayed or recomputed; recomputed trajectories do not
  automatically have matched counts.
- Full-posterior FOLA / dense FedAvg: obey the same radio limits and report when
  their dense payload cannot meet the target. FedAvg transmits one float32 vector,
  so its payload is 32*d bits, not 64*d. Its training time may also differ.

Keep old clipped FedAvg curves as historical references. A cleaner new wireless
MNIST FedAvg control would match unclipped SGD, but that is a separately documented
experimental choice, not permission to replace the historical baseline.

Better F(m) than feasible Version A is guaranteed only for the same frozen
posteriors when its incumbent is included. Multi-round accuracy, calibration,
energy savings and faster convergence remain hypotheses. Equal bandwidth limits
under fixed PSD are not automatically equal-energy experiments. Energy for
Version A uses its actual on-air duration unless idle transmission is modeled.

## 4. What can be reused from existing outputs

All 24 primary CSV checkpoints and exact byte totals were read successfully from
`baseline_src.zip`. The CIFAR R150-budget endpoints are now available: dense53.10,
FedAvg47.28; KL53.70/53.58/57.53/53.42/51.98 and
random52.73/50.07/44.76/38.73/33.63 for r=.75/.5/.25/.1/.05.
See the full raw table and exact configuration records under `evidence/`.

Global metrics and reliability-bin NPZs do not contain current-round full local
posterior pairs or sorted KL tails. The primary run trees in this archive have no
`compression_snapshots/*.npz`. Frozen-posterior replay therefore starts with
synthetic states or a separately authorized client-side capture of real local
states; it cannot reconstruct the missing per-client scores from accuracy curves.
Offline diagnostic access must not become a hidden production upload.
