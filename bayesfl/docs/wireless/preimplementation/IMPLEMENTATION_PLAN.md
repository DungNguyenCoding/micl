# Implementation and documentation plan

**All new paths, types and status names below are proposed, not installed APIs.**
The current production learner remains the uploaded baseline.

## 1. Preserve these boundaries

Keep the local trainer equations, models, partitions, data transforms, score-only
precision policy, normalized aggregation weights and existing evaluation unchanged.
`wireless.enabled=false` must preserve legacy behavior, byte totals, old packet
validation and clean-round resume compatibility. Historical `run_*` output trees
and published plot scripts are read-only regression references.

Prefer a separate wireless controller/state implementation, reusing pure baseline
helpers, rather than extending the legacy constant-cost RunState with ambiguous
mode-dependent assumptions. No Flower upgrade, AirComp conversion, client dropping
or learning-rate tuning is part of this implementation.

## 2. Proposed new module structure

```text
src/bayesfl/wireless/
    __init__.py
    types.py              # ChannelState, ClientScoreState, Grant, Allocation
    channel.py            # topology, phase-specific fading, PSD/rates and trace ids
    timing.py             # parallel preparation, DL, controls, fixed-slot time
    payload.py            # shared byte-padded sparse / bitmap-free dense cost
    local_oracle.py       # sorted scores, reverse tails, exact-count local response
    allocation.py         # shared price, bracket/bisect, incumbent, diagnostic gap
    baselines.py          # equal bandwidth, feasible uniform count, dense feasibility
    protocol.py           # versioned grants, heterogeneous codec, snapshot checks
    client_state.py       # persistent client-owned prepare/query/upload state
    client.py             # wireless operation dispatch; calls original trainers
    state.py              # dynamic commitments, modeled time and resume state
    server.py             # multi-phase round controller on pinned Flower API
    accounting.py         # separate model/control/radio/emulator accounting
```

This is a responsibility map, not a requirement to create many empty files. Combine
small pure modules if warranted while retaining the logical boundaries.

Core proposed interfaces:

```text
payload_bytes(d, m, format_version) -> exact integer
prepare_score_state(global_arrays, local_arrays, precision_policy, layout) -> local state
local_price_response(local_state, price_lambda, pi, U_s, c) -> (m, discarded_kl)
allocate(local_oracles, weights, U_s, efficiencies, total_hz, options) -> Allocation
encode_granted_update(local_state, grant) -> arrays + scalar metadata
decode_granted_update(packet, grant, broadcast) -> reconstructed_arrays + mask
```

`m` is an integer throughout. `m/d` is only a logged/display ratio. The coordinator
operates on scalar oracle responses; its implementation must not inspect the
closure/cache containing untransmitted scores. Full vectors may be used only by a
separately labeled offline test oracle.

`Allocation` records feasibility, counts, formats, per-client bandwidth, total
bandwidth, objective, best dual lower-bound diagnostic, gap, bracket/query counts,
termination reason and snapshot/channel ids. Suggested reasons include
`all_dense`, `zero_distortion`, `gap_target`, `price_tolerance`, `iteration_limit`,
`bracket_limit`, `infeasible` and `numerical_failure`. `price_tolerance` alone is
not a success certificate.

## 3. Existing files requiring deliberate changes

| Existing file | Required work | Protection |
|---|---|---|
| `src/bayesfl/config.py` | Parse/validate/serialize opt-in wireless settings, policies and physical units | Preserve every existing config default; reject ignored wireless keys |
| `src/bayesfl/main.py` | Route new mode; explicitly support any new overrides and resume schema | Existing CLI remains valid; no guessed switches |
| `src/bayesfl/server.py` | Select wireless server/client/state only when enabled; pass persistent Context | Keep baseline client/server path unchanged |
| `src/bayesfl/client.py` or a small shared training helper | Expose original full local preparation for reuse | Equivalence tests against original fit output; no duplicated/reimplemented learner |
| `src/bayesfl/posterior/sparse.py` | Reuse stable scores, ties, flattening and mask helpers; only additive refactoring if needed | Preserve v1 encoder/decoder and bitmap-at-full behavior |
| `src/bayesfl/communication.py` | Factor shared byte helpers where appropriate | Never reinterpret old counters as measured wireless traffic |
| `src/bayesfl/experiment_state.py` | Backward-compatible fingerprint dispatch; possibly shared commit helpers | Old fingerprint stays identical when wireless is disabled |
| `src/bayesfl/budget_server.py` | Prefer unchanged legacy loop, with separate wireless subclass/controller | Old true-round/byte stopping unchanged |
| `src/bayesfl/strategies/research_strategy.py` | Reuse normalized reducer/evaluation utilities or route to wireless strategy | No call to fixed-cost aggregate path for preparation/query replies |
| `src/bayesfl/evaluation.py` | Attach additional wireless fields via state interface | Preserve prediction, loss, ECE and legacy columns |
| `src/bayesfl/logging_utils.py` | Extend only if new log record types require it | Atomic metric writes, no duplicate committed rounds |
| `docs/tools/export_handoff.py` | Add explicit wireless registry/export handling | Keep legacy 24-run check as a separately supported mode |

The original `posterior/aggregation.py`, `training/fola.py`,
`training/deterministic.py`, model and partition code should need no algorithmic
changes. Any unavoidable refactoring must be covered by identical-input regression
tests and disclosed in the change manifest.

## 4. Multi-phase protocol and local state

A viable design retaining the baseline's legacy RPC interface is:

```text
server: choose/validate fixed roster, round/slot identity and known input plan
PREPARE FitIns: one broadcast posterior + operation/round/snapshot metadata
client: original full training once -> prepare sorted scores/tails -> store locally
PREPARE FitRes: empty model arrays + scalar identity/n/examples/preparation report
barrier: all required current local states exist
channel: reveal/refresh current upload efficiency for this phase
PRICE GetPropertiesIns: scalar price, U, weight, efficiency, query/snapshot id
PRICE response: integer m and scalar D(m), plus protocol identity fields
server: bracket/bisect; maintain feasible incumbent and lower-bound diagnostic
UPLOAD FitIns: empty model arrays + immutable per-client integer-count grant
client: encode from stored state; do not train or rescore/re-sort
UPLOAD FitRes: granted mean/precision packet and scalar validation metadata
server: charge/validate packet -> prior fill -> original reducer -> omission invariant
evaluate -> persist completed state -> acknowledge/release client scratch state
```

These are proposed phases on existing RPC families, not already-supported CLI or
Flower operations. Test them against the pinned Flower 1.29 installation. A process
restart or a new ClientApp instance must not lose preparation state between queries.
Do not infer persistence from a Python instance attribute.

State keys need run id, logical client id, true model round, snapshot/layout digest
and protocol version. Query ids must be unique; upload must refer to the selected
grant, not whichever price trial happened last. Reject stale/surplus/repeated
messages. Preserve independent training, mask, scheduling and channel RNG streams.

Client caches hold CPU arrays, sorted global coordinate indices, score tails,
original local metrics and identifying digests. They are not server input. Release
them after a completed transaction; do not retain GPU models through price queries.
A large client state serialized inside simulator infrastructure is not a measured
radio upload or a cryptographic privacy guarantee.

## 5. Time, bytes, controls and resumption

Use distinct names and fields for:

- actual finite-precision model-array bytes and bitmap bytes;
- modeled control reports/grants/CSI framing, query counts and latency;
- modeled downlink, local preparation maximum, upload interval and round time;
- simulation/emulator wall time;
- implied per-client transmit power and uplink energy under fixed PSD.

Count one modeled multicast downlink, K distinct final uploads and every declared
control exchange. A free in-process oracle call is not evidence of zero signaling
latency. A JSON/Python object's host size is not a wireless framing size.

Initial implementation recommendation: support fixed modeled completion slots with
explicitly recorded zero-overhead analytical mode or a finite reserved control-time
budget. Do not infer true hard real-time delivery from idealized rates. When controls
are modeled, bound queries and abort on allowance overrun rather than claim the
original deadline was met. Policy-specific computation/overhead changes U; the
frozen-round no-worse-than-A guarantee applies only when A's counts are feasible
under the same U and constraints.

Infeasibility handling must record any consumed work without advancing the global
model. Initial fail-fast behavior is simpler than continuing through no-update
slots. If later allowing missed slots, add `slot_id` distinct from committed
`round_id` so fading and elapsed time advance even without aggregation.

Preserve clean completed-round resume only at first. Save channel seed/trace id,
slot/model indices, cumulative modeled time/energy/control costs, grant/allocation
summary, ledger hash/count, and versioned config/protocol fingerprint. Reject
in-flight mismatches rather than discard costs. Old non-wireless resumes must still
work unchanged. Do not turn a historical completed run into a wireless continuation
by editing its config: it is a new experiment/protocol.

## 6. Configuration choices still required

The proposal contains no locked numerical radio profile. A future validated config
must state topology/reference gain/path-loss exponent (distinct from Dirichlet
alpha), client locations, separate UL/DL bandwidths, transmit PSD or specified SNR
trace, noise density, block-fading/CSI assumptions, seed/trace policy, local compute
model/profile, score/ranking-time policy, target slot seconds and control allowance.

Also specify integer min count=1, d>=2, float32 wire values, padded bitmap, dense
format version, allocation policy, finite bracket/query limits, numerical tolerances,
failure policy and stopping mode. Derive d from the model/layout; do not independently
hardcode a contradictory dimension. The first implementation can reject BBB,
client subsampling, fixed-total-power mode, random power clamps, mid-round resume
and simultaneous total-byte optimization as unsupported.

Keep the existing MNIST s10/N20/seed0/E5/B10/LR.10/no-clip FOLA and CIFAR
alpha.01/f1/N20/seed0/E10/B32/LR.02/cos400/clip10 FOLA profiles. New wireless policy
comparisons need the same deterministic identity/order policy. The saved historical
CIFAR dense/FedAvg runs have deterministic scheduling false while sparse runs have
true; adopting matched deterministic ordering for all new wireless controls must be
explicitly documented, not described as bitwise replay of those old trajectories.

## 7. Deliver in gates, with docs in every gate

| Gate | Source/tests | Required documentation | Exit condition |
|---|---|---|---|
| W0 — reviewed snapshot | Source/config/metric audit, existing tests, mathematical probes | This folder and evidence | Complete here; no production wireless code claimed |
| W1 — numerical contract | Byte-padded payload, score state, integer oracle, A/uniform/B policies, pure CPU tests | Numerical model/units, representation adaptations, optimum-vs-diagnostic language | Local brute-force and tiny joint enumeration pass; incumbent always feasible |
| W2 — transport and compatibility | Mixed packet codec, grant validation, optional config, fingerprint tests | Packet/schema/version reference, legacy invariants, resume semantics | v1 unchanged; v2 actual bytes/counts match; disabled hashes match old checkpoints |
| W3 — end-to-end lifecycle | Persistent client state, staged RPC server, clocks/accounting, evaluation, clean resume | Runbook, command rules, logging schema, source map, exporter and handoff | Real pinned-Flower integration plus uninterrupted/resumed equivalence |
| W4 — bounded validation | Explicitly named short smoke; known partition, matched traces | Validation log with exact environment and skips/failures | No full-study launch until runtime checks pass |
| W5 — experiments | New uniquely named wireless configs/queues; terminal extraction/plots | Locked wireless profile, new registry, result provenance and analysis | Interpret requested results before expanding queue |

No large queue is prescribed at W0. First use synthetic frozen posteriors, then
separately captured real local states, then a short MNIST test, then controlled
MNIST/CIFAR wireless studies. Do not silently add five-seed sweeps or retune E/LR.

## 8. Existing documentation requiring synchronized updates

| File(s) | Update required with production integration |
|---|---|
| `README.md`, `AGENTS.md`, `docs/START_HERE.md`, `docs/README.md` | Add opt-in wireless entry point and reading order; distinguish implemented stages from proposals |
| `docs/CHAT_HANDOFF.md`, `docs/DECISIONS_AND_NEXT_WORK.md` | Record actual baseline SHA, implemented gate, open radio choices and next exact task |
| `docs/EVIDENCE_AND_GAPS.md` | Close supplied-source and CIFAR-R150 gaps; retain GPU/Flower and Git-status limits |
| `docs/LOCKED_EXPERIMENTS.md` | Mark now-read resolved precision/CNN settings verified for snapshot; preserve old non-wireless locks |
| `docs/COMMUNICATION_ACCOUNTING.md`, `docs/ALGORITHM_NOTES.md` | Distinguish v1/v2 packet formulas, bits/bytes/seconds, score surrogate and actual reducer epsilon |
| `docs/REPOSITORY_MAP.md`, `docs/SCRIPT_REGISTRY.md` | Add actual installed modules/scripts only |
| `docs/RUNBOOK.md`, `docs/COMMAND_RULES.md` | Add tested wireless commands, staging, queue validation and actual output extraction |
| `docs/RESULTS_AND_PLOTTING.md` | Keep legacy figures; add separately named time/resource diagnostics; do not imply wireless results exist |
| `docs/EXPERIMENT_REGISTRY.json` | Preserve the 24 primary non-wireless figure records; record verified values without erasing old provenance |
| proposed `docs/WIRELESS_EXPERIMENT_REGISTRY.json` | Explicit new run metadata, policy/trace/clock/deadline and outputs; populate only after runs exist |
| `docs/DATA_AND_GIT.md`, `docs/tools/README.md`, exporter | Document/capture small radio traces and wireless metrics explicitly; distinguish handoff from resume backup |
| `IMPLEMENTATION_REPORT.md`, `VALIDATION.md`, `VALIDATION_SPARSE.md`, `SPARSE_README.md`, `docs/DOC_CHANGELOG.md` | Report precise changes/tests with dates/environment; never replace historical evidence with an unrun claim |

Archive old evidence rather than rewrite history. The supplied main docs still say
CIFAR R150 values are absent, current source unavailable and Git-ignore changes
unconfirmed; this review closes the first two for the uploaded snapshot and confirms
the actual ignore file, not tracked Git status.

## 9. Exporter compatibility is part of the feature

The present `docs/tools/export_handoff.py` hardcodes 24 primary figure records,
fixed ratios, constant per-round byte formulas and a two-profile printout. Its
output allowlist selects only those registered experiments. Adding wireless code
will copy the `.py` files but not automatically export its new experiment outputs.

Add an explicit wireless registry/mode or a validated extensible registry schema,
keeping the old 24-run mode intact. Select wireless per-round/client metrics,
channel/topology/config metadata and needed small control diagnostics. Do not pass
wireless metrics through `cost == round * fixed_ratio_cost` assertions. New TeX
proposal files also require an explicit suffix/allowlist decision; the present
exporter excludes `.tex`. This planning package includes numbered proposal text as
TXT so it is not lost by the existing exporter.
