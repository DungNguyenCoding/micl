# Decisions and unresolved implementation choices

This is not a record of user approval for new numerical hyperparameters. Sources,
recommendations and open decisions are separated below.

## Specified in the supplied proposal

- Reliable orthogonal digital links, not analog AirComp.
- All K clients train the full local FOLA model; no selection of whole clients.
- Current-round KL(global || local), with local sorted scores/tails.
- Integer counts from 1 through d, with no retention-ratio menu.
- Full dense downlink multicast; concurrent orthogonal uploads.
- A synchronized preparation barrier and prescribed common completion target.
- Bandwidth-independent spectral efficiency, physically interpreted with fixed PSD.
- Bitmap for sparse packets and no bitmap at the dense endpoint.
- One solver family: common-price Lagrangian decomposition, threshold plus dense
  comparison, bracketing/bisection, feasible incumbent and lower-bound diagnostic.
- Full local posterior estimator/reconstruction/aggregation is not replaced.
- Overheads are omitted analytically but must be documented/accounted separately.
- No wireless accuracy or convergence improvement is established by the draft.

## Source-preserving adaptations recommended by this review

1. Retain raw float32 precision/omega on the wire. Treat CIFAR's score floor as a
   declared surrogate, not optimizer/posterior clipping.
2. Use byte-padded bitmap lengths in the executable physical cost model; keep the
   manuscript's ideal d-bit formula as its disclosed analytical approximation.
3. New versioned mixed sparse/dense packet protocol, leaving legacy v1 unchanged.
4. Keep counts as integers, never recover them through keep_ratio multiplication.
5. Separate wireless controller/state and explicit client-local persistence.
6. Keep legacy byte experiments intact; first wireless study stops on modeled
   time/round limits, not an implicit combination of byte and time constraints.
7. First support only clean completed-round resume. Record failed attempts without
   inventing progress or dropping costs.
8. Use common trace/seed/topology and fixed-slot timing for policy comparison;
   label modeling approximations and measure overhead sensitivity separately.
9. Update source, tests, user commands, output schemas and handoff docs together.

## Still unset; do not invent as locked experiment values

Total uplink/downlink Hz; reference gain, path-loss exponent and distances;
UL/DL PSD/SNR convention and power limits; noise PSD; channel correlation and seed;
modeled device throughput/local durations; KL-sort-time treatment; target seconds;
control allowance/framing; objective/price/feasibility tolerances; finite query cap;
missed-slot/abort rules; exact FedAvg control clipping; whether random controls
replay count/grant traces or recompute on their own evolving states.

The proposal allows a zero-overhead first analytical experiment. That is not a
claim that control signaling costs nothing. An equal-bandwidth baseline can be
infeasible when joint allocation is feasible; do not silently force it to run.

## Existing numerical experiment locks remain intact

MNIST primary FOLA: CNN16-32-FC128 (80,202), N20, fixed one label and 10 samples/client,
seed0, E5/B10, SGD LR .10 constant, momentum/decay0, lambda .01,
paper_reference, no gradient clipping, strict scoring, initial omega1.

CIFAR primary FOLA: BasicCNN (878,538), alpha .01, local fraction1, N20, seed0,
E10/B32, SGD LR .02 cosine horizon400/min.0001, momentum/decay0, lambda.01,
paper_reference, clip10, initial omega0, sparse floor_for_score=1e-12.

The selected historical FedAvg curves remain clipped MNIST LR.10 and clipped
CIFAR LR.01 constant. They are not a matched optimizer ablation. No historical
output may be overwritten or relabeled as a wireless result.

## Next engineering unit

Implement W1's independent payload/scoring/oracle/allocator layer and tests, with
explicit adaptation docs. Then W2's packet/config compatibility and W3's real
multi-phase persistence. Do not launch a full GPU queue before those gates pass.
