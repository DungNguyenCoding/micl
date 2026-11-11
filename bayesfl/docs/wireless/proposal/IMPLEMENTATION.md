# Suggested implementation direction

This is an integration plan, not a replacement for the existing FedKLS learner. The mathematical equation numbers are in the compiled manuscript. Do not implement the client training procedure from scratch while evaluating a new allocator.

## 1. Configuration and units

Use K clients, d model coordinates, b bits per mean/variance (e.g. 32), B uplink Hz, B_DL downlink Hz, and times in seconds. Keep the common round target, local training settings, and posterior estimator fixed for policy comparisons.

Maintain separate variables for:
- `round_target_s`: the fixed shared completion target;
- `local_time_s[k]`: each client's full local preparation time;
- `dl_time_s`: multicast duration;
- `upload_window_s = round_target_s - dl_time_s - max(local_time_s)`;
- `spectral_efficiency[k]`: positive, constant during the current bandwidth allocation;
- `keep_count[k]`: integer in [1,d], never a percentage-menu index;
- `bandwidth_hz[k]`: L(keep_count)/(upload_window_s * spectral_efficiency).

Start with fixed client locations and block-Rayleigh fading. Use linear-scale path gains. Under the document's PSD convention:

    gain[k,t] = beta[k] * abs(h[k,t])**2
    snr[k,t] = psd_w_per_hz[k] * gain[k,t] / noise_w_per_hz
    c[k,t] = log2(1 + snr[k,t])

Do not simultaneously keep total power fixed as bandwidth changes. Log the implied power `psd_w_per_hz[k] * bandwidth_hz[k]` and energy `power_w[k] * upload_window_s`. For automatic compliance with an intended total-power limit, a conservative sufficient setting is `psd_w_per_hz[k] * B <= Pmax_w[k]`.

## 2. Suggested interfaces

The following signatures describe responsibilities, not installed library APIs:

    payload_bits(m: int, d: int, b: int) -> int
    prepare_scores(global_posterior, local_posterior) -> ClientScoreState
    local_price_response(state, price, weight, U, c, b) -> ClientResponse
    allocate_price(client_oracles, weights, U, c, B, settings) -> Allocation
    serialize_sparse_posterior(state, local_posterior, m) -> Packet
    reconstruct_and_aggregate(global_posterior, decoded_packets, weights)

`ClientScoreState` holds sorted scores, coordinate indices, and tail sums. No full score vector needs to be exposed to the server. `ClientResponse` contains the selected integer count and its discarded-KL scalar. The server can derive payload and required bandwidth from the count.

`Allocation` should expose counts, realized ratios, bandwidths, required total bandwidth, objective, lower bound, objective-gap diagnostic, query count, and termination status. Use statuses such as `all_dense`, `zero_distortion`, `gap_target`, `price_tolerance`, `iteration_limit`, or `infeasible`. A price tolerance is not an integer optimality certificate.

## 3. Local preparation

Keep original full-model training and curvature estimation. Compute the Gaussian coordinate KL once; use strictly positive variances with identical damping in all baselines. Sort descending, retaining original coordinate indices for serialization. Construct tails using reverse cumulative summation:

    tail[m] = sum(sorted_scores[m:])  # zero-based indexing
    tail[d] = 0

Use float64 for scores, tails, and allocator arithmetic initially, even if transmitted posterior values use float32. This reduces but does not mathematically eliminate rounding error.

For a supplied price lambda, set the threshold `theta = 2*b*lambda/(pi*U*c)`. Count scores strictly exceeding theta and clip only to the sparse interval [1,d-1]. Compare this sparse candidate's *complete priced objective* with the dense candidate. Resolve exact ties by smaller required bandwidth. Do not skip the bitmap in this comparison.

A binary search on sorted scores finds the threshold count in O(log d); evaluating the tail and dense cost then takes O(1). Selection does not trigger retraining.

## 4. Server price coordination

1. Validate inputs and calculate the minimum-bandwidth feasible endpoint vector by comparing m=1 with m=d for each client. If even that vector exceeds B, return infeasible.
2. Keep that vector as a feasible incumbent. Add the Version A vector if feasible and its frozen-round objective is lower.
3. If all-dense demand fits, return it with zero reconstruction cost.
4. Query lambda=0. Exact local minimizers have zero reconstruction cost; if that response fits, return it.
5. Otherwise bracket the demand transition by doubling a positive high price until feasible, with a finite query limit and overflow checks. Keep recording bounds and feasible responses.
6. Bisect the bracket. At each trial, request exact local count/cost responses, compute total required bandwidth, the response objective, and the Lagrangian lower bound. Move the low endpoint up for infeasible demand and move the high endpoint down for feasible demand.
7. Maintain the best feasible objective across *all* queries. Stop on the objective gap, a price-width tolerance, or the iteration limit.
8. Optionally perform a bounded number of feasible additions/dense switches. Do not call this optional greedy refinement globally optimal. Use actual piecewise payload differences at every switch.
9. Recover bandwidth exactly from the feasible retained counts. Never return an infeasible final trial instead of the incumbent.

The dual lower bound is valid only when local priced minima are evaluated exactly or their errors have certified bounds. An arbitrary tolerance that changes a local response can make the computed “lower bound” invalid. In ordinary floating-point experiments, label the gap a numerical diagnostic unless error control justifies a formal certificate.

## 5. End-to-end round order

    multicast global posterior
    -> parallel full local training and score preparation
    -> wait for all required current information
    -> price-response exchanges and allocation
    -> bandwidth grants
    -> concurrent posterior uploads
    -> prior-fill reconstruction and original weighted Gaussian-product aggregation

The code may execute clients sequentially on one GPU, but the network model uses the maximum modeled local duration, not the sum of emulator wall-clock times. Keep these two times separate.

If a report contains an integer count and one floating scalar per query, its control payload is small but not zero. Model transmission/framing and repeated-message latency separately; a Python function-call time is not a network latency measurement. Reserve a control allowance before optimization when an end-to-end hard target is required.

## 6. Unit and integration tests

- Count 1 and count d; d=2 as a boundary case.
- All zero scores and repeated threshold scores.
- Sparse versus dense packet cost, including near-dense sparse packets larger than dense.
- Dense feasible, minimum feasible, and infeasible budgets.
- Compare each local price response with direct evaluation of every count on small d.
- Verify nonincreasing actual bandwidth demand across increasing prices.
- On tiny K,d only, enumerate all count combinations to check feasibility and objective bounds. This is a unit-test oracle, not the deployed solver.
- Verify every output has integer counts in [1,d], sum(B_k)<=B, and L_k/(B_k*c_k)=U within declared numerical tolerance.
- Reconstruct packets and compare Gaussian aggregation to existing FedKLS at identical masks and posteriors; full retention must reproduce the dense reference.
- Do not renormalize client aggregation weights per coordinate.

## 7. Experimental progression

First replay frozen local posteriors across channel states and budgets. This isolates allocator quality from divergent learning trajectories. Then run complete training with matched initialization, data partition, optimizer settings, channel draws, and prescribed round targets.

Compare feasible equal-bandwidth adaptive retention, feasible uniform retention, and the joint allocator. Compare random coordinate selection at *matched allocated counts and bandwidths*. For each policy, log discarded KL, counts, bandwidth, time/energy accounting, control queries, gaps, deadline failures, accuracy, and calibration.

Initializing the incumbent with a feasible Version A count vector prevents a worse frozen-round reconstruction objective. It does not guarantee higher multi-round test accuracy. Under fixed round lengths, the principal hypothesis is better learning within the same time budget, not necessarily faster individual rounds.
