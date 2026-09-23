# Implementation guideline: bitmap-based sparse posterior communication for FOLA

**Purpose:** Hand this document to the implementation chat together with the existing source repository. Extend the working FOLA and FedAvg experiments with three sparse-FOLA variants and communication-budget-aware evaluation.

**Status:** This is an implementation specification for the proposed method, not a report of completed repository changes or benchmark results. The repository has not been inspected in preparing this handoff. Preserve the existing baseline, inspect the actual code and dependency versions, and identify concrete edit locations before making changes.

**Confirmed by the researcher:** The communicated mean and covariance-related arrays currently use `float32`. The requested sparse representation uses a **packed bitmap**, not an index list. The main communication comparison includes **both server-to-client download and client-to-server upload**.

> Implement communication sparsification **after full local FOLA training**. Do not prune the model, change the local training loss, freeze omitted weights during training, or replace FOLA's Gaussian-product aggregation with FedAvg.

---

## 1. Required experiment variants

Retain the two existing baselines and add the following three variants. Use explicit direction names rather than ambiguous labels such as “forward KL” or “reverse KL.”

| Method identifier | Local training | Upload selection | Server aggregation |
|---|---|---|---|
| `fedavg_dense` | Existing FedAvg trainer | Existing dense upload | Existing FedAvg aggregator |
| `fola_dense` | Existing FOLA trainer | All posterior coordinates | Existing FOLA aggregator |
| `fola_sparse_kl_global_local` | Same FOLA trainer | Top scores from `KL(q_global_before || q_local_after)` | FOLA on reconstructed client distributions |
| `fola_sparse_kl_local_global` | Same FOLA trainer | Top scores from `KL(q_local_after || q_global_before)` | Same reconstructed-distribution aggregator |
| `fola_sparse_random` | Same FOLA trainer | Uniform random subset with exactly the same number of coordinates | Same reconstructed-distribution aggregator |

For a given keep ratio, the three sparse variants must use the same message format, number of selected coordinates, numerical policies, reconstruction rule, and client weighting. Only the selection rule changes.

**Experiment hypothesis, not an assumed result:** Global-first KL may retain useful changes because its mean-change component uses local posterior precision, which also enters FOLA aggregation. However, neither KL order is established as universally superior. Local-first KL also depends on local variance through its variance-change component. The experiments must fairly measure all three rules.

**Outside the initial implementation:** subnet training, permanent pruning, sparse downlink, quantization, error feedback, adaptive keep ratios, mask reuse across rounds, client scheduling based on scores, symmetric KL, and changes to FOLA's posterior estimator. Do not silently introduce any of these.

## 2. Inspect and preserve the working baseline

Before editing, locate the existing client training function, posterior extraction/loading functions, global aggregation strategy, serializer, client sampler, evaluation path, configuration system, checkpointing, and plotting/logging code. Record their actual paths in the implementation report; module names in this guideline are suggestions, not claims about the repository.

### 2.1 Verify the statistical meaning of each array

Use the following canonical notation at the compression boundary:

- `mu`: posterior mean for each modeled scalar weight.
- `var`: posterior variance, equal to the diagonal of the covariance matrix.
- `precision`: reciprocal variance, `1 / var`.
- `std`: standard deviation, `sqrt(var)`.

The variable name `cov`, `sigma`, or `fisher` in source code is not enough to establish its meaning. Trace how it is produced and used. If the baseline stores precision, convert it correctly when evaluating variance-form KL scores. Do not pass precision, standard deviation, raw Fisher accumulators, or a variational `rho` parameter into a formula expecting variance.

FOLA uses a Gaussian-product server update in paper 02, Eq. (12), PDF p. 6. Its local prior penalty is in Eqs. (15)–(16), PDF p. 7, and its online curvature/precision construction is in Section III-D, including Eqs. (22)–(24), PDF pp. 7–8 [FOLA]. Preserve the working implementation of those mechanisms. If the code and paper disagree, document the discrepancy instead of silently “fixing” it as part of the sparsification experiment.

The main specification below sends selected **means and variances**. If retaining an existing precision-based message is necessary to reproduce the baseline, make that an explicit, shared protocol choice for dense and sparse FOLA, identify it in metadata, and adapt the reciprocal operations. One `float32` precision uses the same payload bytes as one `float32` variance, but they are not statistically interchangeable.

### 2.2 Define the common coordinate layout

Let `d` be the number of scalar weights with a modeled mean–variance pair. It is not the total number of mean plus variance entries; that total is `2*d`.

Create one deterministic layout manifest containing tensor name, shape, flattened offset, scalar count, and representation. Use the same manifest on every client and the server, and attach a `layout_id` to messages. Convolution kernels can be flattened for communication and reshaped afterward without changing their role in the network. Weight tying/shared storage must not create inconsistent duplicate coordinates.

Use **one model-wide bitmap of length `d`** and global selection over these `d` coordinates. Do not silently switch to a separate keep ratio per layer.

Audit buffers and non-posterior parameters separately. In particular, BatchNorm activation statistics are not weight-posterior variances. Preserve the baseline's handling of auxiliary state and record its transmitted bytes separately. If some parameters are excluded from sparsification, document this policy, use `d` for the eligible posterior coordinates, and add the excluded transmitted state to the cost ledger.

### 2.3 Freeze everything unrelated to compression

Preserve local epochs/steps, learning rates, optimizer state policy, prior regularization strength, online precision estimator, damping, initialization, data partition, augmentation, evaluation semantics, and client-weighting policy. Changing a score must not change the training loss.

Save an immutable copy of the broadcast posterior **before local training**. This reference changes every global round: it is the previous global posterior, not the model's original initialization.

## 3. Mathematical definition of the three selection rules

For global round `t`, client `k`, and coordinate `i`, define

$$
q_{0,i} = q_{G,i}^{t-1} = \mathcal N(\mu_{0,i},v_{0,i}),
\qquad
q_{k,i} = q_{k,i}^{t} = \mathcal N(\mu_{k,i},v_{k,i}),
\qquad v=\sigma^2>0.
$$

The server broadcasts `q_0`; the existing FOLA trainer produces `q_k`. Define

$$
\Delta\mu_{k,i}=\mu_{k,i}-\mu_{0,i}.
$$

Let `r` be the **keep ratio**, not the drop ratio:

$$
0<r\le1,\qquad m=\lceil rd\rceil,\qquad r_{\mathrm{actual}}=m/d.
$$

Use a documented, consistent ceiling operation, keep exactly `m` coordinates, and log both requested and realized ratios. A zero-keep mask is useful in unit tests but is not a normal training configuration.

### 3.1 Proposed rule: global first, local second

Configuration value: `kl_global_local`.

$$
D^{G\|L}_{k,i}
=\operatorname{KL}(q_{0,i}\|q_{k,i})
=\frac12\left[
\log\frac{v_{k,i}}{v_{0,i}}
+\frac{v_{0,i}+(\Delta\mu_{k,i})^2}{v_{k,i}}
-1\right].
$$

Select the `m` **largest** scores. The mean-change term is

$$
\frac{(\Delta\mu_{k,i})^2}{2v_{k,i}}
=\frac{(\Delta\mu_{k,i})^2}{2\sigma_{k,i}^2}.
$$

It measures mean movement relative to the client's new uncertainty. This is the direction previously recommended in the proposal.

### 3.2 Direction-reversed ablation: local first, global second

Configuration value: `kl_local_global`.

$$
D^{L\|G}_{k,i}
=\operatorname{KL}(q_{k,i}\|q_{0,i})
=\frac12\left[
\log\frac{v_{0,i}}{v_{k,i}}
+\frac{v_{k,i}+(\Delta\mu_{k,i})^2}{v_{0,i}}
-1\right].
$$

Select the `m` largest scores. Its mean-change term uses the old global variance:

$$
\frac{(\Delta\mu_{k,i})^2}{2v_{0,i}}.
$$

Reverse the **entire KL formula**, not just this denominator. Both mean changes and variance changes matter in both directions. When `v_k == v_0` coordinate-wise, the two scores coincide.

**Do not confuse the communication score with a VI training regularizer.** Paper 01's Eq. (1) contains posterior-first/prior-second KL as part of variational free energy [BBB]. Here both distributions are fixed after local training, and the score is used only to select a message. It is not added to the loss, differentiated, or used to change the existing FOLA optimizer.

### 3.3 Random sparse control

Configuration value: `random`.

Sample exactly `m` distinct positions uniformly without replacement from the same eligible `d` coordinates. Generate a fresh subset for each client and round. Send a bitmap just as for the KL variants; do not replace the bitmap with a shared random seed in this comparison.

Use a dedicated mask RNG derived reproducibly from `(experiment_seed, client_id, round_id)`. Do not let selection consume the RNG stream used by client sampling, minibatch shuffling, data augmentation, or local training. Avoid process-randomized hashes when constructing seeds.

This is a **random communication mask**, not FedDrop subnet training. There is no `1/r` rescaling, inverted-dropout factor, or Bernoulli mask with a merely approximate cardinality.

### 3.4 Selection and numerical requirements

Compute one score per coordinate, not a single KL sum for the entire model. Select globally over the eligible coordinates. Break score ties deterministically, for example by smaller global coordinate index, so identical posteriors do not produce irreproducible masks.

Scoring may use `float64` arithmetic for stability while the payload remains `float32`. This does not change the communication bit width. Keep the chosen scoring precision the same in both KL variants and record it.

For a stable form of `KL(P || Q)`, set `s = log(var_P) - log(var_Q)` and use

$$
\operatorname{KL}(P\|Q)
=\frac12\left[\operatorname{expm1}(s)-s
+\frac{(\mu_P-\mu_Q)^2}{v_Q}\right].
$$

Use the baseline's documented positivity/damping policy consistently. Reject NaN, infinity, and invalid variances rather than silently assigning low scores. Clamp only tiny negative score roundoff to zero after checking it is numerical error. Do not introduce new clipping that changes posterior values without applying and reporting the same policy in the dense reference. Validate the final `float32` representation as well as higher-precision intermediates.

## 4. Client-to-server packet: packed bitmap and selected pairs

The agreed wire-level model representation is

```text
metadata:
    protocol_version
    round_id / base_snapshot_id
    client_id, when not already provided by the transport
    layout_id
    covariance_representation = "variance"
    d, m, when not already known from the round configuration
    num_examples = n_k, when not already carried by the FL API

model arrays:
    bitmap       uint8[ceil(d / 8)]
    mean_values  float32[m]
    var_values   float32[m]

auxiliary arrays:
    only the state already required by the baseline, with a documented policy
```

`n_k = |D_k|` is the local training sample count used by the baseline's aggregation weighting. It is not the number of transmitted coordinates and not the sample count multiplied by local epochs.

### 4.1 Ordering and packing contract

Let `M` be the logical mask. Order transmitted pairs by **ascending selected coordinate index**, not by descending KL score:

```python
selected = np.flatnonzero(mask)  # Ascending positions; local only.
bitmap = np.packbits(mask, bitorder="little")
mean_values = np.ascontiguousarray(local_mu[selected], dtype=np.float32)
var_values = np.ascontiguousarray(local_var[selected], dtype=np.float32)
```

`selected` is not transmitted. The server recovers it from the bitmap. NumPy packs bits into `uint8` and zero-pads the last byte; use the same `bitorder` for unpacking [NP-PACK, NP-UNPACK].

A normal boolean array is not a packed bitmap. A length-`d` `uint8` array containing zeros and ones also wastes one byte per coordinate. The transmitted bitmap must contain exactly

$$
L_d=\left\lceil d/8\right\rceil
$$

bytes. Send the bitmap, selected means, and selected variances as separate typed arrays; concatenating them can upcast the bitmap and invalidate the cost model.

Do **not** send the full score vector, a separate index array, a dense zero-filled model, or a dense model “for debugging” through the training transport. Local debug files are acceptable but are not communication messages.

### 4.2 Tiny packet example

For `d=4`, let the logical mask be `[1, 0, 1, 0]` and use little bit order. The packed bitmap is one byte with value `5` (`0x05`). A message might contain

```text
bitmap:       uint8([5])
mean_values:  float32([0.70, 0.70])
var_values:   float32([0.18, 0.25])
```

The first pair belongs to coordinate 0; the second belongs to coordinate 2. Coordinates 1 and 3 are omitted. The numerical arrays require `8*m = 16` bytes and the bitmap requires one byte, so this upload has **17 model-payload bytes**, excluding metadata and serializer headers.

### 4.3 Full-retention policy

For the first implementation, use `bitmap_always`: even at `r=1`, a sparse variant sends the bitmap. This preserves one packet format and makes byte accounting explicit. Dense FOLA sends no bitmap.

At `r=1`, sparse and dense aggregation should agree numerically, but sparse communication has the bitmap overhead. Do not claim identical communication cost. A future explicit dense-format fallback may remove this overhead, but it is not the default in this specification.

## 5. Server reconstruction and aggregation

### 5.1 Decode without accessing hidden local values

The server retains an immutable copy of the exact posterior broadcast in round `t`. For each accepted packet:

1. Validate protocol, round/snapshot, client identity, layout, and covariance representation.
2. Check the bitmap length is `ceil(d/8)`, its dtype is `uint8`, and unused trailing bits are zero.
3. Recover exactly the first `d` bits and verify the number of ones equals `m`.
4. Check both value arrays have length `m`, dtype `float32`, finite means, and finite strictly positive variances.
5. Reconstruct the contribution using only the packet and the fixed broadcast snapshot.

```python
# Assumes packet validation has already succeeded.
mask = np.unpackbits(bitmap, bitorder="little", count=d).astype(bool)
mu_effective = global_mu_before.copy()
var_effective = global_var_before.copy()
mu_effective[mask] = mean_values
var_effective[mask] = var_values
```

Do not reconstruct omitted positions from the client's full locally trained model, even when a simulator can access it in the same process. That would leak information not present in the communication protocol.

The proposed effective distribution is

$$
\widetilde q_{k,i}=
\begin{cases}
\mathcal N(\mu_{k,i},v_{k,i}),&M_{k,i}=1,\\
\mathcal N(\mu_{0,i},v_{0,i}),&M_{k,i}=0.
\end{cases}
$$

**Omission means “no new contribution from this client-coordinate,” not zero weight, zero variance, removal of the client, or reuse of that client's older personal model.** Another client can still transmit the same coordinate and change the global result.

### 5.2 Apply FOLA Eq. (12), not arithmetic variance averaging

Let `S_t` be the set of valid client results used for aggregation. Preserve the baseline weighting policy. For sample-weighted aggregation,

$$
\pi_{k,t}=\frac{n_k}{\sum_{j\in S_t}n_j},
\qquad \sum_{k\in S_t}\pi_{k,t}=1.
$$

With the reconstructed values, the coordinate version of FOLA Eq. (12) [FOLA] is

$$
v_{G,i}^{t}
=\left(\sum_{k\in S_t}\frac{\pi_{k,t}}{\widetilde v_{k,i}}\right)^{-1},
$$

$$
\mu_{G,i}^{t}
=v_{G,i}^{t}\sum_{k\in S_t}
\pi_{k,t}\frac{\widetilde\mu_{k,i}}{\widetilde v_{k,i}}.
$$

Client weights are normalized **once across accepted clients**, not separately across the clients that sent each coordinate. Do not multiply by another copy of the old global prior. Do not average the variances arithmetically. Do not apply a `1/r` correction.

First implement reconstruction and invoke the existing dense FOLA aggregator, preserving its numerical policy. Once verified, an equivalent sparse accumulator can avoid materializing a full vector for every client, but equivalence must be tested. Transmitting changes in precision/natural parameters is not required for this first version.

Broadcast the resulting **full** global mean–variance vectors for the next round. Preserve the baseline policy for resetting or retaining local optimizer/auxiliary state between rounds.

### 5.3 Required invariants

- With all mask entries equal to one, obtain dense FOLA's result for the same local posterior inputs, within a documented floating-point tolerance.
- If all clients omit coordinate `i`, its global mean and variance are unchanged. Preserve these entries directly from the snapshot if exact equality is required.
- If a single client omits a coordinate, that client's contribution equals the old global distribution; other clients may still change that coordinate.
- Positive input variances and nonnegative normalized client weights produce positive aggregate precision.
- The three sparse methods differ only through their masks, not through reconstruction or aggregation.

## 6. Integration into the existing Flower source

Do not migrate Flower versions or APIs merely to add this experiment. Inspect the project's installed version and use its current extension points. Suggested responsibilities are:

| Responsibility | Implementation action |
|---|---|
| Posterior adapter | Extract/load aligned means and true diagonal variances, with explicit conversion from any precision storage. |
| Selection function | Implement the two coordinate KL scores and exact-cardinality random selection. |
| Packet codec | Pack the one-bit mask, select sorted values, validate/decode, and reconstruct missing values. |
| Client fit path | Snapshot broadcast posterior, run unchanged local FOLA training, then call the selector and encoder. |
| FOLA server strategy | Decode sparse replies, reconstruct, and call the Gaussian-product aggregator. |
| Communication ledger | Count uploads, downloads, auxiliary state, and optional serialized sizes at logical message boundaries. |
| Experiment runner | Support both round-limited and communication-budget-limited execution. |
| Evaluation/plotting | Log global accuracy with cumulative bytes and retain the existing accuracy-versus-round output. |

For a legacy `NumPyClient` path, `fit()` returns arrays, a training-example count, and metrics [FL-CLIENT]. A possible integration fragment is:

```python
# Fragment only: use the repository's existing fit() signature and metadata.
return [bitmap, mean_values, var_values], int(n_k), {
    "protocol": "sparse_fola_bitmap_v1",
    "round_id": int(round_id),
    "layout_id": layout_id,
    "covariance_representation": "variance",
}
```

Adapt the existing FOLA aggregation callback (commonly `aggregate_fit` in legacy code). For a Message/ServerApp implementation, use its corresponding records and aggregation hook rather than mixing the two APIs [FL-STRATEGY]. Array position in different clients' packets does **not** identify the same model coordinate; the bitmap must be decoded first. The default FedAvg reducer must never aggregate packed sparse arrays directly.

Flower's NumPy serializer uses array serialization, so array bytes and serialized message bytes are different quantities [FL-SERIALIZER]. Test the actual repository's serializer round trip to ensure the bitmap remains `uint8`, the values remain `float32`, and no unintended dense arrays are included.

## 7. Communication cost: equations to implement

### 7.1 Counting convention

Count one full download to **each participating recipient** and one upload from each replying client. This is aggregate logical client-server communication, not a single physical radio broadcast and not an estimate of communication latency.

The closed-form equations below count the posterior/model arrays and bitmap only. They exclude metadata, serializer headers, transport framing, evaluation messages, retries, and auxiliary state. Account for those separately where relevant; do not label the closed-form payload as measured on-wire network traffic.

Let

$$
K_t=\text{number of participating clients in round }t,
\quad d=\text{number of posterior weight coordinates},
\quad b_\mu=b_v=32.
$$

`b_v` is the earlier `b_sigma`: bits for one variance value, not bits for a standard deviation plus another scalar. Each mean–variance pair occupies 64 bits = 8 bytes.

### 7.2 Main overview equations: one global round

With all participating clients completing the round, uniform keep ratio `r`, and ignoring rounding/byte padding:

**Original FOLA**

$$
\boxed{
C_{\mathrm{FOLA},t}
=K_t[\underbrace{64d}_{\text{full download}}
+\underbrace{64d}_{\text{full upload}}]
=128K_td\quad\text{bits}.
}
$$

**Proposed sparse FOLA with a bitmap**

$$
\boxed{
C_{\mathrm{sparse},t}
=K_t[\underbrace{64d}_{\text{full download}}
+\underbrace{64rd}_{\text{selected mean--variance pairs}}
+\underbrace{d}_{\text{bitmap}}]
=K_td(65+64r)\quad\text{bits}.
}
$$

Use these formulas in paper-level summaries, with their assumptions stated. They are derived from the agreed payload specification, not equations quoted from FOLA.

### 7.3 Exact packed-array byte counts for the code

Use actual `m = ceil(r*d)` and `L_d = (d+7)//8`:

| One client's transfer | Posterior/model-array payload in bytes |
|---|---:|
| Dense FOLA download | `8*d` |
| Dense FOLA upload | `8*d` |
| Sparse FOLA download | `8*d` |
| Sparse FOLA upload | `8*m + L_d` |

Therefore,

$$
\boxed{B_{\mathrm{FOLA},t}=16K_td\quad\text{bytes}}
$$

and

$$
\boxed{B_{\mathrm{sparse},t}=K_t(8d+8m+\lceil d/8\rceil)\quad\text{bytes}.}
$$

Multiplication by 8 converts either byte count into bits. Do not multiply by 8 twice. Use Python integers or sufficiently wide integer counters, never `float32`, for cumulative byte totals.

For differing client keep counts, replace `K_t * 8*m` with the sum of `8*m_k`. For incomplete participation, distinguish download recipients `R_t` from clients whose upload messages actually arrive `U_t`:

$$
B_{\mathrm{sparse},t}^{\mathrm{core}}
=\sum_{k\in R_t}8d
+\sum_{k\in U_t}(8m_k+\lceil d/8\rceil).
$$

An upload that arrives but is rejected still consumed communication; its client need not belong to the aggregation set `S_t`. If partial packets or retries are not observable, report the modeled accounting convention rather than guessing physical byte counts.

### 7.4 FedAvg baseline

For ordinary dense FedAvg exchanging `d_F` scalar `float32` weights in both directions,

$$
B_{\mathrm{FedAvg},t}=8K_td_F\quad\text{bytes},
\qquad
C_{\mathrm{FedAvg},t}=64K_td_F\quad\text{bits}.
$$

Use the actual FedAvg transmitted tensors and their dtypes. Do not charge FedAvg for a covariance that it does not send. Do not assume `d_F == d` without checking the parameter layout and treatment of buffers.

### 7.5 Auxiliary state and payload measurement

If the model also transmits dense non-posterior parameters or buffers, add their actual bytes:

$$
B_{\mathrm{training},t}^{\mathrm{arrays}}
=B_{\mathrm{training},t}^{\mathrm{core}}
+B_{\mathrm{training},t}^{\mathrm{aux}}.
$$

Measure array payload from the final typed arrays:

```python
upload_core_bytes = (
    int(bitmap.nbytes)
    + int(mean_values.nbytes)
    + int(var_values.nbytes)
)
expected_upload_core_bytes = 8 * m + (d + 7) // 8
assert upload_core_bytes == expected_upload_core_bytes
```

Metadata such as `n_k`, round ID, representation, and layout ID is excluded from this core formula. It is not free on the wire: include it in a separately measured whole-message metric when that measurement is available.

Maintain an optional **serialized** ledger as well. In a NumPy/Parameters implementation, summing the lengths of serialized tensor byte strings measures serialized tensor storage, including its array headers but not necessarily the complete message envelope. Give this metric a specific name. Only call a count “on-wire bytes” if the transport layer is actually measured. Do not estimate communication from Python object memory size or checkpoint file size.

### 7.6 Savings and why more rounds may fit

In the ideal uniform setting,

$$
\frac{C_{\mathrm{sparse}}}{C_{\mathrm{FOLA}}}
=\frac{65+64r}{128},
\qquad
R_{\mathrm{total}}=1-\frac{65+64r}{128}.
$$

At a fixed total communication budget and constant round costs, the approximate round-count multiplier is

$$
\boxed{\frac{T_{\mathrm{sparse}}}{T_{\mathrm{FOLA}}}\approx\frac{128}{65+64r}.}
$$

| Keep ratio | Sparse/dense total payload | Total payload reduction | Approximate rounds at the same budget |
|---|---:|---:|---:|
| `r=0.50` | `97/128 = 0.7578125` | `24.21875%` | `1.3196` times |
| `r=0.10` | `71.4/128 = 0.5578125` | `44.21875%` | `1.7927` times |

Do not use `1/r` as the total-round multiplier: the downlink is still dense. More affordable rounds do not guarantee better accuracy and also require more local computation.

With a compulsory bitmap, very high retention can increase cost. Ignoring rounding, positive savings require `r < 63/64 = 0.984375`. At `r=1`, the sparse payload is `129*K*d` bits versus dense FOLA's `128*K*d` bits. This is expected with `bitmap_always`, not a counting bug.

## 8. Communication ledger and cumulative accuracy logging

### 8.1 Record events, not only a constant multiplied by round number

Introduce a central ledger at the logical client-server message boundary. Record download when a model message is dispatched and upload when the reply arrives. Do not count a message both when serialized and when deserialized. Cache reads, internal tensor copies, packing/unpacking, and local checkpoint writes are not additional network transfers.

The following event fields are suggested:

```text
run_id, method, seed, round_id, model_version
client_id, message_id, attempt_id
phase: initialize | fit | evaluate
direction: downlink | uplink
status: dispatched | received | rejected | failed
layout_id, d, m, keep_ratio_requested, keep_ratio_actual
mean_bytes, covariance_bytes, bitmap_bytes, auxiliary_array_bytes
array_payload_bytes
serialized_tensor_bytes          # Optional; null when unavailable.
serialized_message_bytes         # Optional; null when unavailable.
accounting_kind: logical_payload | measured_transport
```

Use each unique transfer/attempt once. Retransmissions are additional communication if they occur and are observed. In the initial no-failure simulator, logical packet accounting is sufficient; disclose that it is modeled client-server communication rather than observed physical traffic.

### 8.2 Required cumulative counters

At minimum, persist:

```text
train_uplink_array_bytes
train_downlink_array_bytes
train_total_array_bytes
initialization_array_bytes
evaluation_array_bytes
all_array_bytes
```

Each also needs a cumulative counterpart. Define

$$
B_{\mathrm{train,cum}}(T)
=\sum_{t=1}^{T}(B_{t}^{\mathrm{train,up}}+B_{t}^{\mathrm{train,down}}),
$$

$$
B_{\mathrm{all,cum}}
=B_{\mathrm{init}}+B_{\mathrm{train,cum}}+B_{\mathrm{eval,cum}}.
$$

Count a first-round download once. If an additional initialization/model query actually crosses the client-server boundary, record it separately instead of assuming it is free or counting the first download twice.

If evaluation runs centrally without client communication, additional evaluation communication is zero. If the evaluation callback sends the model to clients again, log that download and any returned arrays. Scalar metric/control traffic belongs to the whole-message ledger when measured. Do not quietly exclude extra evaluation downloads from a plot labeled “all model communication.”

### 8.3 Global evaluation record

After round `t` is aggregated, evaluate the corresponding global model and save its accuracy with the cumulative cost incurred up to that evaluated state:

```text
run_id, method, selection_rule, seed, round_id, model_version
keep_ratio_requested, keep_ratio_actual
d, num_download_recipients, num_upload_replies, num_aggregated_clients
round_train_uplink_array_bytes
round_train_downlink_array_bytes
round_train_total_array_bytes
cumulative_train_uplink_array_bytes
cumulative_train_downlink_array_bytes
cumulative_train_total_array_bytes
cumulative_initialization_array_bytes
cumulative_evaluation_array_bytes
cumulative_all_array_bytes
cumulative_serialized_message_bytes   # Optional; null when not measured.
global_accuracy, global_loss
local_training_steps_or_examples_seen
round_wall_time_seconds, selection_time_seconds
budget_metric, budget_bytes, stop_reason
```

Use a fixed accuracy unit, such as a fraction in `[0,1]`, in machine-readable logs. Convert to percent only when plotting. Save CSV or JSONL; do not rely only on console output.

If evaluation is performed every several rounds, maintain costs every round and attach the accumulated amount to each actual evaluation. Never reset the counter when a fresh evaluation interval begins.

Checkpoint the global mean and covariance/precision, optimizer-related state required by the baseline, true round index, RNG state/seed policy, client schedule, and communication ledger together. Resuming must not reset the budget or charge the same completed transfer again.

### 8.4 Optional local diagnostics

Useful diagnostics include omitted-score sums for both KL orders, score quantiles, mask overlap with the previous round, and selected counts by layer. Store full diagnostic arrays locally/offline rather than transmitting them. If scalar diagnostics are sent to the server, count them in the whole-message metric when available.

An offline replay test may examine saved full local posteriors to compare dense versus sparse aggregation. That is a diagnostic, not the operational sparse server: the live sparse aggregator must not access untransmitted values.

## 9. Fair accuracy-versus-communication experiments

### 9.1 Retain two x-axes

Keep the existing plot of global accuracy versus global round. Add:

- **Primary:** global accuracy versus cumulative uplink-plus-downlink array payload, including initialization and evaluation arrays when transmitted.
- **Secondary:** global accuracy versus cumulative training array payload only, clearly labeled.
- **Optional:** accuracy versus cumulative uplink-only payload and separately measured serialized communication.

Use `cumulative_all_array_bytes` as the default communication budget metric. In an experiment with no extra initialization/evaluation messages or auxiliary arrays, it reduces to the round-cost equations in Section 7.

Convert units consistently:

```text
MB  = bytes / 1_000_000
GB  = bytes / 1_000_000_000
MiB = bytes / 2**20
GiB = bytes / 2**30
bits = 8 * bytes
```

Do not put theoretical bit counts for one method and serialized byte counts for another on the same axis. Label the exact accounting level and units.

### 9.2 Support both round-limited and budget-limited runs

Keep `max_rounds` and add `max_communication_bytes` plus an explicit `budget_metric`.

For a strict array-payload budget in a no-failure fixed-layout run, calculate the next round's required download and upload payload, including planned auxiliary and evaluation arrays. Start the round only if its full cost fits:

$$
B_{\mathrm{cum}}+B_{\mathrm{next}}\le B_{\max}.
$$

Otherwise stop with `stop_reason = communication_budget` and evaluate the last completed model. If the remaining budget cannot fit another full round, do not change the client count or keep ratio merely to fill the budget.

For measured message/network accounting where the next cost is not known exactly, either reserve a justified upper bound or explicitly permit and flag an overshooting round. An accuracy checkpoint that exceeds the budget must not be reported as achieved within that budget.

For constant per-round cost and no extra communication,

$$
T_{\max}=\left\lfloor B_{\max}/B_{\mathrm{round}}\right\rfloor.
$$

For example, a budget equal to 100 dense-FOLA rounds allows approximately 131 complete sparse rounds at `r=0.5`, or 179 at `r=0.1`, under the ideal assumptions. Actual stopping must use the ledger, not these approximations.

### 9.3 Comparing accuracy at a common budget

For method `a` and budget `B`, use its latest evaluated checkpoint that does not exceed `B`:

$$
t_a(B)=\max\{t:B_{a,\mathrm{cum}}(t)\le B\},
\qquad A_a(B)=A_{a,t_a(B)}.
$$

Use the same evaluation/checkpoint policy for all methods. Plot observed points or a clearly labeled step function. Do not extrapolate accuracy beyond a run's completed training, borrow an over-budget checkpoint, or select the maximum test accuracy as though it were the current model. Budgets must be jointly covered by the runs; use validation data for model/hyperparameter selection.

A baseline might have less than one round's worth of budget remaining while a sparse variant fits another round. That is the intended comparison, not a reason to force identical round counts.

### 9.4 Control confounders

Use matched initialization, data splits, local training settings, client-sampling schedule, evaluation rule, and seeds across FOLA variants. Pre-generate or reproducibly extend the client schedule so all methods share the same prefix when they run different numbers of rounds. Use multiple seeds for reported comparisons.

For the initial experiment, preserve the existing learning-rate and FOLA round-dependent precision schedules as functions of the true global round number. Do not secretly rescale them by communication. If a scheduler depends on a total-round horizon, make that dependence explicit and report the configured horizon; a communication-based schedule would be a separately labeled ablation.

Cheaper rounds do not automatically imply greater accuracy at the same byte budget. They also entail additional local optimization/data exposure, which should be logged rather than confused with computation savings.

## 10. Suggested configuration and implementation sequence

### 10.1 Configuration example

These field names are illustrative; adapt them to the repository's existing configuration system.

```yaml
method: fola_sparse
selection_rule: kl_global_local   # kl_global_local | kl_local_global | random
keep_ratio: 0.50
selection_scope: global_coordinates
selection_count_rounding: ceil
random_mask_sampling: without_replacement
random_mask_seed_policy: separate_seed_client_round
score_dtype: float64
payload_mean_dtype: float32
payload_covariance_dtype: float32
payload_covariance_representation: variance
mask_encoding: bitmap
bitmap_bitorder: little
bitmap_policy: always
missing_coordinate_policy: broadcast_posterior
sparse_downlink: false
error_feedback: false

# Preserve the existing trainer/estimator and document its damping policy.
local_training: existing_fola
aggregation: existing_fola_on_reconstructed_posteriors

communication:
  count_downlink_per_recipient: true
  primary_metric: cumulative_all_array_bytes
  track_train_up_down_separately: true
  track_serialized_message_bytes: optional
  max_communication_bytes: null   # Configure one common budget for budget runs.

max_rounds: 1000                 # Safety cap; not the comparison budget.
```

Do not interpret `score_dtype: float64` as 64-bit communication. Count the final `float32` mean and variance arrays that are actually encoded.

### 10.2 Implementation stages

1. **Baseline audit and golden run:** record current FOLA/FedAvg outputs, dtype and state layouts, seed behavior, and actual message contents. No algorithm changes yet.
2. **Accounting on dense baselines:** add ledgers and cumulative-communication evaluation columns before adding sparsity. Verify the formulas against encoded array sizes.
3. **Pure codec and score functions:** implement the three selectors, bitmap packing, validation, and reconstruction with unit tests independent of Flower.
4. **Sparse FOLA path:** add packet handling and reuse the baseline Gaussian-product aggregator. Verify full retention first, then mixed masks.
5. **Short controlled experiments:** run all three sparse variants at the same keep ratio and settings. Confirm their per-round payload counts agree.
6. **Budget-aware runner and plots:** support extra rounds when affordable and compare accuracy at common byte budgets.
7. **Report:** list modified files, runnable commands, tests executed, counting scope, numerical/representation decisions, and any unresolved baseline-paper mismatch.

Do not replace the implementation with a proposal-only response once the repository is available. Inspect, make the scoped changes, run the applicable tests, and report what actually ran.

## 11. Acceptance tests

### Mathematical and selection tests

- Identical global/local Gaussians give zero KL in both directions within numerical tolerance.
- Equal global/local variances give identical coordinate scores in both directions.
- Argument order is verified numerically. With `q0=N(0,1)` and `qk=N(1,0.01)`, expect `KL(q0||qk) ≈ 97.197414907` and `KL(qk||q0) ≈ 2.307585093`.
- A direction-sensitive ranking test uses two old coordinates `N(0,1)` and local coordinates `N(0,0.01)` and `N(3,1)`. At 50% retention, global-first selects the first coordinate; local-first selects the second.
- All three selectors keep exactly `ceil(r*d)` positions, including noninteger products and tied scores.
- Random selection is reproducible under its dedicated seed, does not sample duplicate coordinates, and does not advance training/client-sampling RNG streams. Do not require different seeds to always produce different masks, since coincidence is possible.

### Packet and aggregation tests

- Bitmap round trip works for `d` not divisible by 8; packed length is `(d+7)//8` and unused bits are zero.
- Selected values are in ascending coordinate order and recover the correct coordinates after decoding.
- The server reconstructs omitted values from the round's broadcast snapshot, not from zero or any hidden local state.
- All-selected reconstruction followed by aggregation matches dense FOLA for the same inputs.
- All-omitted coordinates remain unchanged, including their variances.
- Mixed masks agree with explicit reconstructed-distribution FOLA Eq. (12), using the original client weights.
- Validate stale snapshot IDs, incorrect layouts, wrong array lengths/dtypes, invalid variances, and malformed padding. Fail explicitly rather than silently accepting corrupt messages.
- The full global posterior remains available for the next dense download, and the client-side bitmap has not pruned or frozen the local model.

### Communication and evaluation tests

- `d=4, m=2`: bitmap is 1 byte, sparse upload is 17 bytes, download is 32 bytes, and combined core traffic is 49 bytes per client. Dense FOLA is 64 bytes per client. This checks padding rather than the ideal fractional-byte approximation.
- `d=1_000_000, K=10, r=0.5`: dense FOLA costs 160,000,000 bytes per round; sparse FOLA costs 121,250,000 bytes per round, with no extras.
- The same example at `r=0.1` costs 89,250,000 sparse bytes per round.
- For the same `d`, `m`, clients, and auxiliary state, all three sparse selectors have equal raw array payload cost, even when their actual masks differ.
- `r=1` preserves dense aggregation numerically but includes the compulsory bitmap overhead.
- Dense FedAvg is charged only for what it sends, not for Bayesian variances.
- Additional evaluation model transfers are recorded instead of hidden.
- Upload bytes and download bytes are summed, not sample-weight averaged; a message is not counted twice.
- Resume restores cumulative counters and round/schedule state without duplicate charges.
- Strict-budget mode never uses an over-budget checkpoint in a budget comparison and logs whether a run stopped on communication or the safety round cap.
- The accuracy-versus-communication plot reads cumulative counters from the recorded run, rather than multiplying all methods by the same dense round cost.

## 12. Reference snippets for the implementation chat

These small snippets specify the arithmetic and bitmap convention. They are not a substitute for integrating with and testing the actual repository.

### 12.1 Coordinate-wise Gaussian KL, with explicit argument order

```python
import numpy as np


def diagonal_kl(mu_p, var_p, mu_q, var_q):
    """Return per-coordinate KL(P || Q), where every variance is positive."""
    mp, vp, mq, vq = (
        np.asarray(x, dtype=np.float64)
        for x in (mu_p, var_p, mu_q, var_q)
    )
    if mp.ndim != 1 or mp.size == 0:
        raise ValueError("Posterior vectors must be nonempty and one-dimensional")
    if not all(x.shape == mp.shape for x in (vp, mq, vq)):
        raise ValueError("Posterior vectors must have the same shape")
    if not all(np.isfinite(x).all() for x in (mp, vp, mq, vq)):
        raise ValueError("Posterior contains NaN or infinity")
    if np.any(vp <= 0) or np.any(vq <= 0):
        raise ValueError("KL requires strictly positive variances")

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        s = np.log(vp) - np.log(vq)
        score = 0.5 * (np.expm1(s) - s + (mp - mq) ** 2 / vq)
    if not np.isfinite(score).all() or np.any(score < -1e-10):
        raise FloatingPointError("Invalid KL score; inspect posterior estimation")
    return np.maximum(score, 0.0)


# Proposed order:
# score = diagonal_kl(global_mu_before, global_var_before, local_mu, local_var)
# Direction-reversed ablation:
# score = diagonal_kl(local_mu, local_var, global_mu_before, global_var_before)
```

For top selection, a deterministic reference is sorting by `(-score, coordinate_index)` and selecting the first `m` entries. An optimized partial-selection implementation is acceptable after its cardinality and tie behavior match this reference. Random selection should use a separate generator's exact-size sampling without replacement.

### 12.2 Packet packing and reconstruction convention

```python
# Inputs already validated; mask is a bool vector of length d.
# Arrays on this protocol boundary contain true means and variances.
selected = np.flatnonzero(mask)
bitmap = np.packbits(mask, bitorder="little")
mu_send = np.ascontiguousarray(local_mu[selected], dtype=np.float32)
var_send = np.ascontiguousarray(local_var[selected], dtype=np.float32)

# Recheck the transmitted representation, not just higher-precision inputs.
if not np.isfinite(mu_send).all():
    raise FloatingPointError("Mean cannot be represented safely as float32")
if not np.isfinite(var_send).all() or np.any(var_send <= 0):
    raise FloatingPointError("Variance cannot be represented safely as float32")

# Decoder must check byte length BEFORE unpacking: unpackbits can pad [NP-UNPACK].
if bitmap.dtype != np.uint8 or bitmap.ndim != 1 or bitmap.size != (d + 7) // 8:
    raise ValueError("Invalid bitmap storage")
all_bits = np.unpackbits(bitmap, bitorder="little")
if np.any(all_bits[d:]):
    raise ValueError("Nonzero padding bits")
received_mask = all_bits[:d].astype(bool)
if int(received_mask.sum()) != mu_send.size or mu_send.size != var_send.size:
    raise ValueError("Bitmap and value arrays disagree")

mu_effective = global_mu_before.copy()
var_effective = global_var_before.copy()
mu_effective[received_mask] = mu_send
var_effective[received_mask] = var_send

raw_upload_bytes = int(bitmap.nbytes + mu_send.nbytes + var_send.nbytes)
```

### 12.3 No-extra-traffic round-cost helper

```python
from numbers import Integral


def core_round_bytes(d, k, *, sparse=False, m=None):
    """Return down/up/total array bytes; excludes headers and auxiliary state."""
    if any(isinstance(x, bool) or not isinstance(x, Integral) for x in (d, k)):
        raise TypeError("d and k must be integers")
    if d <= 0 or k < 0:
        raise ValueError("Require d > 0 and k >= 0")
    d, k = int(d), int(k)
    down = k * 8 * d
    if sparse:
        if isinstance(m, bool) or not isinstance(m, Integral) or not 0 <= m <= d:
            raise ValueError("Require an integer keep count 0 <= m <= d")
        up = k * (8 * int(m) + (d + 7) // 8)
    else:
        up = k * 8 * d
    return {"down_bytes": down, "up_bytes": up, "total_bytes": down + up}
```

For the actual ledger, prefer summing final array sizes per transferred packet, then use these formulas as consistency checks. The helper assumes every download recipient replies once and all use the same `d` and `m`.

## 13. Source map and attribution

The new selection rules, bitmap protocol, reconstruction convention, and cost ledger are this project's proposed extension. Do not attribute them to the source papers. The references below identify the inherited mathematics and implementation interfaces.

**[FOLA]** Liangxi Liu et al., *A Bayesian Federated Learning Framework with Online Laplace Approximation*. Project file: `02_OnlineLaplaceApproximationBayesianFL.pdf`; uploaded version arXiv:2102.01936v3, 2 December 2023. Relevant locations: Eq. (12), PDF p. 6, for Gaussian-product aggregation; Eqs. (15)–(16), PDF p. 7, for the local prior loss; Section III-D, PDF pp. 7–8, for diagonal and online curvature/precision approximation; Algorithm 1, PDF p. 9, for the training/exchange structure. The bitmap and communication formulas in this guideline are derived for our protocol, not quoted results of this paper.

**[BBB]** Charles Blundell et al., *Weight Uncertainty in Neural Networks*. Project file: `01_WeightUncertaintyinNeuralNetworks.pdf`; Eq. (1) and its derivation, PDF p. 3, explain posterior-first/prior-second KL in variational training. This does not mandate the orientation of a post-training communication score.

**[FedAvg]** H. Brendan McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*. Project file: `00_CommunicationEfficientLearningofDeepNetworksfromDecentralizedData.pdf`; corrected Algorithm 1, PDF p. 5, normalizes data-size weights over participating clients. Use the working repository's documented weighting/failure policy consistently.

**[NP-PACK]** NumPy official reference, `numpy.packbits`: packs a logical mask into `uint8` and pads the last byte with zeros. Documentation: `https://numpy.org/doc/stable/reference/generated/numpy.packbits.html`.

**[NP-UNPACK]** NumPy official reference, `numpy.unpackbits`: specifies bit order and trimming through `count`; requesting more bits than supplied can add zeros, so validate storage length first. Documentation: `https://numpy.org/doc/stable/reference/generated/numpy.unpackbits.html`.

**[FL-CLIENT]** Flower official reference, `NumPyClient.fit`: describes the arrays, training-example count, and metrics return contract. Documentation: `https://flower.ai/docs/framework/ref-api/flwr.client.NumPyClient.html`.

**[FL-SERIALIZER]** Flower official source documentation, `flwr.common.parameter`: array-to-bytes conversion through NumPy serialization. Documentation: `https://flower.ai/docs/framework/_modules/flwr/common/parameter.html`.

**[FL-STRATEGY]** Flower official strategy reference; the Message/ServerApp interface and legacy strategy interface have different hooks. Consult the installed version rather than assuming the latest signature. Documentation: `https://flower.ai/docs/framework/ref-api/flwr.serverapp.strategy.FedAvg.html`.

Official API references are supplementary implementation documentation. They are not evidence that the proposed sparse method improves accuracy or preserves convergence.

---

## Handoff completion criterion

The implementation is ready for the main experiment when the existing FedAvg and FOLA baselines still run, all three sparse selectors use the same packed-bitmap protocol, the server reconstructs and aggregates exactly as specified, tests establish full-retention equivalence and correct byte counts, and every evaluated global model is logged against cumulative uplink-plus-downlink communication. Report observed results without assuming the proposed KL order must win.
