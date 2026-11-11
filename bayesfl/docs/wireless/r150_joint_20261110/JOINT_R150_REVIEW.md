# Joint FedKLS on CIFAR-10: R150 Audit and Process Review

**Reviewed:** 2026-11-10  
**Research scope:** Joint KL-based bandwidth-and-retention optimization **only**, no equal-bandwidth performance comparison.  
**Evidence:** User-provided `bayesfl_joint_R150_audit_20261110_200059.zip` (SHA-256 `b881b4df6d12e9d836411393ef62c70b862cf0926fd7a988e825c52eb25ffd31`).  
**Source run:** `outputs/run_wireless_cifar_eqb_joint150_fixed_v1_cifar10_a0p01_f1_joint_seed0_20261109_230207/`  
**Prior audit status:** `PASS_ENGINEERING_RECORD_INTEGRITY`.

## Executive conclusion

The CIFAR-10 joint-FedKLS run **completed 150 rounds and passed a detailed record-level engineering audit**. Global accuracy and NLL exhibit sustained improvement over the trajectory despite ordinary round-to-round fluctuations. Allocation, payload, bandwidth, power, deadline, training-step, and event accounting were internally consistent in the exported records.

This supports **engineering readiness in the tested fixed-channel digital simulator**, **not** global integer optimality, physically validated radio performance, statistical replication, Bayesian predictive uncertainty validation, or full checkpoint tensor integrity. Continue joint-only investigation without rewriting original learner or historical paper results.

## Provenance / verification

- Verified every one of the **161 archived evidence SHA-256 values** against the corresponding ZIP member.
- Inspected **151 global evaluation rows (R0–R150), 150 round records, 3,000 client-round rows, 150 allocation JSON files**, and **133,509 event-ledger records**.
- Reconstructed the objective from source client rows as `sum_k client_weight_k * actual_discarded_kl_k` in every round. It matches `wireless_rounds.csv`'s actual objective within numerical tolerance.
- Reviewed original audit's `joint_audit.json` with **169 labeled hard checks**, all passed. Not all 169 checks are independent properties; most are each of the 150 individual allocation-JSON round-presence checks.
- Checkpoint tensors themselves were **not included**, so this review cannot independently inspect numerical health of full mean/precision arrays or re-evaluate model weights.

## Learning trajectory

| Round | Accuracy (%) | NLL | Mean retained (%) | Weighted discarded KL | Price queries |
|---:|---:|---:|---:|---:|---:|
| 0 | 11.91 | 2.3091 | — | — | — |
| 1 | 22.43 | 2.7680 | 46.93 | 3.75873 × 10^9 | 37 |
| 20 | 38.05 | 1.8330 | 47.79 | 0.304195 | 39 |
| 50 | 44.14 | 1.6309 | 47.83 | 0.0375042 | 37 |
| 100 | 49.11 | 1.4943 | 47.25 | 0.00822589 | 39 |
| 150 | **52.63** | **1.4045** | 47.55 | 0.00377063 | 37 |

- Best checkpoint-free *recorded evaluation* is **52.76% at R148**. Mean accuracy over R131–R150 is **52.1115%**.
- There are **56 downward accuracy transitions** among R1–R150, with largest drop **−1.20 pp at R78**, and 60 upward NLL transitions. This does not indicate sustained divergence: each successive 10-round block mean accuracy improved, and successive 10-round block mean NLL declined.
- Accuracy at R150 is **+3.52 pp** versus R100; a plateau is **not proven**.
- Round-1 discarded KL is extremely large because the initial global precision is floored for KL **scoring** (`floor_for_score`), not because the transmitted posterior precision was replaced by that floor. Compare later rounds separately; their KL scores also reflect evolving posteriors and cannot alone prove learning improvement.

Plots: `joint_accuracy_R150.png`, `joint_nll_R150.png`, `joint_retention_R150.png`, and `joint_discarded_KL_R2_R150.png`.

## Learner and local computation

The resolved configuration specifies CIFAR-10 BasicCNN, 878,538 coordinates, full participation of 20 clients, Dirichlet alpha 0.01, seed 0, 10 local epochs, batch 32, SGD LR 0.02 with cosine horizon 400 and minimum LR 0.0001, FOLA prior lambda 0.01, and clipping norm 10. Local precision remains raw on the uplink and `kl_global_local` is used for coordinate ranking.

The client records indicate **49,988 samples** in the partition, with extreme non-IID client sizes (two clients have only **one** sample; largest client has 8,706). All 20 participate every round. For each client, the logged step count equals `10 * ceil(num_examples / 32)`, giving:

- **15,740 local training steps per round** and **2,361,000 total steps**;
- **499,880 local example presentations per round** and **74,982,000 cumulative presentations** (including repetition over 10 epochs).

These are **record-consistency checks**: they establish that recorded workload matches config, not that checkpoint tensors or gradient values are independently validated.

## Wireless allocation, feasibility, and optimization

- Fixed channel topology, OFDMA-like orthogonal digital uplink, **1 MHz total bandwidth**, **180 s modeled round**, **30 s fixed control reserve**; **58.9727404 s** upload window.
- All 150 allocations report `numerical_gap_target`, **0** globally certified integer-optimal. Feasible lower-bound and objective-gap diagnostics are not global certificates.
- Per-round total allocated bandwidth: from **999,985.15** to **999,999.999 Hz**, never exceeding 1 MHz in the recorded results.
- Individual client bandwidth ranges across all rounds: **6,804.70–113,446.03 Hz**.
- Actual transmitted counts are positive integers (`1 <= m <= 878,538`); client retained fractions span **11.410%–88.964%**; **no dense endpoint** and **no zero-coordinate upload** occurred.
- Mean retention across all client-round rows **47.688%**; data-size-weighted **49.055%**. The minimum 11.41% occurs for a one-sample client; it is allowable under the present objective but deserves fairness/rare-class study, not automatically a bug.
- Maximum deadline overrun is approximately **1.42e−14 s**, attributable to floating-point roundoff. Model bandwidth/power/timing checks passed.
- All modeled uplink transmission durations nearly equal the 58.97274 s window because the policy grants bandwidth against that window; this is a structural feature of the allocation model and not an RF observation.
- Maximum implied individual device transmit power **0.02850 W** under the configured fixed PSD, below the **0.25119 W** power cap. The cap is not normally active; this is intentional under the fixed-PSD convention.

## Communication and control overhead

| Counter | R150 total |
|---|---:|
| Model-array bytes (downlink + uplink) | **11,438,601,248** |
| Downlink model-array bytes | 1,054,245,600 |
| Uplink model-array bytes | 10,384,355,648 |
| Control-payload bytes | **111,770,677** |
| Recorded model-array + control bytes | **11,550,371,925** |
| Price queries / client replies | **5,579 / 111,580** |
| Price-query/control-reply bytes | **97,301,483** |
| Price traffic share of all control bytes | **87.05%** |
| Mean / maximum modeled control time per round | **6.0033 / 6.5367 s** |
| Reserved control time | **30 s per round** |
| Mean unused control reserve | **23.9967 s per round** |

Communication is based on typed model-array counts and canonical JSON control messages, not measured RF packets. Modeled durations use a configured control rate and include per-exchange overhead. The repeated price exchange is the largest control-plane contributor, but it remained within the conservative control reserve in all 150 rounds.

## Transmit energy (modeled only)

The event ledger matches **25,315.7414 J** cumulative modeled data-plane transmit energy. **23,093.7565 J (91.22%)** is attributed to the broadcast/downlink; **2,221.9849 J (8.78%)** to client uploads. The number is not complete device energy: compute, circuitry, receive, and control-energy costs are omitted.

## Predictive-quality nuance

Final accuracy **52.63%**, NLL **1.40449**, Brier score **0.61704**, ECE **0.08143**, average confidence **60.54%**. ECE around 8.14 percentage points suggests calibration is worth analyzing. The existing FOLA evaluation is **mean-only**, and the record has mutual information equal to zero by design, so it is not evidence that uncertainty-calibrated Bayesian prediction was tested.

## Correction: prior audit's per-client CSV column

The exported `audit/joint_clients_audit.csv` named `weighted_discarded_kl` actually contains `n_k * actual_discarded_kl`. The **true** weighted contribution to the objective is `(n_k / N) * actual_discarded_kl`, with `N = 49,988`. For example, at R1 client 0, the exported value was about **1.11455 × 10^13**, while the correct normalized contribution was **2.22964 × 10^8**.

**This is a derived audit-CSV labeling/calculation issue, not a production allocator or training bug.** The original audit's *round-level* objective used normalized weights and passed checks; our independently re-summed client-level contributions match it. Use `joint_clients_true_weighted_KL.csv` supplied with this review, not the misleading original column, for per-client weighted-KL plotting or analysis.

## Status and recommended next actions

**STATUS:** `PASS_ENGINEERING_RECORD_INTEGRITY / EXPLORATORY_JOINT_CIFAR10_R150_VALIDATED`.

The evidence supports the interpretation that the joint-only CIFAR-10 training and simulated radio allocation process operated normally **under this one fixed-channel, seed-0 scenario**. There is no reason to change the FedKLS algorithm solely because of the current audit. Preserve its R150 outputs and configuration.

To strengthen the scientific conclusion, next (1) inspect saved checkpoint **parameter means and raw precisions** for NaN/Inf and range/pathology without changing weights; (2) repeat joint-only runs over independent seeds/topologies; (3) study sensitivity to modeled compute and control timing and bandwidth; (4) specify a fading/outage policy before claiming robustness; and (5) refine the price-query control overhead for practical deployment. Do **not** treat this fixed-scenario simulation as measured physical-link performance.

**Documentation note:** The uploaded audit archive contains the archived metrics and report, **not** the current live canonical `docs/CHAT_HANDOFF.md`, `docs/wireless/STATUS.md`, and `docs/wireless/VALIDATION.md`. Their actual installed contents cannot be inferred from the ZIP. A guarded, additive documentation sync is provided separately.
