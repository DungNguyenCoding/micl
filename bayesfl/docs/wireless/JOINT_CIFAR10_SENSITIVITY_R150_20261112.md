# Joint FedKLS CIFAR-10: Fixed-Channel Bandwidth/Upload-Window Sensitivity (R150)

**Evidence date:** 2026-11-12  
**Status:** `PASS_ENGINEERING_RECORD_INTEGRITY` for all six new R150 runs. The original joint reference R150 audit also passed.  
**Scope:** Joint-KL (`kl_global_local`) wireless FedKLS only; no equal-bandwidth comparison, no new simulation or source change.  
**Scientific scope:** Single-seed, fixed-channel **exploratory** resource sensitivity under a scenario marked `EXAMPLE_NOT_LOCKED`; **not** a calibrated wireless deployment.

## Research question and experiment design

How does tightening *one* uplink resource at a time affect joint FedKLS training?

- **U sweep**: keep uplink `Bmax=1 MHz`; change only the modeled round target (180 to 170/160/150 seconds) to obtain shared uplink windows `U=58.972740 / 48.972740 / 38.972740 / 28.972740 seconds`.
- **B sweep**: keep `U=58.972740` seconds and downlink bandwidth at 1 MHz; vary only **uplink** `Bmax=1.00 / 0.55 / 0.75 / 0.95 MHz`.
- Fixed `uplink_psd_w_hz`, noise PSD, 20 channel path gains, channel mode, local compute coefficients, 30-s control reservation, 1-Mbit/s control service, price-search settings, 20-client CIFAR Dirichlet partition (`alpha=0.01`, seed0), model (`paper_basiccnn`, `d=878538`), FOLA training settings (SGD LR0.02 cosine, E10, B32, lambda0.01), and 150-round cap.
- Evaluations use mean-only CIFAR-10 centralized accuracy/NLL. All six new outputs are independent fresh R150 trajectories, not restarted from the original R150 checkpoint.

In this fixed-PSD configuration, the optimizer approximately allocates integer packet counts under

\[
\sum_{k=1}^{20}\frac{L(m_k)}{U c_k}\leq B_{\max}, \qquad c_k=\log_2\!\left(1+\frac{p_k^{PSD}g_k}{N_0}\right).
\]

Thus `Bmax * U` summarizes **feasible uplink information capacity** within this fixed-channel model. It does not completely determine accuracy: independently trained posterior trajectories and model aggregation are nonlinear.

## Verified final outcomes

`GB` and `MB` below are **decimal** (10^9 and 10^6 bytes). `Array GB` includes the recorded model arrays (multicast downlink plus clients' uplink); `Control MB` is separate logical serialized control payload, not measured PHY/MAC traffic.

| Run | Uplink Bmax (MHz) | Shared U (s) | Bmax × U (% of reference) | R150 accuracy (%) | R150 NLL | R150 mean retention (%) | Model arrays (GB) | Control (MB) | Price-tolerance rounds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Reference** | 1.00 | 58.973 | 100.00 | **52.63** | 1.4045 | 47.55 | 11.438601 | 111.771 | 0 |
| `u170` | 1.00 | 48.973 | 83.04 | 50.25 | 1.5105 | 39.14 | 9.707082 | 114.195 | 2 |
| `u160` | 1.00 | 38.973 | 66.09 | 52.47 | 1.5052 | 31.03 | 7.997870 | 115.404 | 3 |
| `u150` | 1.00 | 28.973 | 49.13 | **43.43** | **2.1130** | 23.02 | 6.254902 | 117.648 | **21** |
| `b0550` | 0.55 | 58.973 | 55.00 | **47.29** | **1.9207** | 25.70 | 6.861030 | 119.192 | **27** |
| `b0750` | 0.75 | 58.973 | 75.00 | 52.53 | 1.4623 | 35.30 | 8.901418 | 113.771 | 4 |
| `b0950` | 0.95 | 58.973 | 95.00 | **54.01** | **1.3948** | 45.12 | 10.942543 | 112.725 | 2 |

### Independent verification of uploaded audit ZIPs

All six new audit ZIPs, and the preceding reference ZIP, passed the following **independent checks** (not just relying on a printed `PASS`):

- Each ZIP contains its `audit/joint_audit.json` with status `PASS_ENGINEERING_RECORD_INTEGRITY`, 150 wireless rounds, 151 global-evaluation rows (including round zero), 3,000 client-round records and 150 per-round allocation JSONs.
- **All 161 source-evidence SHA-256 hashes in each ZIP matched the actual archived bytes**: **1,127/1,127 verified** across seven ZIPs.
- The **resolved configurations of all seven experiments are identical** after removing only `run_name`, `wireless.uplink_bandwidth_hz`, and `wireless.round_target_s`. The same fixed channels, seed, client partition and learning hyperparameters are therefore confirmed within the saved configuration records.
- Price-search status distribution was independently counted from `metrics/wireless_rounds.csv`: 150/150 `numerical_gap_target` for reference; 148/150 for U170; 147/150 for U160; 129/150 for U150; 123/150 for B0550; 146/150 for B0750; 148/150 for B0950. Remaining statuses were `price_tolerance`, **not training failures**.
- Max modeled control times: reference 6.537 s; U170 9.087 s; U160 9.223 s; U150 9.250 s; B0550 9.094 s; B0750 9.083 s; B0950 9.089 s. All below the 30-second reservation.
- All seven audits explicitly state **no certified globally integer-optimal solution**; the feasible incumbent and resource integrity are validated, not global optimality.

The completed six audits appear in `/mnt/data` in the reviewed conversation as `bayesfl_joint_R150_audit_20261112_033623.zip` through `...033628.zip`. The original reference archive is `bayesfl_joint_R150_audit_20261110_200059.zip`. On the Linux server, the original run CSVs and configuration YAMLs remain under their existing `outputs/` and `scripts/configs/wireless/` paths; **the plotting script reads those directly**.

## Learning interpretation

1. **Severe constraints hurt this seed's training**: U150 reaches 43.43% (−9.20 pp vs reference), B0550 47.29% (−5.34 pp). Both have much lower retention.
2. **The accuracy relationship is not monotone**: U160 (52.47%) finishes above U170 (50.25%), and B0950 (54.01%) finishes above 1 MHz (52.63%). Joint minimizes discarded KL, not validation error. These results are observations, not a claim of statistically significant differences.
3. **Non-monotonic training can be quite severe**: U150 has worst adjacent-round accuracy change **−10.86 pp**, and B0550 **−7.09 pp**; their NLL curves repeatedly oscillate late in training. Other sweeps are noticeably smoother. Their engineering PASS must **not** be interpreted as a guarantee of smooth or converged learning.
4. **Price search warrants further analysis in the harsh cases**: U150 has 21 and B0550 has 27 `price_tolerance` rounds. The per-round feasibility/byte checks still passed. This is a solver-tolerance behavior warning, **not** an infeasible-allocation finding. Do not retroactively adjust the saved experiments.
5. **Array traffic responds as expected**: fewer available uplink resources yield fewer transmitted posterior-array bytes. Modeled control bytes tend to increase as the search works at a tighter constraint, but they remain inside the reservation.
6. **Comparing R150 is comparing equal learning rounds, not equal wireless time**: U170/U160/U150 have 25,500/24,000/22,500 modeled seconds; reference and B-sweep have 27,000 seconds. Plot **accuracy versus rounds and versus modeled hours**; no unsupported interpolation or claim of equal time.
7. **One seed / fixed channels / hypothetical timing**: realistic claims require independent seeds, device compute calibration, defensible control rate/latency, and fading/outage sensitivity.

**Consequences:** Keep the joint code and all seven source runs untouched. These results establish a useful first joint-only fixed-channel sensitivity dataset. Further work may evaluate the observed poor stability in severe cases before modifying the optimizer.

## Reproduce publication-style, side-by-side figures

Use the installed plotting script `scripts/plot_joint_cifar_sensitivity_v1.py` (see `docs/wireless/JOINT_SENSITIVITY_PLOTS_RUNBOOK_v1.md`). It automatically resolves **exactly one completed R150 joint run per study** by manifest and validates the seven saved configurations/metrics.

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
PY=/home/micl/anaconda3/envs/dungndh/bin/python
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" scripts/plot_joint_cifar_sensitivity_v1.py --root "$PWD" --check-only
"$PY" scripts/plot_joint_cifar_sensitivity_v1.py --root "$PWD" \
  --output-dir "plots/wireless/joint_cifar_sensitivity_r150_$(date +%Y%m%d_%H%M%S)"
```

The output contains 4 paired PNG/PDF figures, `joint_sensitivity_endpoints.csv`, `joint_sensitivity_trajectories.csv`, provenance hashes and a small reading guide. The **primary figure** is `joint_sensitivity_side_by_side.png` (two columns: shorter U at 1 MHz versus less B at fixed U), with four aligned rows: accuracy, NLL, retention, and cumulative arrays+control bytes. Additional figures compare accuracy vs *modeled time*, discarded-KL vs round (exclude R1's initialization artifact), and endpoint sensitivity vs `Bmax*U`.

## Documentation synchronization

The `FedKLS_Joint_CIFAR10_Sensitivity_R150_Plots_Docs_v1.zip` installer **adds this document** and the plotting runbook, then **appends a tagged cross-reference** to `docs/CHAT_HANDOFF.md`, `docs/wireless/STATUS.md`, `docs/wireless/VALIDATION.md`, and `docs/wireless/README.md`, optionally `docs/RUN_REGISTRY.md` if that file exists. It backs up each changed canonical document outside the repository. It does not edit experiment source, training configurations, checkpoints, CSVs or locked registries.

After applying and verifying the installer on the Linux server, the **canonical handoff is updated** for the validated seven-run study. Until then, this update is prepared but not yet installed.
