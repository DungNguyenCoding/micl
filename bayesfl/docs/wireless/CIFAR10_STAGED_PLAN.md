# CIFAR-10 wireless pilot: equal bandwidth before joint allocation

Prepared 2026-11-09. **PROPOSED ENGINEERING PILOT, NOT A LOCKED PAPER SCENARIO.**

This note records the current requested order: resource/logic checks -> actual CIFAR-10
smoke -> equal-bandwidth CIFAR-10 pilot -> joint-KL CIFAR-10 pilot. It does not authorize
MNIST jobs, retuning the saved learner, replacing source files, or long unattended studies.
Read this note alongside `CHAT_HANDOFF.md` in a subsequent conversation. The existing
handoff exporter copies Markdown under docs; it does not automatically add new wireless
runs unless supplied using its documented wireless-run options.

## 1. Evidence and what is transferred

**S00:** Project `00_SchedulingFL.pdf`, Ren et al., *Scheduling for Cellular Federated Edge
Learning with Importance and Channel Awareness*, Section II.C equations (4)-(5),
Section III.D equations (15)-(18), and Section VI.A, PDF pages 20-21. Its digital OFDMA
model uses a total bandwidth constraint, a full-band multicast at the weakest downlink
rate, and a maximum-client computation barrier. The experiment specifies a 500 m cell,
uniformly placed devices, LTE path loss `128.1 + 37.6*log10(distance_km)` dB,
noise density -174 dBm/Hz, nominal device power 24 dBm, BS power 46 dBm, and 1 MHz.
Its experiment has 30 devices and uses 16 bits/parameter; those two values are NOT
substituted for the existing FedKLS client count or wire representation.

**S07:** Project `07_AirCompWirelessFederatedLearning.pdf`, Hong et al., journal version,
*Distribution-Level AirComp for Wireless Federated Learning Under Data Scarcity and
Heterogeneity*, DOI 10.1109/TWC.2026.3731174. Section III.C, PDF page 5, uses complex
Rayleigh block fading and effective receiver noise including out-of-cell interference.
Table I, PDF page 10 (printed page 23089), specifies 300 m, 100 devices, maximum OFDM
symbol power 20 dBm, effective noise power -74 dBm, 1024 subchannels, and path-loss
exponent 4. Its datasets include CIFAR-100, not the selected CIFAR-10 experiment.

The online arXiv v1 has different simulation numbers. This note uses the **uploaded
journal version**, not those earlier numbers. Source PDF hashes are in the setup manifest.

For the initial pilot, use a coherent S00 radio-scale scenario, with fixed path loss.
S07 motivates a later, separately audited Rayleigh extension. Do NOT multiply S00 LTE
power gains by another distance^-4 factor; that would charge path loss twice. Do NOT
put -74 dBm into a W/Hz noise-density field. S07's OFDM-subchannel count is not a bandwidth
in Hz and does not define a conversion from its effective noise variance to our noise PSD.
No AirComp distortion, analog superposition, amplitude alignment, or additive parameter
noise is imported into this reliable-digital model.

## 2. Keep the selected learning profile

The setup is additive to `BayesFL_Wireless_Full_Source_v1.zip` (the version installed and
tested by the user). The normal generator reads the selected actual resolved configs.

- CIFAR-10, `paper_basiccnn`, d=878538; 20/20 clients each round; seed 0.
- `paper_dirichlet`, alpha=0.01, local_data_fraction=1.0; preserve the actual NPZ indices.
- E=10, batch size=32, SGD, zero momentum/weight decay, norm-10 gradient clipping.
- FOLA: base LR .02, cosine minimum .0001, fixed horizon 400; lambda .01,
  `paper_reference`, initial raw omega 0, aggregation epsilon 1e-5.
- `floor_for_score` with resolved score floor 1e-12; transmit raw float32 means and
  precision. Preserve augmentation, seeds, evaluation and prior-fill reducer.
- Fixed-scenario comparison uses KL(global || local) for BOTH policies.
- Keep `keep_ratio=1.0` only as the required adaptive-controller placeholder. The actual
  integer grants set the number transmitted. The old ratio menu is not an action menu.

Partition content SHA256:
`914fe035f7b504b0059ca85b5a6a194bd41027fb2cfe2765ee777142d68280f2`.
Actual assigned samples: min 1, mean 2499.4, max 8706, total assignments 49988. Do not
replace this with 2500 examples on every client or regenerate the historical partition.
The copy contains two historical backfills; this setup does not repair/change those.

Existing paper endpoints are non-wireless reference data. They cannot be relabeled as
new wireless training results. The historical R150 FOLA / R300 FedAvg equal-byte rule
is NOT an equal-time rule in this controller.

## 3. Proposed radio and topology values

| Quantity | Pilot | Origin |
|---|---:|---|
| Total UL bandwidth | 1,000,000 Hz | S00 VI.A |
| Total DL bandwidth | 1,000,000 Hz | S00 full-band multicast |
| BS location | Origin | S00 |
| Cell radius | 500 m | S00 |
| Locations | Uniform in disk area, fixed PCG64 seed 0 | Geometry S00; seed/design here |
| Path loss | 128.1+37.6 log10(r/1000), r in meters | S00 |
| Power gain | 10^(-PL_dB/10), not an amplitude | Derived |
| Channel mode for first pair | `fixed` | Deliberate diagnostic stage |
| Noise density | 3.981071705534986e-21 W/Hz | -174 dBm/Hz from S00 |
| BS PSD | 3.9810717055349735e-5 W/Hz | 39.810717055349734 W / 1 MHz |
| Device PSD | 2.51188643150958e-7 W/Hz | Conservative adaptation below |
| Device cap | 0.251188643150958 W | 24 dBm from S00 used as a cap |
| Shadowing / extra interference / noise figure | Not added | Explicit idealized pilot scope |
| Successful data decode | Exact finite-precision packet | Existing reliable-digital abstraction |
| Channel knowledge | Ideal known gains | Pilot assumption, no CSI acquisition simulator |

Use `r=500*sqrt(U)` and `theta=2*pi*V`, not r uniform on [0,500]. The included topology
is pinned to client IDs. It was not selected using accuracies, data labels or KL scores.
There is no minimum-distance guard, clipping, redraw, or score-aware placement. This
particular realization has nearest/farthest distances 26.165340/483.495715 m.

### Essential power convention

The v1 allocator assumes `c_k=log2(1+PSD_k*g_k/N0)` independent of its bandwidth variable.
Set PSD = P_cap / **total** system bandwidth, so `P_k=PSD_k*B_k <= P_cap` for every grant.
At equal bandwidth, B_k=50000 Hz and P_k=0.0125594321575479 W = about 10.99 dBm.
Thus this does NOT reproduce fixed 24 dBm on every 50 kHz device.

Do not use P_cap/(B/K) while claiming the same 24 dBm cap for arbitrary joint allocations.
That would allow up to K*P_cap if one client receives all the spectrum. The configuration
validator deliberately enforces PSD*B_total <= cap. To model fixed total power P_k for
all bandwidths instead, the rate becomes B_k*log2(1+P_k*g_k/(N0*B_k)); the supplied
constant-efficiency allocator is not the derived optimizer for that different model.

## 4. Explicit timing assumptions (NOT supplied by either paper)

| Field | Pilot | Interpretation |
|---|---:|---|
| `round_target_s` | 180 | Fixed 3-minute modeled slot, not GPU runtime |
| `total_time_budget_s` | null | Initially use explicit round caps |
| `seconds_per_sample_epoch` | .001 on all clients | Nominal 1000 sample-epochs/s, not measured |
| `fixed_seconds` | 0 on all clients | No separate fixed preparation cost |
| `score_seconds` | .1 on all clients | Assumed scoring/sorting overhead, not profiled |
| `control_rate_bps` | 1,000,000 | Assumed serial full-band robust control service |
| `control_reserve_s` | 30 | Shared headroom for both policies, verify actual use |
| `control_exchange_latency_s` | .001 | Per-exchange latency, not a source measurement |
| `rpc_timeout_s` | 3600 | Emulator timeout only, not radio latency |

S00 provides the functional compute model n_k*C/f_k. For E local passes, map the sample
coefficient to C/f_k and use E*n_k*C/f_k, plus scoring. It does not provide a measured
BasicCNN/FOLA throughput for these devices. The chosen coefficient must later be
profiled/calibrated, or retained as a disclosed hypothetical-device assumption with a
sensitivity study. Do not turn the workstation's Ray wall time into per-edge-device time.

The controller counts canonical JSON control bytes, not the full Flower/network envelope.
The 1 Mbit/s abstraction is feasible against the weakest ideal full-band fixed link in
this realization (7.24036 Mbit/s UL). It is not guaranteed under deep fading. A 30-second
reservation is not a statement that the allocator consumes 30 seconds each round. Check
`modeled_control_seconds`, `price_queries`, and control bytes after the actual smoke.
Unused time stays idle. Do not increase control budget/deadline for joint only.
Radio energy output currently covers the modeled data plane, not circuit, compute,
CSI acquisition or control-transmission energy. Do not report it as total device energy.

The 180-second slot is a **pilot design**, chosen before learning outcomes. The no-training
resource calculation yields mixed sparse/dense packets rather than trivial all-dense
allocation. It is neither an optimum nor a value extracted from either cited paper.

Solver settings retain v1 numerical defaults: initial price 1.0, max bracket queries 48,
max bisection queries 64, price atol 0/rtol 1e-12, gap atol 1e-10/rtol 1e-6. These are
implementation controls, not RF or paper parameters. Log fallback/iteration-limit status;
float64 gaps are diagnostics, not certified integer-global-optimality or accuracy bounds.

## 5. Equal bandwidth is not equal retention

Equal-bandwidth policy assigns B/20=50 kHz to every client. For each client it takes the
largest integer m whose typed packet fits its own upload capacity; it then selects the
top m KL(global||local) coordinates. Different gains cause different m, even with equal B.
The `uniform` policy instead forces a common count and is not the requested first study.

For d=878538, actual v2 payload bits are:

```
L(m) = 64*m + 878544    for 1 <= m < 878538
L(d) = 56226432        for m = d
```

Bitmap = 109818 bytes (6 pad bits); minimum sparse packet = 109826 bytes; dense posterior
packet = 7028304 bytes. The dense endpoint has no bitmap. Always compare it separately;
some near-dense sparse packets exceed dense size. Use production integer-count helpers,
not count -> ratio -> ceil round trips.

Current production-code resource preflight, using the supplied saved partition:

| Quantity | Calculated pilot value |
|---|---:|
| Multicast DL | 3.867259582 s |
| Max modeled preparation | 87.16 s |
| Upload window | 58.972740418 s |
| Equal-share keep min / mean / max | .364074 / .525082 / 1.0 |
| Dense-format clients | 1 / 20 |
| Equal-share array bytes/round | 82923630 (control separate) |
| Minimum feasible all-dense FOLA slot with joint bandwidth | 241.023289 s |

These are capacity calculations, NOT learned accuracies, KL values, or observed radio
measurements. The fixed scenario yields constant equal-share counts across rounds.
Joint-KL counts can change with the locally trained posteriors. Joint optimization has
the feasible equal-share allocation as an incumbent on each frozen input; this protects
its local reconstruction objective under the implementation's numerical rules, not a
universal cross-trajectory accuracy advantage.

Dense FOLA is infeasible at the 180-second slot even with optimal bandwidth splitting.
Do not silently enlarge its deadline or reduce its participation. It is intentionally
not in the first equal-vs-joint pilot. A later dense comparison must declare a genuinely
common feasible scenario or explicitly address infeasibility.

## 6. Staged execution

The user has reported: 59 wireless tests passed; 197 repository tests passed; all 24 old
endpoints passed; six synthetic-policy smokes passed; Flower test passed; two-round real
Flower/Ray synthetic smoke passed. These are existing gates, not fresh execution by this
setup. The actual CIFAR-10/CUDA path remains to be run.

### A. Read-only resource test

```
cd ~/DungNDH/micl/bayesfl || exit 1
PY=/home/micl/anaconda3/envs/dungndh/bin/python
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" scripts/check_cifar_wireless_scenario_v1.py
```

Requires the protected saved configs, layout metadata, and original partition NPZ. No
training, downloads, config writes, queues, or metric modifications occur. No current-round
CIFAR KL objective is evaluated; zero tails inside the equal-count helper are placeholders.

### B. Two-round actual CIFAR equal-bandwidth smoke

```
SCENARIO=scripts/configs/wireless/cifar10_scheduling_pilot_v1/scenario_fixed.yaml
TAG=cifar_eqb_smoke_v1
"$PY" scripts/prepare_wireless_study.py \
  --profile cifar10_a0p01_f1 --scenario "$SCENARIO" --tag "$TAG" \
  --rounds 2 --smoke --allow-example --backend flower \
  --policies equal_bandwidth --prepare-partition
```

The two queue files are generated by the existing tool; only GPU0's queue contains a
job. Do not interpret the empty GPU1 queue as missing work. The historical loader keeps
E=10, batch32 and the existing augmentation even for this smoke.

Check running processes/GPU occupancy, then run the one-job queue:

```
pgrep -af 'bayesfl.main|_run_gpu_queue.sh|_run_wireless_queue.sh' || true
nvidia-smi
# Only continue when the selected GPU is free.
mkdir -p logs
TS="$(date +%Y%m%d_%H%M%S)"
Q="scripts/queues/wireless_${TAG}_gpu0.txt"
LOG="logs/wireless_${TAG}_gpu0_${TS}.log"
if [ -s "$Q" ]; then
  nohup bash scripts/_run_wireless_queue.sh 0 "$Q" > "$LOG" 2>&1 &
  echo "PID=$! | queue log=$LOG"
else
  echo "No job in $Q; inspect generation output rather than launching an empty queue."
fi
```

The queue log prints the separate job-log path; inspect that job log for live training
progress. A QUEUE FINISHED line alone is not success: require the job SUCCESS message
and the summary below.

```
"$PY" scripts/summarize_wireless_study.py \
  --study-dir scripts/configs/wireless/cifar_eqb_smoke_v1
```

For the pinned scenario expect completed round 2, modeled time 360 s, and array bytes
165847260. Accuracy and actual control bytes are unknown until the server runs it.
Do not delete/overwrite an existing study directory to rerun; inspect its output and use
an explicitly new tag only for an intentional new run.

### C. Equal-bandwidth learning pilot, then joint

After a successful smoke, generate a separate 20-round equal-bandwidth run (omit --smoke,
use tag cifar_eqb_pilot_v1). Run and inspect it before joint; 20 rounds is an engineering
pilot, not a final convergence assessment. Then use the SAME scenario with policy joint
for a separately named two-round joint smoke, followed by a fresh 20-round joint pilot.

The joint pilot starts at the same initial seed/model/partition, not from the trained
equal-bandwidth model. Keep source learning fields and scenario bytes identical except
policy/run identity. Use the generator and matching summary command for each tag.
After validation, a 150-round pair is an available next study, not an automatic launch.
At 150 slots each has 27000 modeled seconds; this is not the old equal-byte comparison.
Do not set a different cosine horizon when changing the cap.

Inspect client identities/counts, grant bytes, retained ratios, total bandwidth, per-client
power, observed control-time utilization, failed/pending rounds, and allocation status.
Expect adaptive methods to train the full local model exactly once per round, not once
per price query. Fixed-slot time is different from Flower/Ray execution wall time.

## 7. Rayleigh extension: later, not silently enabled now

S07 motivates power gains g_{k,t}=beta_k*Z_{k,t}, with Z exponential mean1, one block per
modeled round. Using beta from S00's LTE formula is an explicit hybrid model; S07's own
mean gain is distance^-4. The current `block_rayleigh` option already implements independent
exponential power gains and optional reciprocal UL/DL fades.

A diagnostic scan is available without changing the scenario:

```
"$PY" scripts/check_cifar_wireless_scenario_v1.py --rayleigh-scan-rounds 150
```

For the supplied topology and seed0, the reviewed v1 channel implementation produced
nine infeasible tested rounds under this pilot: one equal-share minimum-payload failure
(R30) and eight extra assumed-control-rate violations. The latter link-rate check is an
additional check in this helper, not a guarantee imposed by the current controller.
The controller's fixed control-rate model itself does not couple control traffic to fading.

The current all-client controller has no fade/outage continuation policy. A finite pilot
cannot establish all-future-round feasibility for unbounded Rayleigh fades. Do not hide
failures by resampling, clipping gains, dropping clients, deleting ledger rows or enlarging
only one method's deadline. Before long fading studies, explicitly decide and implement
outage/retry behavior, control-link adaptation and the treatment of spent time/bytes, or
use a separately justified bounded/trace scenario and report its conditioning honestly.

## 8. Installation scope and handoff

This package ADDS one scenario, topology, preflight helper, this note, and a setup manifest.
It does not replace trainers, the allocator, existing docs, datasets, outputs or the
installation receipt. Keep the earlier installer backup. After actual smoke results,
record the selected scenario status, validation outcome and run names in the handoff;
do not label this proposal a final paper lock until explicitly approved.
