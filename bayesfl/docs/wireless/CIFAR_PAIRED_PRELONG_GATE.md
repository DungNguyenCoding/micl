# CIFAR-10 paired aggregation/evaluation and pre-long validation gate

**Stage:** after original CIFAR R20 equal-bandwidth and joint-pilot runs; before new R100/R150 training.

## Protocol

Study profile: CIFAR-10 `paper_basiccnn`, 878,538 coordinates, 20 clients (full participation), saved Dirichlet alpha=0.01 and original full local fraction, seed 0, ten local epochs, batch size 32, SGD LR0.02 with saved cosine schedule, FOLA `paper_reference`, lambda0.01, clip10, `kl_global_local` score and `floor_for_score` precision scoring. The original FOLA raw precision must be uploaded exactly, never floored in transit.

Source checkpoint: **equal-bandwidth pilot `cifar_eqb_pilot20_v1`, global R10**. Local preparation is **R11** so it uses exactly the trained R10 posterior and the original client/round RNG seeding; one FOLA call per original client. Both policies see exactly those R11 client posteriors and KL scores. This differs fundamentally from comparing two already-diverged R20 trajectories.

Let \(d=878538\), \(\pi_k=n_k/\sum_j n_j\), \(U\) the common upload window, \(c_k\) the spectral efficiency and \(B=1\) MHz. Sparse posterior packets have \(L(m)=64m+8\lceil d/8\rceil\) bits; the dense endpoint has \(L(d)=64d\) bits. Equal bandwidth assigns \(B/K\) and uses each client's maximal feasible count. Joint allocation chooses integer counts and bandwidth grants to minimize \(\sum_k\pi_k\sum_{i\notin S_k}\mathrm{KL}_{k,i}\), subject to budget/deadline. Selected means and **raw precisions** plus bitmap are sent; the server fills omitted coordinates from the R10 prior. Coordinates omitted by all clients restore that prior exactly after the weighted FOLA product.

The protocol branches run on a local emulation transport:

1. `prepare` 20 times across clients in **one shared pass**;
2. Scalar `tail` and `price` queries for both policies against the same frozen scores (no remote score arrays);
3. Save the prepared cache **only in memory**, before upload, and use separate branch states to run the production `upload` and `release` protocol for each policy;
4. Decode each actual packet with its exact integer grant and verify selected values, omitted prior fill, bitmap cardinalities, transmitted lengths and actual discarded-KL tails;
5. Call `bayesfl.wireless.protocol.aggregate`, which invokes the original `aggregate_fola` and old-prior restoration;
6. Use an independent **full posterior FOLA aggregation** from the frozen prepared client posteriors as an offline oracle; verify it matches the wireless reducer with all-true masks. **Do not equate this oracle to a feasible wireless method at 180s**;
7. Compute original R10 and three R11 test metrics using `CentralEvaluator._collect_probabilities` and `predictive_metric_bundle` (the same mean-only point-predictive path used in paper experiments), fixed seeded evaluation per round;
8. Export scalar audit evidence only, check checkpoint/partition/resolved-config/run-summary hashes unchanged.

The only new GPU work is that one 20-client local preparation and evaluation. No model fit occurs after the two policy branches diverge. This is a **one-step counterfactual**, not an estimate of future convergence.

## One-shot server command

After installing the separate ZIP and ensuring a GPU is free:

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
bash scripts/run_cifar_prelong_gate.sh 0
```

The wrapper executes the existing complete wireless tests, repository tests, **24 legacy MNIST/CIFAR endpoint check**, read-only radio check, R10 preflight and paired experiment, then builds a `.zip` containing only source provenance, accuracy/NLL, per-client allocations, model-array packet counts, canonical control-message costs, test logs and readiness assertions. No long run and no source replacement occurs.

### Pass/fail gate

- Tests, full source regression, original experiment endpoint check and fixed-gain resource preflight pass.
- No existing training process is using the selected GPU (operator verifies before launch).
- Exactly one local R11 FOLA fit per client; neither policy retrains; source and partition hashes stable.
- Both policies' real uploaded masks and packet byte sizes match exact grant counts; actual weighted KL tails match queried KL objective within numerical tolerance.
- Each packet reconstructs selected values from transmitted arrays and unselected values from immutable R10 posterior; aggregation matches original FOLA behavior.
- Both allocations fit 1 MHz, the same \(U\), and the 30-second modeled control reserve.
- Centralized test evaluation produces finite R10/R11 mean-only CIFAR metrics on all **10,000 test images**.
- Dense uncompressed R11 FOLA aggregate is **diagnostic only**; actually check its minimum required bandwidth versus 1 MHz.
- The final read-only evidence bundle contains no raw score vectors, images, local parameter arrays, or checkpoints.

A lower joint discarded-KL objective is expected by incumbent construction but its **accuracy need not be higher**. Report R11 equal, joint, dense oracle accuracy/NLL and discrepancies explicitly; do not silently select favorable checkpoints.

## Before a 100–150-round study

A pass is a **software/experiment-engineering gate**, not a certification of the chosen radio scenario:

- Decide whether to keep the **fixed path-loss, ideal-CSI, 1 MHz, fixed-PSD, 180-second** scenario as a clearly labeled simulation study. The 180s slot, 0.001s/sample/epoch compute coefficient, control rate/reserve and score time are engineering assumptions, not readings from either reference paper.
- Use the same immutable geometry/topology, CIFAR partition, local epochs, LR/cosine horizon, seeds, rounds and deadline for both methods. Never make an unequal time/communication condition appear matched.
- Long comparison should train **independent equal-bandwidth and joint trajectories from the same initial global model**, not resume equal at R20 then switch to joint. Save endpoints at R100 and R150 within each **150-round trajectory**, not separate differently trained studies unless specifically desired.
- Maintain a separate **wireless modeled time** axis, **model-array bytes**, **canonical control bytes**, latency and **transmit energy**. Do not label any of these observed RF airtime/energy or measure GPU wall time as wireless time.
- R150 cannot be guaranteed from this single R11 test; rare long-run numerical or resource exceptions may appear, and many-seed studies remain necessary for scientific conclusions.
- Fixed gains eliminate random deep fading but cannot test outage robustness. Rayleigh block fading needs an explicit failed-round protocol and control-channel capacity treatment, not channel redrawing or dropping weak clients.
- Report solver numerical gap only as a heuristic; the integer global optimum is not certified.
- Repeat at least the paired diagnostic from **joint-trained R10** to reduce checkpoint-selection bias. A one-step gap also cannot establish statistical significance or lifelong convergence benefits.

After reviewing the read-only readiness bundle, one can prepare a **new**, non-overwriting 150-round pair without starting it:

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
PY=/home/micl/anaconda3/envs/dungndh/bin/python
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
TAG="cifar_fixed_pair150_$(date +%Y%m%d_%H%M%S)"
"$PY" scripts/prepare_wireless_study.py \
  --profile cifar10_a0p01_f1 \
  --scenario scripts/configs/wireless/cifar10_scheduling_pilot_v1/scenario_fixed.yaml \
  --tag "$TAG" --rounds 150 --backend flower \
  --policies equal_bandwidth joint --allow-example --prepare-partition
```

**That command only creates configs/queues; it does not run training.** It deliberately uses `--allow-example` because this engineering scenario is not physically locked. Inspect both configs and queue mappings before any GPU launch. If you want to make final publication-grade claims, calibrate or explicitly justify the remaining assumptions first and repeat multiple seeds.
