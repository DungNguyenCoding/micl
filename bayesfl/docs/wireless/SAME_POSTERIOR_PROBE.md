# CIFAR-10 same-prepared-posterior counterfactual — v1

**Status:** Additive analysis utility; not a change to the wireless training algorithm, baseline configurations, historical output runs, or locked paper experiments. Prepared 2026-11-09.

## Question

Given one exact, common collection of locally trained CIFAR-10 FOLA posteriors, is joint allocation's **weighted discarded Gaussian KL** lower than equal bandwidth, and what differences in retention, model-array bytes, bandwidth and allocation-phase control overhead result?

The two completed 20-round runs alone cannot answer this *exactly*. They retain only the KL tails at the selected counts and release each client's full sorted score state after upload. Moreover, the two methods' global posteriors diverge after the first round; even in round 1 some per-client precision-floor fractions are not identical between the two independent Flower executions. **Do not assume that the historical joint and equal selected-tail values were computed from identical frozen scores.**

## Protocol-faithful diagnostic design

1. Read the existing **equal-bandwidth R20** run's `resolved_config.yaml`, content-hashed cached partition NPZ, and the **saved server checkpoint from round 10** (`checkpoints/global_round_0010.npz`). No modification to these files.
2. Reconstruct the server's FOLA `mean` and **raw precision** state from the checkpoint, validating every tensor name, shape, dtype and raw-precision constraint. At R11, choose the original seed, CIFAR-10 augmentation, learning-rate schedule and FOLA trainer as recorded; do not reset to R1.
3. Perform exactly **one `PREPARE` training pass per client** (20 total) through the installed `ClientEngine` and `LocalTransport`. The clients retain the trained posterior and their complete sorted KL scores in their in-memory client caches. A guarded trainer counter rejects repeated PREPARE.
4. Query all 20 client caches using the same existing `tail` and `price` instruction/response API. Run the production `allocate(... policy="equal_bandwidth")` and then `allocate(... policy="joint")` with **identical weights, bandwidth, time window, channel efficiencies and cached scores**. No repeated local training, and no retraining of one model per policy.
5. Compare selected count, per-client bandwidth, weighted discarded-KL contribution, modeled exact typed packet bytes, arithmetic data-plane energy and **canonical JSON allocation-request/reply control bytes**. Assert feasibility and `joint.objective <= equal.objective + tolerance` (the joint solver explicitly uses the equal-bandwidth allocation as a feasible incumbent).
6. Write only scalar result tables and hashes to a **new sibling directory outside the `bayesfl` repo**; automatically create a ZIP of those small files. No per-coordinate scores, raw local posteriors, model arrays, checkpoint contents or raw dataset are exported. The caches disappear on process exit.

**What `without retraining` means:** This experiment avoids rerunning historical R1–R10 or any new multi-round FL run, and avoids training *twice* for the two policies. Because the past runs did not save full per-client KL scores, one new **R11 local preparation** per client is unavoidably required. This is computational work, not a second global training round; the tool performs no upload/aggregation/global evaluation. A literal zero-training exact counterfactual is impossible from the two uploaded CSV/JSON archives alone.

The diagnostic uses a **sequential local emulator** rather than Flower/Ray RPC. The original local FOLA trainer, original `ClientEngine` logic and original scalar price/tail queries are used. The emulator shares one process for all clients, so it is not a privacy-preserving deployment. It validates the allocator's mathematics and logical control bytes, not hardware timing/energy or a new real-world radio transmission.

## Install (safe and additive)

Copy `FedKLS_CIFAR10_SamePosterior_Probe_v1.zip` to `~/DungNDH/micl/`.

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
for f in scripts/probe_cifar_same_posterior.py docs/wireless/SAME_POSTERIOR_PROBE.md tests/wireless/test_same_posterior_probe.py; do
  if [ -e "$f" ]; then echo "STOP: already exists: $f"; exit 1; fi
done
unzip -n ../FedKLS_CIFAR10_SamePosterior_Probe_v1.zip
```

## Test and preflight — no training

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
PY=/home/micl/anaconda3/envs/dungndh/bin/python
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -m pytest -q -rs tests/wireless/test_same_posterior_probe.py
"$PY" scripts/probe_cifar_same_posterior.py --preflight --checkpoint-round 10
```

The preflight refuses missing historical checkpoints, modified partition hashes, incorrect BasicCNN tensor shape/precision or incompatible original experiment. It must show `PASS: READ-ONLY PREFLIGHT`, `d = 878538`, `Client count / samples: 20 49988`, and `Source round: 10 -> probe PREPARE round 11`.

## Execute one real CIFAR-10 local-preparation pass

**Only after checking GPU availability and making sure other training jobs do not occupy GPU 0:**

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
pgrep -af 'bayesfl.main|_run_gpu_queue.sh|_run_wireless_queue.sh|probe_cifar_same_posterior.py' || true
nvidia-smi

PY=/home/micl/anaconda3/envs/dungndh/bin/python
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
OUT="$PWD/../bayesfl_cifar_same_posterior_R11_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/probe_cifar_same_posterior.py \
  --execute \
  --checkpoint-round 10 \
  --require-cuda \
  --output-dir "$OUT"
```

Expected success headline: `PASS: EXACT SAME-PREPARED-POSTERIOR ALLOCATION PROBE`. The tool prints both exact objectives, counts and model-array bytes, allocation-only control query bytes, price-query count and the output ZIP path. **No accuracy is generated or claimed.** The numerical outcomes must come from this actual run; do not assume the historical R20 objectives are same-posterior results.

Upload the new sibling `bayesfl_cifar_same_posterior_R11_<timestamp>.zip` (contains `summary.json`, `per_client.csv`, `allocation_query_events.csv`, and `README.txt`). These are sufficient for a detailed diagnosis without sharing model weights or raw scores.

## Interpretation and limitations

The joint optimizer is constructed to retain equal bandwidth as a feasible incumbent. Hence **joint weighted discarded KL cannot exceed equal bandwidth on the same scores**, up to numerical tolerance, assuming successful solver operation. The important empirical findings are *how much* it reduces KL, which clients' counts change, whether total bytes change, and the **marginal overhead required to compute that decision**. It does not certify a globally optimal integer allocation. The lower-bound/gap fields are finite-precision numerical diagnostics only.

Control bytes in this probe cover **allocation-phase scalar queries and replies**, counted with the same `canonical` encoding as the production controller. PREPARE control costs are separately reported; no actual uploads, releases, Flower envelopes, TCP/IP headers, RF airtime or physical device energy are modeled. Model-array and data-plane energy figures are what *would have been sent* under each decision in the existing ideal digital-link model, not observed data transmissions.

Use the R11 probe to avoid the exceptionally large KL magnitudes of R1 caused by zero initial precision being floored for scoring. The source checkpoint is taken from the **equal-bandwidth R10 global trajectory**; this is a deliberate single-anchor counterfactual, not a reproduction of historical joint R11 local scores. For robustness a later probe may use a different round or the joint run's own checkpoint, while keeping that probe's two policies evaluated against the *same* local preparation.

The original 180-second deadline, 1-MHz bandwidth and hypothetical edge-compute/control parameters remain **pilot assumptions**. No result of this test establishes a long-run accuracy advantage, client fairness, security, or real-world wireless performance. The original 24 nonwireless experiment endpoints are untouched.

## Package integrity and regression tests

This ZIP only adds the three paths above. `tests/wireless/test_same_posterior_probe.py` tests one PREPARE for every client, both allocations on one cached score state, schema/binding rejection, scalar-only control replies, and small output generation. Its validation is independent of the real CIFAR dataset, so it runs quickly before the GPU probe.
