# Validation actually performed — wireless v1

Authoring check date: 2026-11-09. See repository-root `validation/wireless/` for logs.

| Check | Observed result |
|---|---|
| Full available legacy + wireless pytest | **186 passed, 6 skipped** |
| Wireless-only tests within the suite | 58 passed; real-Flower module skipped |
| Original YAML load / serialized config / legacy fingerprint | **891 / 891 identical**, no original config edits |
| Primary-paper CSV endpoints and exact byte totals | **24 / 24 pass** |
| Synthetic CPU complete multi-phase CLI | Six policies, three rounds each; original trainers |
| Clean evaluated-round resume | All six policies' local interrupted-at-commit versus uninterrupted state/bytes match |
| Current MNIST CNN + CentralEvaluator | Synthetic 28x28 image test; real runner + evaluated resume pass |
| Integer oracle / feasibility tests | Tiny exhaustive spaces, ties, zero tails, padded sizes, dense switch, finite limits |
| Dynamic reporting / export | Exact-count packet checks, malformed-cost rejection, hash-checked explicit wireless-only ZIP |
| Study generation | Both locked profiles preserve all inherited data/model/runtime/FOLA and training fields except explicit total rounds |
| Source Python 3.10 grammar | 142 active Python files parse; not Python 3.10 runtime certification |
| New queue Bash syntax | `bash -n` passes |
| Overlay installer | 9 checks pass on temporary repositories: preview, confirmation, backup/apply/verify, idempotence, prior-review compatibility and refusal guards |
| Original trainers/models/partition/reducer/legacy protocol/dependencies/primary files | Byte-preserved; exact record in baseline_preservation.json |

The six skips are three missing Bayesian-Torch tests, the original Flower smoke, the
original sparse Flower integration module, and the new wireless real-Flower module.
**A skipped module does not validate its contained scenarios.** No fake Flower package
was used as a substitute. No actual Flower/Ray/CUDA call was executed in this environment.

Authoring runtime: Python3.13.5, Torch2.10.0+cpu, Torchvision0.25.0+cpu, NumPy2.3.5,
PyYAML6.0.3, pytest9.0.2. Flower, Ray and Bayesian-Torch were absent. Target environment
remains the supplied Python3.10/Flower1.29/Bayesian-Torch0.5 working CUDA stack.
No dependency manifest was modified to fit the authoring environment.

## What synthetic checks do not establish

No real-dataset wireless training, generalization advantage, statistically significant
accuracy improvement, physically measured latency/energy, true radio framing cost,
universal integer optimum, power-loss atomicity or target-GPU performance was established.
Dense cases in the synthetic CLI use a higher bandwidth solely to cover the dense code;
they are not matched-scenario scientific comparisons. Numerical dual gaps are diagnostics.

## Target-environment gates

```bash
cd ~/DungNDH/micl/bayesfl || exit 1
PY=/home/micl/anaconda3/envs/dungndh/bin/python
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -m pytest -q tests/wireless
"$PY" -m pytest -q
"$PY" -m pytest -q tests/wireless/test_flower.py
CUDA_VISIBLE_DEVICES="" "$PY" scripts/smoke_wireless_flower.py \
  --output-dir "outputs/smoke_wireless_ray_$(date +%Y%m%d_%H%M%S)" --rounds 2
"$PY" docs/tools/export_handoff.py --check
```

Only after those runtime checks should a separately named two-round real dataset/CUDA
smoke be launched with a reviewed scenario. Record the actual output, environment,
file versions and any failures in this document; do not overwrite historical evidence.

<!-- BEGIN JOINT_FEDKLS_CIFAR10_R150_VERIFIED_20261110 -->
### 2026-11-10 — Joint-only CIFAR-10 R150 audit (supersedes earlier pending CIFAR/Flower statuses)

- **Primary research method:** FedKLS joint KL-guided bandwidth and integer coordinate retention. Equal bandwidth is a historical engineering pilot, **not the focus** of the current research.
- **Verified run:** `run_wireless_cifar_eqb_joint150_fixed_v1_cifar10_a0p01_f1_joint_seed0_20261109_230207`; **R150 accuracy 52.63%**, NLL 1.40449; best 52.76% at R148; final-20-round mean 52.1115%.
- **Source audit:** `PASS_ENGINEERING_RECORD_INTEGRITY`; 150 rounds, 3,000 client-round rows, 5,579 price queries, 11,438,601,248 model-array bytes and 111,770,677 control bytes. All 161 included exported-evidence hashes independently verified.
- **Limits:** One CIFAR-10 seed 0 and fixed wireless path-loss scenario; no globally certified integer solution; power/control/compute times remain modeled, not physical measurements; no independent checkpoint-tensor examination.
- **Evidence & detailed findings:** [CIFAR-10 Joint R150 Review](r150_joint_20261110/JOINT_R150_REVIEW.md) (includes control breakdown, retention, learning curves and normalized KL contribution correction).
- **Derived audit caveat:** Previously generated per-client `weighted_discarded_kl` audit CSV contains `n_k*tail` rather than `(n_k/N)*tail`; the production round objective is correct. Use the corrected per-client CSV included in the R150 review package.

<!-- END JOINT_FEDKLS_CIFAR10_R150_VERIFIED_20261110 -->

<!-- BEGIN FEDKLS_JOINT_SENSITIVITY_R150_V1 -->
## Joint CIFAR-10 wireless sensitivity R150 — verified 2026-11-12

- Main method: **joint-KL** bandwidth/retention; 20 clients, CIFAR BasicCNN, seed 0, fixed channel gains, 150 rounds. The original joint reference (Bmax=1 MHz, U=58.973 s) reached **52.63%**. Six additional fixed-channel R150 cases passed engineering record-integrity audits.
- **U sweep** at Bmax=1 MHz: U=48.973 s → 50.25%; U=38.973 s → 52.47%; U=28.973 s → 43.43%.
- **B sweep** at U=58.973 s: Bmax=0.55 MHz → 47.29%; 0.75 MHz → 52.53%; 0.95 MHz → 54.01%.
- All seven source-audit ZIPs independently matched their **161 SHA-256 evidence hashes**. Configurations match except run_name, uplink bandwidth and round target. The R150 learning outcomes are **non-monotone**, and the two harshest cases show pronounced learning oscillations.
- Price-search `price_tolerance` in U150 **21/150** and B0550 **27/150** rounds. This is a solver stopping caution, not an audit failure; global integer optimality is not certified.
- This is a **single-seed engineering simulation**. The reference scenario is `EXAMPLE_NOT_LOCKED`, not physically calibrated. Shorter U also means shorter modeled round duration; compare both accuracy vs rounds and modeled time.

**Full verified audit and results:** `docs/wireless/JOINT_CIFAR10_SENSITIVITY_R150_20261112.md`  
**Reproducible plotting guide:** `docs/wireless/JOINT_SENSITIVITY_PLOTS_RUNBOOK_v1.md`  
**Plot script:** `scripts/plot_joint_cifar_sensitivity_v1.py` (generates 4 side-by-side PNG/PDF figures + CSVs without retraining).
<!-- END FEDKLS_JOINT_SENSITIVITY_R150_V1 -->
