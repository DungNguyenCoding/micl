# Current wireless status / next-chat checkpoint

**Release v1, 2026-11-09.** Implemented: numerical core, padded integer packet v2,
strict opt-in parsing, current-round prepare/query/grant lifecycle, client Context cache,
local and Flower/Ray adapters, dynamic byte/control/time/energy logs, completed-round
resume, six policy cases, study generator/two queues, terminal summaries/new plots,
and explicit wireless artifacts in lightweight handoffs.

Executed locally: numerical/protocol tests, synthetic CPU end-to-end cases, local
uninterrupted/resumed equivalence, actual MNIST CNN and CentralEvaluator on synthetic
images, generator invariance and lightweight export tests, legacy suite/24 endpoint
checks. Read VALIDATION.md for exact counts. None are a new MNIST/CIFAR performance run.

**Not yet executed:** real Flower1.29/Ray adapter on the user's environment, CUDA, real
dataset wireless training, long queues, or physical radio experiments. The supplied
real Flower test and Ray smoke are runtime gates, not claimed successful results.

**Not locked:** physical bandwidth/gains/PSD/noise, device preparation coefficients,
round deadline/control allowance, total modeled-time budget and experimental seeds for
wireless. `scenario_example.yaml` is an engineering illustration. The old learning
profiles are retained but do not supply these physical parameters.

Not implemented by design: AirComp, fixed-total-power bandwidth-dependent rate solver,
client scheduling, retransmissions, adaptive local compute, mid-round recovery, privacy
mechanism, certified finite-precision optimality, or arbitrary legacy-checkpoint import.
A future extension must state any departure from this contract explicitly.

Next action: install/verify, run CPU tests, run actual Flower/Ray smoke, approve a physical
scenario, then a separately named short dataset smoke before long comparative runs.
Preserve all finished legacy paper outputs and current plot command.

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
