# Wireless FedKLS v1 — implemented digital simulation path

**2026-11-09 · protocol `fedkls_wireless_integer_precision_v2`.** This directory
supersedes the earlier preimplementation review as the status authority. The supplied
proposal is preserved under [proposal](proposal/README.md); review material is under
[preimplementation](preimplementation/README.md).

## Implemented scope

The opt-in runner performs **PREPARE → scalar TAIL/PRICE requests → exact UPLOAD grant
→ packet validation/prior fill → RELEASE → aggregation/evaluation → commit**.
Every participating client trains the full model once per round with the existing trainer.
Prepared state survives ephemeral Flower client objects via `Context.state`; the local
backend uses an explicit client-owned memory store. Only scalar replies and granted
posterior packets are visible through the server interface.

Implemented policies: joint KL threshold allocation (Version B); joint counts with random
coordinates; equal-bandwidth maximal feasible counts (Version A); uniform feasible count;
dense FOLA; dense FedAvg. All use the same declared fixed-slot policy. Successful digital
transmission leaves raw float32 values unchanged. No AirComp, noisy weight perturbation,
client scheduling/dropout, adaptive local epochs, sparse downlink, quantization, error
feedback or new learning objective is added.

Read [IMPLEMENTATION.md](IMPLEMENTATION.md) for mechanism/adaptations,
[CONFIGURATION.md](CONFIGURATION.md) for fields and scenario decisions,
[RUNBOOK.md](RUNBOOK.md) for commands, [OUTPUTS.md](OUTPUTS.md) for results/resume,
[VALIDATION.md](VALIDATION.md) for actual tests, and [STATUS.md](STATUS.md) for open gates.

## Compatibility

`wireless` absent or disabled leaves the legacy serialized configuration/fingerprint
unchanged. Existing packets/byte-budget stopping/resume and plots remain on their original
path. Wireless uses exact counts directly, a new packet version, separate accounting and
`wireless_resume/`. Never resume a legacy checkpoint through the wireless controller.
Existing source docs before this release are under `../history/pre_wireless_v1/`.

## Runtime boundary

The local CPU backend and synthetic integrations were executed. The real Flower/Ray
adapter and its tests are supplied but could not be executed in the authoring environment.
Do not call a skipped Flower test a pass. Run the target-environment gates before
launching dataset studies. No new MNIST/CIFAR wireless accuracy result is claimed.

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
