# Locked simulation profiles

**Current choice, 2026-10-26:** main paper plots use MNIST **s10 / seed 0 / N=20** and CIFAR-10 **alpha=.01 / local-data fraction 1.0 / seed 0 / N=20**, with a **dense FOLA R150 budget for each**. This choice supersedes old MLP/N100, ResNet and R120 defaults. Evidence and limits: [EVIDENCE_AND_GAPS.md](EVIDENCE_AND_GAPS.md).

## Main settings

| Field | MNIST primary | CIFAR-10 primary |
| --- | --- | --- |
| Model label | CNN 16–32–FC128 | BasicCNN |
| Config model identifier | `mnist_cnn_16_32_128` (conversation-reported; validate live factory) | `paper_basiccnn` |
| Scalar parameter dimension d | 80,202 | 878,538 |
| Clients / participating each round | 20 / 20 | 20 / 20 |
| Partition | `fixed_labels`, samples_per_client=10, labels_per_client=1 | `paper_dirichlet`, dirichlet_alpha=0.01, local_data_fraction=1.0 |
| Seed | 0 | 0 |
| Local epochs E | 5 | 10 |
| Batch B | 10 | 32 |
| Optimizer | SGD | SGD |
| Momentum / weight decay | 0 / 0 | 0 / 0 |
| Dense and sparse FOLA LR | 0.10, constant | 0.02 base, cosine, minimum 0.0001, horizon 400 |
| FedAvg LR | 0.10, constant | 0.01, constant |
| FOLA prior coefficient | 0.01 | 0.01 |
| FOLA mode | `paper_reference` | `paper_reference` |
| Dense FOLA clipping | Disabled (Stage-18) | Norm 10 |
| KL / random FOLA clipping | Disabled (Stage-22) | Norm 10 |
| Primary plotted FedAvg clipping | Norm 10 (recovered Stage-23) | Norm 10 |
| Additional FedAvg control | Stage-16 unclipped, not the chosen primary curve | Not separately specified here |
| Full local model training | Yes; compress only after local fit | Yes; compress only after local fit |
| Uplink / downlink | Sparse selected mean–precision pairs + bitmap / dense multicast | Same |
| Keep ratios r | 0.75, 0.50, 0.25, 0.10, 0.05 | Same |
| Selectors | `kl_global_local` and `random` | Same |
| Dense / FedAvg plotted endpoint | R150 / R300 | R150 / R300 |
| Exact total budget | 2,021,090,400 bytes | 22,139,157,600 bytes |

No field may be retuned silently. **E <= 10** remains the user's compute limit, not a claim that every historical validator enforces it.

### Partition distinctions

For MNIST, do not include a Dirichlet alpha or local-data-fraction setting as a substitute for fixed labels. Under the recorded fixed-label allocator, N=20 and ten examples/client gives 200 assigned examples and two clients/digit; verify the saved partition's uniqueness and class allocation before reproducing. Changing to five examples/client is a separate condition, not a nested-data guarantee.

For CIFAR, `local_data_fraction=1.0` means no additional per-client subsampling after base allocation. It does **not** mean “all clients participate”; participation is independently 20/20. The historical paper-Dirichlet allocator floors class allocations and can backfill empty clients. Do not assert exactly 50,000 unique, disjoint assignments or equal 2,500 samples/client without the actual partition metadata.

### Additional implementation settings and their evidence boundary

The documented FOLA contract uses `lambda_scale_by_size=false`, `paper_mean_only_eval=true`, and `output.save_full_client_posteriors=false`. Full untransmitted local posteriors must not be secretly sent for a sparse run.

The older inspected CIFAR template uses initial omega 0.0, aggregation epsilon 1e-5, precision_min=1e-12, precision_max=1e12, `precision_policy=floor_for_score`, and `score_precision_floor=null` (resolves to precision_min). Its augmentation is crop(32,padding=4,fill=128), horizontal flip, CIFAR AutoAugment, tensor Cutout(1,16), normalization mean/std=(.5,.5,.5); test evaluation omits stochastic augmentation. Preserve the selected live resolved configuration, not an old template's obsolete byte cap.

The historical MNIST baseline used initial omega 1.0, epsilon 1e-5 and precision_min=1e-8. These are **inherited documented values, not freshly verified Stage-18/22 YAML fields**. Read current resolved YAMLs before reporting all precision/numerical/runtime settings as verified.

The current MNIST CNN module is `src/bayesfl/models/mnist_cnn.py`. Its existence is recorded by the inventory, but its current body is not in the older source ZIP. Do not invent kernels, pooling, normalization or initialization from the 80,202 count. The inspected CIFAR BasicCNN is Conv(3,32,5)–ReLU–Pool(2), Conv(32,64,5)–ReLU–Pool(2), FC(1600,512)–ReLU–FC(512,10), with biases and deterministic Xavier initialization.

## Training caps versus evaluation checkpoints

Stage-18 dense FOLA runs through R200, but the chosen plot reads **R150**. The recovered MNIST Stage-23 FedAvg runs through R400, but the chosen plot reads **R300**. CIFAR dense FOLA R150 and FedAvg R300 are available per the cleanup manifest. Run-name suffixes may reflect the original cap, not the last resumed round.

CIFAR cosine H=400 is independent of the byte budget and total cap. Do not change H to 150, 499 or 1362 when extending a run. The recorded one-based schedule is

```text
j = min(round - 1, H - 1)
lr = lr_min + 0.5*(lr_base-lr_min)*(1 + cos(pi*j/(H-1)))
```

At H=1 the implementation uses lr_min; after H it stays at lr_min. This is the inspected snapshot behavior; inspect current code before editing it.

## Preserved alternative, not the current primary MNIST panel

MNIST s5: N=20, exactly one class/client, seed **3**, E=5, B=5, LR=.10 constant, lambda=.01. Dense FOLA is unclipped Stage-18; sparse KL/random is unclipped Stage-21; plotted historical FedAvg is clipped Stage-15. Budget and sparse targets are the same as s10 because d and K are unchanged. The Stage-15 name says r300 but the cleanup record reports its extension to R400.

Stage-24 clipped dense FOLA s10/seed0 is a protected ablation, **not** the selected dense baseline. N=100 and other seeds remain historical or resume-sensitive research controls. No new multiseed, tuning or wireless jobs are authorized by this document.

Exact outputs and candidate source YAML paths: [EXPERIMENT_REGISTRY.json](EXPERIMENT_REGISTRY.json) and [RUN_REGISTRY.md](RUN_REGISTRY.md).
