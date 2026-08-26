# AirCompBayesFL v1.6.2 — CIFAR-10 + integrated sparse final-accuracy plot

This upgrade is designed to be applied to the exact local v1.6.1 source tree.
It clones that tree and leaves the original untouched.

## What is added

1. **CIFAR-10 opt-in execution path**
   - `main_cifar10.py`
   - `cifar10_support.py`
   - native RGB 32x32 CIFAR-10
   - CIFAR normalization: mean `(0.4914,0.4822,0.4465)`, std `(0.2470,0.2435,0.2616)`
   - no augmentation in the first controlled experiment
   - model: Conv(3->32,5), pool/ReLU, Conv(32->64,5), pool/ReLU, Linear(1600->10)
   - **69,706 trainable parameters**
   - F=1024 => 69 OFDM groups/phase, 138 groups/dense Proposed logical round

2. **Merged final sparse plot inside `utils.py`**
   New mode:

   ```powershell
   .\.venv\Scripts\python.exe utils.py `
     --input results\sparse_proposed_rep3 `
     --figure sparse-final-accuracy `
     --dense-baseline results\fig2_proposed_rep1 `
     --dense-baseline results\sparse_dense_extra_12026_12027 `
     --dense-baseline-round 160
   ```

   It plots final Bayesian/Random sparse accuracy averaged across seeds and ONE grey dashed
   dense-100% line based on the per-seed best dense accuracy up to the target round.

3. **CIFAR configs** copied from the existing local configs so all learning/wireless
   hyperparameters remain the same unless you edit them.

4. `run_cifar10_sparse_rep3.ps1` for the complete 3-seed CIFAR experiment.

## Why the CIFAR path is isolated

The paper reproduction uses MNIST.  `main.py`, the original MNIST dataset path, Proposed
VI, AirComp, wireless code, and old configs are not modified.  CIFAR is activated only by
running `main_cifar10.py`.  This prevents CIFAR support from changing your existing paper
simulation results.

## Build the full v1.6.2 source tree

From `C:\Users\Admin\Desktop\micl` after extracting this upgrade kit:

```powershell
.\AirCompBayesFL\.venv\Scripts\python.exe `
  .\AirCompBayesFL_v1.6.2_UpgradeKit\build_v162.py `
  --source .\AirCompBayesFL `
  --dest .\AirCompBayesFL_v1.6.2
```

The destination is a full cloned source tree (excluding `.venv`, results, data, caches).
Your original `AirCompBayesFL` remains unchanged.

To use the existing venv without copying it:

```powershell
cd .\AirCompBayesFL_v1.6.2
..\AirCompBayesFL\.venv\Scripts\python.exe main_cifar10.py --help
```

If you prefer the usual `.\.venv\...` commands, create a Windows junction after build:

```powershell
cmd /c mklink /J .venv ..\AirCompBayesFL\.venv
```

## Smoke test CIFAR-10 first

```powershell
..\AirCompBayesFL\.venv\Scripts\python.exe main_cifar10.py `
  --config configs\proposed_cifar10_gpu.yaml `
  --experiment fig2 `
  --methods proposed `
  --rounds 2 `
  --replications 1 `
  --seed 12025 `
  --path-loss-reference-m 1000 `
  --output results\cifar10_smoke
```

The first execution downloads CIFAR-10 through torchvision.

## Full 3-seed dense + sparse CIFAR experiment

After creating the `.venv` junction, simply run:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_cifar10_sparse_rep3.ps1
```

This performs:
- 3 dense Proposed runs (100%, one per seed, shared by Bayesian and Random),
- sparse Bayesian/Random for 75, 50, 25, 10, 5, 2%, 3 seeds,
- existing sparse plots,
- final accuracy vs keep ratio with one shared dense grey dashed line.

## Scientific note

This CIFAR experiment is an extension, not a reproduction of the target paper.  The target
paper's reported simulations use MNIST.  CIFAR-10 is intentionally used here to test whether
the sparse posterior ranking generalizes to a harder image dataset while keeping the FL and
wireless settings otherwise as close as possible.
