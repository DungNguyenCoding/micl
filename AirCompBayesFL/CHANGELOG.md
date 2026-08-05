# Changelog

## 1.1.0

- Added `runtime.backend: auto|ray|local`.
- Native Windows + CUDA now defaults to a stable in-process sequential GPU backend.
- Linux/WSL2 retains Flower/Ray virtual-client execution.
- Added backend-neutral client payload aggregation.
- Added fail-fast behavior for Flower client failures and zero-result rounds.
- Cleared Pyro's parameter store after each Bayesian client fit.
- Moved client models back to CPU before CUDA allocator cleanup.
- Added new RTX 3060 GPU profiles and output directories.
- Updated GPU preflight and Windows documentation.

## 1.0.2

- Added device-aware DataLoader pinning and native-Windows Ray cleanup.
