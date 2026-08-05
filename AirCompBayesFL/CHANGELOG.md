# Changelog

## 1.0.2 — Windows GPU reliability update

- Fixed CPU ServerApp evaluation incorrectly requesting pinned CUDA memory.
- Added device-aware `should_pin_memory` handling for server and clients.
- Added CUDA/Ray preflight validation and `gpu_check.py`.
- Added best-effort CUDA cache cleanup after each virtual-client fit.
- Moved Ray shutdown into a `finally` block so crashes do not keep workers alive.
- Added `configs/smoke_gpu.yaml` and `configs/paper_gpu.yaml` for an RTX 3060 Laptop GPU.
- Kept native-Windows server evaluation on CPU and GPU training on Ray ClientApps.
