# Validation

Version: 1.1.0

Completed in the packaging environment:

- `python -m compileall -q .`
- `PYTHONPATH=. pytest -q`
- Result: 7 tests passed.
- YAML parsing and backend resolution checks completed.

The packaging environment has CPU-only PyTorch and does not expose Flower, Ray, Pyro, or an NVIDIA GPU, so an end-to-end CUDA run could not be performed there. The native-Windows fix avoids the failing Ray CUDA worker path and runs local clients sequentially in the launcher process, where the user's independent CUDA check already succeeds.
