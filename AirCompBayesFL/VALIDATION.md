# Validation status

Version: **1.0.2**

The package was validated in the artifact-build environment as follows:

```text
python -m compileall .
python -m pytest -q tests/test_aircomp.py tests/test_aggregation.py \
  tests/test_model_size.py tests/test_runtime_utils.py
4 passed
```

Validated components:

- Python syntax for all modules
- paper CNN parameter count (`d = 62,346`)
- noiseless/unconstrained AirComp weighted aggregation
- Gaussian precision/natural-mean conflation without wireless distortion
- CPU ServerApp pin-memory protection

The artifact-build environment does not expose an NVIDIA GPU and does not
contain Flower, Ray, or Pyro. Therefore an end-to-end CUDA Flower/Ray/Pyro run
was not executed during packaging. The target-machine preflight command is:

```powershell
.\.venv\Scripts\python.exe gpu_check.py --config configs/smoke_gpu.yaml
```
