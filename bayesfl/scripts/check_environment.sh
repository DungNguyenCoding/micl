#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import platform
import sys
import torch
print('python   :', sys.version.split()[0])
print('platform :', platform.platform())
print('torch    :', torch.__version__)
print('cuda     :', torch.version.cuda)
print('cudnn    :', torch.backends.cudnn.version())
print('cuda ok  :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu      :', torch.cuda.get_device_name(0))
try:
    import flwr
    print('flwr     :', flwr.__version__)
except Exception as exc:
    print('flwr     : ERROR', exc)
try:
    import bayesian_torch
    print('bayesian_torch: installed')
except Exception as exc:
    print('bayesian_torch: ERROR', exc)
PY
