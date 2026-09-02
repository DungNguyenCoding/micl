$ErrorActionPreference = "Stop"

$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Virtual environment not found: $Python"
}

& $Python (Join-Path $PSScriptRoot "gpu_check.py") --config configs/smoke_gpu.yaml
& $Python (Join-Path $PSScriptRoot "main.py") `
    --config configs/smoke_gpu.yaml `
    --experiment fig2 `
    --methods fedavg,proposed
