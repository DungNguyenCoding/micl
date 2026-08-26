param(
    [int]$Rounds = 160,
    [int]$Replications = 3,
    [int]$Seed = 12025,
    [double]$PathLossReferenceM = 1000,
    [string]$DenseOutput = "results\cifar10_dense_rep3",
    [string]$SparseOutput = "results\cifar10_sparse_proposed_rep3"
)

$ErrorActionPreference = "Stop"
$Python = ".\.venv\Scripts\python.exe"

Write-Host "=== CIFAR-10 dense Proposed: one 100% run per seed ===" -ForegroundColor Cyan
& $Python .\main_cifar10.py `
  --config configs\proposed_cifar10_gpu.yaml `
  --experiment fig2 `
  --methods proposed `
  --rounds $Rounds `
  --replications $Replications `
  --seed $Seed `
  --path-loss-reference-m $PathLossReferenceM `
  --output $DenseOutput `
  --resume

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== CIFAR-10 sparse Bayesian vs Random ===" -ForegroundColor Cyan
& $Python .\main_cifar10.py `
  --config configs\sparse_proposed_cifar10_gpu.yaml `
  --experiment sparse `
  --methods proposed `
  --rounds $Rounds `
  --replications $Replications `
  --seed $Seed `
  --path-loss-reference-m $PathLossReferenceM `
  --output $SparseOutput `
  --resume

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Existing v1.6.1 sparse plots ===" -ForegroundColor Cyan
& $Python .\utils.py `
  --input $SparseOutput `
  --figure sparse `
  --dense-baseline $DenseOutput `
  --dense-baseline-round $Rounds

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Final accuracy vs keep ratio (3-seed average) ===" -ForegroundColor Cyan
& $Python .\utils.py `
  --input $SparseOutput `
  --figure sparse-final-accuracy `
  --dense-baseline $DenseOutput `
  --dense-baseline-round $Rounds

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Done." -ForegroundColor Green
Write-Host "Dense results : $DenseOutput"
Write-Host "Sparse results: $SparseOutput"
Write-Host "Plots         : $SparseOutput\plots"
