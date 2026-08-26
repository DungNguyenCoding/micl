param(
    [int]$Rounds = 120,
    [int]$Replications = 1,
    [int]$Seed = 12025,
    [double]$PathLossReferenceM = 1000,
    [string]$Output = "results\sparse_proposed_rep1",
    [string]$DenseBaseline = "results\fig2_proposed_rep1"
)

$ErrorActionPreference = "Stop"

.\.venv\Scripts\python.exe main.py `
  --config configs/sparse_proposed_gpu.yaml `
  --experiment sparse `
  --methods proposed `
  --rounds $Rounds `
  --replications $Replications `
  --seed $Seed `
  --path-loss-reference-m $PathLossReferenceM `
  --output $Output

.\.venv\Scripts\python.exe utils.py `
  --input $Output `
  --figure sparse `
  --dense-baseline $DenseBaseline `
  --dense-baseline-round $Rounds
