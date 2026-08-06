param(
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$trainNegative = Join-Path $root "runs\research_direction\prompt_policy_oracle\negative_budget_train_multiref5\per_case.csv"
$valNegative = Join-Path $root "runs\research_direction\prompt_policy_oracle\negative_budget_val\per_case.csv"
$testNegative = Join-Path $root "runs\research_direction\mechanisms\negative_prompt_ablation_test14\per_case.csv"
$outDir = Join-Path $root "runs\research_direction\prompt_policy\adaptive_negative_budget_v1"

while (-not (Test-Path $trainNegative) -or -not (Test-Path $valNegative)) {
    Write-Host "[wait] negative-budget policy is waiting for train/val grids"
    Start-Sleep -Seconds $PollSeconds
}

Set-Location $root
python -B code\benchmarks\train_negative_budget_policy.py `
    --train-negative $trainNegative `
    --val-negative $valNegative `
    --test-negative $testNegative `
    --train-candidates runs\research_direction\prompt_policy_oracle\train_multiref5_reduced_cpu_dinov2l_samvitb\candidates.csv `
    --val-candidates runs\research_direction\prompt_policy_oracle\val_grid_dinov2l_samvitb\candidates.csv `
    --test-candidates runs\research_direction\prompt_policy_oracle\test_grid_dinov2l_samvitb_fixedr3\candidates.csv `
    --out-dir $outDir
if ($LASTEXITCODE -ne 0) { throw "Negative-budget policy training failed" }

python -B code\benchmarks\analyze_prompt_policy_results.py `
    --fixed (Join-Path $outDir "test_fixed_predictions.csv") `
    --learned (Join-Path $outDir "test_learned_predictions.csv") `
    --out-dir (Join-Path $outDir "paired_analysis")
if ($LASTEXITCODE -ne 0) { throw "Negative-budget paired analysis failed" }
