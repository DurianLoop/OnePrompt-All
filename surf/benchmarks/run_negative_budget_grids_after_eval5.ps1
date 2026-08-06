param(
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$eval5Test = Join-Path $root "runs\research_direction\prompt_policy_oracle\test_multiref5_eval5_dinov2l_samvitb\candidates.csv"

while (-not (Test-Path $eval5Test)) {
    Write-Host "[wait] negative-budget grids are waiting for eval5 inference"
    Start-Sleep -Seconds $PollSeconds
}

Set-Location $root
python -B code\benchmarks\evaluate_negative_prompt_ablation.py `
    --manifest runs\research_direction\prompt_policy_splits\monuseg_strict_multiref5\episodes_train.csv `
    --candidates runs\research_direction\prompt_policy_oracle\train_multiref5_reduced_cpu_dinov2l_samvitb\candidates.csv `
    --fixed-action contrast_topk_K40_lam0.75 `
    --negative-counts 0,5,10,20 `
    --sam-device auto `
    --minimal-artifacts `
    --resume `
    --out-dir runs\research_direction\prompt_policy_oracle\negative_budget_train_multiref5
if ($LASTEXITCODE -ne 0) { throw "Train negative-budget grid failed" }

python -B code\benchmarks\evaluate_negative_prompt_ablation.py `
    --manifest runs\research_direction\prompt_policy_splits\monuseg_strict_official\episodes_val.csv `
    --candidates runs\research_direction\prompt_policy_oracle\val_grid_dinov2l_samvitb\candidates.csv `
    --fixed-action contrast_topk_K40_lam0.75 `
    --negative-counts 0,5,10,20 `
    --sam-device auto `
    --minimal-artifacts `
    --resume `
    --out-dir runs\research_direction\prompt_policy_oracle\negative_budget_val
if ($LASTEXITCODE -ne 0) { throw "Validation negative-budget grid failed" }
