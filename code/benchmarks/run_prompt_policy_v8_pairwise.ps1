param(
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$trainCandidates = Join-Path $root "runs\research_direction\prompt_policy_oracle\train_multiref5_reduced_cpu_dinov2l_samvitb\candidates.csv"

while (-not (Test-Path -LiteralPath $trainCandidates)) {
    Write-Host "[wait] pairwise v8 is waiting for 145-episode candidates"
    Start-Sleep -Seconds $PollSeconds
}

Set-Location $root
$outDir = "runs\research_direction\prompt_policy\adaptive_policy_models_v8_pairwise_multiref5"
python -B code\benchmarks\train_pairwise_prompt_ranker.py `
    --train-candidates $trainCandidates `
    --val-candidates runs\research_direction\prompt_policy_oracle\val_grid_dinov2l_samvitb\candidates.csv `
    --test-candidates runs\research_direction\prompt_policy_oracle\test_grid_dinov2l_samvitb_fixedr3\candidates.csv `
    --allowed-rounds round2 `
    --allowed-k 5,10,20,40,75 `
    --allowed-lambda 0.25,0.75,1.5 `
    --candidate-budget 3 `
    --out-dir $outDir
if ($LASTEXITCODE -ne 0) { throw "Pairwise v8 training failed" }

python -B code\benchmarks\analyze_prompt_policy_results.py `
    --fixed "$outDir\test_fixed_predictions.csv" `
    --learned "$outDir\test_pairwise_predictions.csv" `
    --out-dir "$outDir\paired_analysis"
if ($LASTEXITCODE -ne 0) { throw "Pairwise v8 paired analysis failed" }
