param(
    [string]$PythonExe = "D:\conda\python.exe",
    [switch]$WaitForCandidates,
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"
$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$trainCandidates = Join-Path $workspace "runs\research_direction\prompt_policy_oracle\train_multiref5_reduced_cpu_dinov2l_samvitb\candidates.csv"
$valCandidates = Join-Path $workspace "runs\research_direction\prompt_policy_oracle\val_grid_dinov2l_samvitb\candidates.csv"
$testCandidates = Join-Path $workspace "runs\research_direction\prompt_policy_oracle\test_grid_dinov2l_samvitb_fixedr3\candidates.csv"
$outDir = Join-Path $workspace "runs\research_direction\prompt_policy\adaptive_policy_models_v7_multiref5"
$episodes = Join-Path $workspace "runs\research_direction\prompt_policy_splits\monuseg_strict_official\episodes_test.csv"

if ($WaitForCandidates) {
    while (-not (Test-Path -LiteralPath $trainCandidates)) {
        Write-Host "[wait] train candidates are not complete; checking again in $PollSeconds seconds"
        Start-Sleep -Seconds $PollSeconds
    }
}

if (-not (Test-Path -LiteralPath $trainCandidates)) {
    throw "Missing train candidates: $trainCandidates"
}

$candidateCount = (Import-Csv -LiteralPath $trainCandidates).Count
if ($candidateCount -ne 2320) {
    throw "Expected 2320 train candidates, found $candidateCount. Refusing to train on a partial grid."
}

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Push-Location $workspace
try {
    & $PythonExe -B code\benchmarks\train_adaptive_prompt_policy.py `
        --train-candidates $trainCandidates `
        --val-candidates $valCandidates `
        --test-candidates $testCandidates `
        --allowed-rounds round2 `
        --allowed-k 5,10,20,40,75 `
        --allowed-lambda 0.25,0.75,1.5 `
        --anchor-uncertainty-weights 0 `
        --out-dir $outDir
    if ($LASTEXITCODE -ne 0) { throw "v7 policy training failed with exit code $LASTEXITCODE" }

    & $PythonExe -B code\benchmarks\analyze_prompt_policy_results.py `
        --fixed (Join-Path $outDir "test_fixed_predictions.csv") `
        --learned (Join-Path $outDir "test_anchored_predictions.csv") `
        --out-dir (Join-Path $outDir "paired_analysis")
    if ($LASTEXITCODE -ne 0) { throw "v7 paired analysis failed with exit code $LASTEXITCODE" }

    & $PythonExe -B code\benchmarks\analyze_policy_oracle_gap.py `
        --candidates $testCandidates `
        --selected (Join-Path $outDir "test_anchored_predictions.csv") `
        --fixed (Join-Path $outDir "test_fixed_predictions.csv") `
        --shortlist (Join-Path $outDir "test_anchored_shortlist.csv") `
        --allowed-rounds round2 `
        --allowed-k 5,10,20,40,75 `
        --allowed-lambda 0.25,0.75,1.5 `
        --out-dir (Join-Path $outDir "regret_analysis")
    if ($LASTEXITCODE -ne 0) { throw "v7 regret analysis failed with exit code $LASTEXITCODE" }

    & $PythonExe -B code\benchmarks\visualize_prompt_policy_decisions.py `
        --fixed (Join-Path $outDir "test_fixed_predictions.csv") `
        --learned (Join-Path $outDir "test_anchored_predictions.csv") `
        --episodes $episodes `
        --out-dir (Join-Path $outDir "qualitative") `
        --max-cases 6
    if ($LASTEXITCODE -ne 0) { throw "v7 qualitative rendering failed with exit code $LASTEXITCODE" }

    Write-Host "[done] v7 artifacts: $outDir"
}
finally {
    Pop-Location
}
