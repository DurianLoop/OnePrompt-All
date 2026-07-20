param(
    [int]$FoldStart = 0,
    [int]$FoldEnd = 4,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$splitRoot = Join-Path $root "runs\research_direction\prompt_policy_splits\monuseg_nested_cv5"
$runRoot = Join-Path $root "runs\research_direction\nested_cv5_prompt_policy"

if ($FoldStart -lt 0 -or $FoldEnd -gt 4 -or $FoldStart -gt $FoldEnd) {
    throw "Fold range must satisfy 0 <= FoldStart <= FoldEnd <= 4."
}

Set-Location $root
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

for ($fold = $FoldStart; $fold -le $FoldEnd; $fold++) {
    $foldSplit = Join-Path $splitRoot "fold_$fold"
    $foldRun = Join-Path $runRoot "fold_$fold"
    foreach ($split in @("train", "val", "test")) {
        $manifest = Join-Path $foldSplit "episodes_$split.csv"
        $candidateDir = Join-Path $foldRun "candidates_$split"
        $candidateCsv = Join-Path $candidateDir "candidates.csv"
        $episodeCount = (Import-Csv -LiteralPath $manifest).Count
        $expectedCandidates = 16 * $episodeCount
        $actualCandidates = if (Test-Path $candidateCsv) { (Import-Csv -LiteralPath $candidateCsv).Count } else { 0 }
        if ($actualCandidates -eq $expectedCandidates) {
            Write-Host "[nested-cv] fold=$fold split=$split candidates complete ($actualCandidates)"
            continue
        }
        & $PythonExe -B code\benchmarks\run_monuseg_dense_prompt_experiment.py `
            --manifest $manifest `
            --dataset-filter monuseg `
            --out-dir $candidateDir `
            --case-limit 1000 `
            --k-list 40 `
            --lambda-list 0.25,0.75,1.5 `
            --round2-k-list 5,10,20,40,75 `
            --skip-round3 `
            --minimal-artifacts `
            --resume
        if ($LASTEXITCODE -ne 0) { throw "Candidate generation failed for fold=$fold split=$split" }
        $actualCandidates = (Import-Csv -LiteralPath $candidateCsv).Count
        if ($actualCandidates -ne $expectedCandidates) {
            throw "fold=$fold split=$split expected $expectedCandidates candidates, found $actualCandidates"
        }
    }

    $policyDir = Join-Path $foldRun "policy"
    & $PythonExe -B code\benchmarks\train_adaptive_prompt_policy.py `
        --train-candidates (Join-Path $foldRun "candidates_train\candidates.csv") `
        --val-candidates (Join-Path $foldRun "candidates_val\candidates.csv") `
        --test-candidates (Join-Path $foldRun "candidates_test\candidates.csv") `
        --allowed-rounds round2 `
        --allowed-k 5,10,20,40,75 `
        --allowed-lambda 0.25,0.75,1.5 `
        --anchor-uncertainty-weights 0 `
        --out-dir $policyDir
    if ($LASTEXITCODE -ne 0) { throw "Policy training failed for fold=$fold" }

    & $PythonExe -B code\benchmarks\analyze_prompt_policy_results.py `
        --fixed (Join-Path $policyDir "test_fixed_predictions.csv") `
        --learned (Join-Path $policyDir "test_anchored_predictions.csv") `
        --out-dir (Join-Path $policyDir "paired_analysis")
    if ($LASTEXITCODE -ne 0) { throw "Paired analysis failed for fold=$fold" }
}

if ($FoldStart -eq 0 -and $FoldEnd -eq 4) {
    & $PythonExe -B code\benchmarks\aggregate_nested_cv_prompt_policy.py `
        --fold-root $runRoot `
        --out-dir (Join-Path $runRoot "aggregate")
    if ($LASTEXITCODE -ne 0) { throw "Nested-CV aggregation failed" }
}
