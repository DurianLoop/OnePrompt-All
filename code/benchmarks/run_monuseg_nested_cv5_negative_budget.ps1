param(
    [int]$FoldStart = 0,
    [int]$FoldEnd = 4,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$splitRoot = Join-Path $root "runs\research_direction\prompt_policy_splits\monuseg_nested_cv5"
$runRoot = Join-Path $root "runs\research_direction\nested_cv5_negative_budget"

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
        $candidateDir = Join-Path $foldRun "base_candidates_$split"
        $candidateCsv = Join-Path $candidateDir "candidates.csv"
        $negativeDir = Join-Path $foldRun "negative_$split"
        $episodeCount = (Import-Csv -LiteralPath $manifest).Count

        $candidateCount = if (Test-Path $candidateCsv) { (Import-Csv -LiteralPath $candidateCsv).Count } else { 0 }
        if ($candidateCount -ne 2 * $episodeCount) {
            & $PythonExe -B code\benchmarks\run_monuseg_dense_prompt_experiment.py `
                --manifest $manifest `
                --dataset-filter monuseg `
                --out-dir $candidateDir `
                --case-limit 1000 `
                --k-list 40 `
                --lambda-list 0.75 `
                --round2-k-list 40 `
                --skip-round3 `
                --minimal-artifacts `
                --resume
            if ($LASTEXITCODE -ne 0) { throw "Base candidates failed for fold=$fold split=$split" }
        }
        $candidateCount = (Import-Csv -LiteralPath $candidateCsv).Count
        if ($candidateCount -ne 2 * $episodeCount) {
            throw "fold=$fold split=$split expected $($episodeCount * 2) base candidates, found $candidateCount"
        }

        & $PythonExe -B code\benchmarks\evaluate_negative_prompt_ablation.py `
            --manifest $manifest `
            --candidates $candidateCsv `
            --fixed-action contrast_topk_K40_lam0.75 `
            --negative-counts 0,5,10,20 `
            --sam-device auto `
            --minimal-artifacts `
            --resume `
            --out-dir $negativeDir
        if ($LASTEXITCODE -ne 0) { throw "Negative grid failed for fold=$fold split=$split" }
        $negativeCount = (Import-Csv -LiteralPath (Join-Path $negativeDir "per_case.csv")).Count
        if ($negativeCount -ne 4 * $episodeCount) {
            throw "fold=$fold split=$split expected $($episodeCount * 4) negative rows, found $negativeCount"
        }
    }

    $policyDir = Join-Path $foldRun "policy"
    & $PythonExe -B code\benchmarks\train_negative_budget_policy.py `
        --train-negative (Join-Path $foldRun "negative_train\per_case.csv") `
        --val-negative (Join-Path $foldRun "negative_val\per_case.csv") `
        --test-negative (Join-Path $foldRun "negative_test\per_case.csv") `
        --train-candidates (Join-Path $foldRun "base_candidates_train\candidates.csv") `
        --val-candidates (Join-Path $foldRun "base_candidates_val\candidates.csv") `
        --test-candidates (Join-Path $foldRun "base_candidates_test\candidates.csv") `
        --out-dir $policyDir
    if ($LASTEXITCODE -ne 0) { throw "Negative-budget policy failed for fold=$fold" }

    & $PythonExe -B code\benchmarks\analyze_prompt_policy_results.py `
        --fixed (Join-Path $policyDir "test_fixed_predictions.csv") `
        --learned (Join-Path $policyDir "test_learned_predictions.csv") `
        --out-dir (Join-Path $policyDir "paired_analysis")
    if ($LASTEXITCODE -ne 0) { throw "Negative-budget paired analysis failed for fold=$fold" }
}

if ($FoldStart -eq 0 -and $FoldEnd -eq 4) {
    & $PythonExe -B code\benchmarks\aggregate_nested_cv_prompt_policy.py `
        --fold-root $runRoot `
        --learned-filename test_learned_predictions.csv `
        --out-dir (Join-Path $runRoot "aggregate")
    if ($LASTEXITCODE -ne 0) { throw "Nested negative-budget aggregation failed" }
}
