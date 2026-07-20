param(
    [int]$FoldStart = 0,
    [int]$FoldEnd = 4,
    [string]$GpuPython = "D:\SAM\conda_envs\sam_gpu\python.exe",
    [string]$PolicyPython = "D:\conda\python.exe"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$splitRoot = Join-Path $root "runs\research_direction\prompt_policy_splits\monuseg_nested_cv5"
$baseRoot = Join-Path $root "runs\research_direction\nested_cv5_negative_budget"
$runRoot = Join-Path $root "runs\research_direction\nested_cv5_sammed2d_negative_gate_n0n5"

if ($FoldStart -lt 0 -or $FoldEnd -gt 4 -or $FoldStart -gt $FoldEnd) {
    throw "Fold range must satisfy 0 <= FoldStart <= FoldEnd <= 4."
}
if (-not (Test-Path -LiteralPath $GpuPython)) {
    throw "GPU Python not found: $GpuPython"
}

Set-Location $root
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

for ($fold = $FoldStart; $fold -le $FoldEnd; $fold++) {
    $foldRun = Join-Path $runRoot "fold_$fold"
    $baseFold = Join-Path $baseRoot "fold_$fold"
    foreach ($split in @("train", "val", "test")) {
        $manifest = Join-Path $splitRoot "fold_$fold\episodes_$split.csv"
        $candidateCsv = Join-Path $baseFold "base_candidates_$split\candidates.csv"
        $negativeDir = Join-Path $foldRun "negative_$split"
        $episodeCount = (Import-Csv -LiteralPath $manifest).Count

        & $GpuPython -B code\benchmarks\evaluate_negative_prompt_ablation.py `
            --manifest $manifest `
            --candidates $candidateCsv `
            --fixed-action contrast_topk_K40_lam0.75 `
            --negative-counts 0,5 `
            --sam-backend sammed2d `
            --sam-checkpoint assets\checkpoints\sam_med2d_b.pth `
            --sam-device cuda `
            --minimal-artifacts `
            --resume `
            --out-dir $negativeDir
        if ($LASTEXITCODE -ne 0) { throw "SAM-Med2D negative grid failed for fold=$fold split=$split" }
        if ((Import-Csv -LiteralPath (Join-Path $negativeDir "per_case.csv")).Count -ne 2 * $episodeCount) {
            throw "SAM-Med2D negative-grid count mismatch for fold=$fold split=$split"
        }
    }

    $policyDir = Join-Path $foldRun "policy"
    & $PolicyPython -B code\benchmarks\train_negative_budget_policy.py `
        --train-negative (Join-Path $foldRun "negative_train\per_case.csv") `
        --val-negative (Join-Path $foldRun "negative_val\per_case.csv") `
        --test-negative (Join-Path $foldRun "negative_test\per_case.csv") `
        --train-candidates (Join-Path $baseFold "base_candidates_train\candidates.csv") `
        --val-candidates (Join-Path $baseFold "base_candidates_val\candidates.csv") `
        --test-candidates (Join-Path $baseFold "base_candidates_test\candidates.csv") `
        --out-dir $policyDir
    if ($LASTEXITCODE -ne 0) { throw "SAM-Med2D policy failed for fold=$fold" }

    & $PolicyPython -B code\benchmarks\analyze_prompt_policy_results.py `
        --fixed (Join-Path $policyDir "test_fixed_predictions.csv") `
        --learned (Join-Path $policyDir "test_learned_predictions.csv") `
        --out-dir (Join-Path $policyDir "paired_analysis")
    if ($LASTEXITCODE -ne 0) { throw "SAM-Med2D paired analysis failed for fold=$fold" }
}

if ($FoldStart -eq 0 -and $FoldEnd -eq 4) {
    & $PolicyPython -B code\benchmarks\aggregate_nested_cv_prompt_policy.py `
        --fold-root $runRoot `
        --learned-filename test_learned_predictions.csv `
        --out-dir (Join-Path $runRoot "aggregate")
    if ($LASTEXITCODE -ne 0) { throw "SAM-Med2D nested aggregation failed" }
}
