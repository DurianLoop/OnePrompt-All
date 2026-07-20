param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$trainShortlist = Join-Path $root "runs\research_direction\prompt_policy\adaptive_policy_models_v7_multiref5\train_anchored_shortlist.csv"
$valShortlist = Join-Path $root "runs\research_direction\prompt_policy\adaptive_policy_models_v7_multiref5_eval5\val\anchored_shortlist.csv"
$testShortlist = Join-Path $root "runs\research_direction\prompt_policy\adaptive_policy_models_v7_multiref5_eval5\test\anchored_shortlist.csv"
$outDir = Join-Path $root "runs\research_direction\prompt_policy\adaptive_policy_models_v9_shortlist_consistency"

$expected = @{
    $trainShortlist = 435
    $valShortlist = 120
    $testShortlist = 210
}
foreach ($path in $expected.Keys) {
    $count = (Import-Csv -LiteralPath $path).Count
    if ($count -ne $expected[$path]) {
        throw "Expected $($expected[$path]) rows in $path, found $count"
    }
}

Set-Location $root
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

& $PythonExe -B code\benchmarks\materialize_candidate_masks.py `
    --candidates $trainShortlist `
    --episodes runs\research_direction\prompt_policy_splits\monuseg_strict_multiref5\episodes_train.csv `
    --summary runs\research_direction\prompt_policy\adaptive_policy_models_v7_multiref5\train_shortlist_mask_materialization.json
if ($LASTEXITCODE -ne 0) { throw "Train shortlist mask materialization failed" }

& $PythonExe -B code\benchmarks\materialize_candidate_masks.py `
    --candidates $valShortlist `
    --episodes runs\research_direction\prompt_policy_splits\monuseg_strict_multiref5_eval5\episodes_val.csv `
    --summary runs\research_direction\prompt_policy\adaptive_policy_models_v7_multiref5_eval5\val\shortlist_mask_materialization.json
if ($LASTEXITCODE -ne 0) { throw "Validation shortlist mask materialization failed" }

& $PythonExe -B code\benchmarks\materialize_candidate_masks.py `
    --candidates $testShortlist `
    --episodes runs\research_direction\prompt_policy_splits\monuseg_strict_multiref5_eval5\episodes_test.csv `
    --summary runs\research_direction\prompt_policy\adaptive_policy_models_v7_multiref5_eval5\test\shortlist_mask_materialization.json
if ($LASTEXITCODE -ne 0) { throw "Test shortlist mask materialization failed" }

& $PythonExe -B code\benchmarks\train_shortlist_consistency_selector.py `
    --train-shortlist $trainShortlist `
    --val-shortlist $valShortlist `
    --test-shortlist $testShortlist `
    --out-dir $outDir
if ($LASTEXITCODE -ne 0) { throw "Shortlist consistency selector training failed" }

foreach ($split in @("val", "test")) {
    $fixed = Join-Path $outDir "${split}_fixed_predictions.csv"
    $base = Join-Path $outDir "${split}_base_post_sam_predictions.csv"
    $consistency = Join-Path $outDir "${split}_plus_matcher_mask_consistency_predictions.csv"

    & $PythonExe -B code\benchmarks\analyze_prompt_policy_results.py `
        --fixed $fixed --learned $base --out-dir (Join-Path $outDir "${split}_paired_base_vs_fixed")
    if ($LASTEXITCODE -ne 0) { throw "$split base-vs-fixed analysis failed" }

    & $PythonExe -B code\benchmarks\analyze_prompt_policy_results.py `
        --fixed $fixed --learned $consistency --out-dir (Join-Path $outDir "${split}_paired_consistency_vs_fixed")
    if ($LASTEXITCODE -ne 0) { throw "$split consistency-vs-fixed analysis failed" }

    & $PythonExe -B code\benchmarks\analyze_prompt_policy_results.py `
        --fixed $base --learned $consistency --out-dir (Join-Path $outDir "${split}_paired_consistency_vs_base")
    if ($LASTEXITCODE -ne 0) { throw "$split consistency ablation analysis failed" }

    & $PythonExe -B code\benchmarks\analyze_reference_robustness.py `
        --fixed $fixed --learned $consistency --out-dir (Join-Path $outDir "${split}_reference_robustness")
    if ($LASTEXITCODE -ne 0) { throw "$split reference robustness analysis failed" }

    & $PythonExe -B code\benchmarks\analyze_policy_switches.py `
        --fixed $fixed --learned $consistency --out-dir (Join-Path $outDir "${split}_switch_audit")
    if ($LASTEXITCODE -ne 0) { throw "$split switch audit failed" }
}

& $PythonExe -B code\benchmarks\visualize_prompt_policy_decisions.py `
    --fixed (Join-Path $outDir "test_fixed_predictions.csv") `
    --learned (Join-Path $outDir "test_plus_matcher_mask_consistency_predictions.csv") `
    --episodes runs\research_direction\prompt_policy_splits\monuseg_strict_multiref5_eval5\episodes_test.csv `
    --out-dir (Join-Path $outDir "test_qualitative") `
    --max-cases 10
if ($LASTEXITCODE -ne 0) { throw "V9 qualitative rendering failed" }
