param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$manifest = Join-Path $root "runs\research_direction\prompt_policy_splits\monuseg_strict_multiref5_eval5\episodes_test.csv"
$testCandidates = Join-Path $root "runs\research_direction\prompt_policy_oracle\test_multiref5_eval5_dinov2l_samvitb\candidates.csv"
$negativeDir = Join-Path $root "runs\research_direction\prompt_policy_oracle\negative_budget_test_multiref5"
$outDir = Join-Path $root "runs\research_direction\prompt_policy\adaptive_negative_budget_v1_eval5"

Set-Location $root
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

& $PythonExe -B code\benchmarks\evaluate_negative_prompt_ablation.py `
    --manifest $manifest `
    --candidates $testCandidates `
    --fixed-action contrast_topk_K40_lam0.75 `
    --negative-counts 0,5,10,20 `
    --sam-device auto `
    --minimal-artifacts `
    --resume `
    --out-dir $negativeDir
if ($LASTEXITCODE -ne 0) { throw "Five-reference negative-budget grid failed" }

$negativeRows = (Import-Csv -LiteralPath (Join-Path $negativeDir "per_case.csv")).Count
if ($negativeRows -ne 280) { throw "Expected 280 five-reference negative rows, found $negativeRows" }

& $PythonExe -B code\benchmarks\train_negative_budget_policy.py `
    --train-negative runs\research_direction\prompt_policy_oracle\negative_budget_train_multiref5\per_case.csv `
    --val-negative runs\research_direction\prompt_policy_oracle\negative_budget_val\per_case.csv `
    --test-negative (Join-Path $negativeDir "per_case.csv") `
    --train-candidates runs\research_direction\prompt_policy_oracle\train_multiref5_reduced_cpu_dinov2l_samvitb\candidates.csv `
    --val-candidates runs\research_direction\prompt_policy_oracle\val_grid_dinov2l_samvitb\candidates.csv `
    --test-candidates $testCandidates `
    --out-dir $outDir
if ($LASTEXITCODE -ne 0) { throw "Five-reference negative-budget policy inference failed" }

& $PythonExe -B code\benchmarks\analyze_prompt_policy_results.py `
    --fixed (Join-Path $outDir "test_fixed_predictions.csv") `
    --learned (Join-Path $outDir "test_learned_predictions.csv") `
    --out-dir (Join-Path $outDir "paired_analysis")
if ($LASTEXITCODE -ne 0) { throw "Five-reference negative-budget paired analysis failed" }

& $PythonExe -B code\benchmarks\analyze_reference_robustness.py `
    --fixed (Join-Path $outDir "test_fixed_predictions.csv") `
    --learned (Join-Path $outDir "test_learned_predictions.csv") `
    --out-dir (Join-Path $outDir "reference_robustness")
if ($LASTEXITCODE -ne 0) { throw "Five-reference negative-budget robustness analysis failed" }

& $PythonExe -B code\benchmarks\analyze_policy_switches.py `
    --fixed (Join-Path $outDir "test_fixed_predictions.csv") `
    --learned (Join-Path $outDir "test_learned_predictions.csv") `
    --out-dir (Join-Path $outDir "switch_audit")
if ($LASTEXITCODE -ne 0) { throw "Five-reference negative-budget switch audit failed" }
