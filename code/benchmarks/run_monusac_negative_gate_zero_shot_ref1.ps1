param(
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$manifest = Join-Path $root "runs\research_direction\prompt_policy_splits\monusac_strict_patient_disjoint_ref1\episodes_test.csv"
$candidateDir = Join-Path $root "runs\research_direction\monusac_prompt_policy\negative_gate_ref1_candidates"
$candidateCsv = Join-Path $candidateDir "candidates.csv"
$negativeDir = Join-Path $root "runs\research_direction\monusac_prompt_policy\negative_gate_ref1_grid_v2"
$modelDir = Join-Path $root "runs\research_direction\prompt_policy\adaptive_negative_budget_v1"
$outDir = Join-Path $root "runs\research_direction\monusac_prompt_policy\negative_gate_ref1_frozen_v2"
$assignments = Join-Path $root "runs\research_direction\prompt_policy_splits\monusac_strict_patient_disjoint_ref1\case_assignments.csv"

if (-not $PythonExe) {
    $gpuPython = Join-Path $root "conda_envs\sam_gpu\python.exe"
    $PythonExe = if (Test-Path -LiteralPath $gpuPython) { $gpuPython } else { "python" }
}
$policyPython = if (Test-Path -LiteralPath "D:\conda\python.exe") { "D:\conda\python.exe" } else { $PythonExe }

Set-Location $root
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$episodeCount = (Import-Csv -LiteralPath $manifest).Count
$expectedCandidates = 2 * $episodeCount
$actualCandidates = if (Test-Path $candidateCsv) { (Import-Csv -LiteralPath $candidateCsv).Count } else { 0 }
if ($actualCandidates -ne $expectedCandidates) {
    & $PythonExe -B code\benchmarks\run_monuseg_dense_prompt_experiment.py `
        --manifest $manifest `
        --dataset-filter monusac `
        --out-dir $candidateDir `
        --case-limit 1000 `
        --k-list 40 `
        --lambda-list 0.75 `
        --round2-k-list 40 `
        --skip-round3 `
        --minimal-artifacts `
        --resume
    if ($LASTEXITCODE -ne 0) { throw "MoNuSAC negative-gate candidate generation failed" }
}
$actualCandidates = (Import-Csv -LiteralPath $candidateCsv).Count
if ($actualCandidates -ne $expectedCandidates) {
    throw "Expected $expectedCandidates MoNuSAC candidates, found $actualCandidates"
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
if ($LASTEXITCODE -ne 0) { throw "MoNuSAC negative-budget grid failed" }
$negativeRows = (Import-Csv -LiteralPath (Join-Path $negativeDir "per_case.csv")).Count
if ($negativeRows -ne 4 * $episodeCount) {
    throw "Expected $($episodeCount * 4) MoNuSAC negative rows, found $negativeRows"
}

& $policyPython -B code\benchmarks\apply_saved_negative_budget_policy.py `
    --negative (Join-Path $negativeDir "per_case.csv") `
    --candidates $candidateCsv `
    --model-dir $modelDir `
    --out-dir $outDir
if ($LASTEXITCODE -ne 0) { throw "Frozen MoNuSeg negative gate on MoNuSAC failed" }

& $policyPython -B code\benchmarks\analyze_prompt_policy_results.py `
    --fixed (Join-Path $outDir "fixed_predictions.csv") `
    --learned (Join-Path $outDir "learned_predictions.csv") `
    --out-dir (Join-Path $outDir "paired_analysis")
if ($LASTEXITCODE -ne 0) { throw "MoNuSAC negative-gate paired analysis failed" }

& $policyPython -B code\benchmarks\analyze_monusac_patient_level.py `
    --fixed (Join-Path $outDir "fixed_predictions.csv") `
    --learned (Join-Path $outDir "learned_predictions.csv") `
    --assignments $assignments `
    --out-dir (Join-Path $outDir "patient_level_analysis")
if ($LASTEXITCODE -ne 0) { throw "MoNuSAC negative-gate patient analysis failed" }
