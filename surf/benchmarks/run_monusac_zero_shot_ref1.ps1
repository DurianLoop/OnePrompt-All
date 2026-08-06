param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$manifest = Join-Path $root "runs\research_direction\prompt_policy_splits\monusac_strict_patient_disjoint_ref1\episodes_test.csv"
$candidateDir = Join-Path $root "runs\research_direction\monusac_prompt_policy\zero_shot_ref1_candidates"
$candidateCsv = Join-Path $candidateDir "candidates.csv"
$policyDir = Join-Path $root "runs\research_direction\prompt_policy\adaptive_policy_models_v7_multiref5"
$outDir = Join-Path $root "runs\research_direction\monusac_prompt_policy\zero_shot_ref1_v7"

Set-Location $root
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$episodeCount = (Import-Csv -LiteralPath $manifest).Count
$expectedCandidates = 16 * $episodeCount
$actualCandidates = if (Test-Path $candidateCsv) { (Import-Csv -LiteralPath $candidateCsv).Count } else { 0 }
if ($actualCandidates -ne $expectedCandidates) {
    & $PythonExe -B code\benchmarks\run_monuseg_dense_prompt_experiment.py `
        --manifest $manifest `
        --dataset-filter monusac `
        --out-dir $candidateDir `
        --case-limit 1000 `
        --k-list 40 `
        --lambda-list 0.25,0.75,1.5 `
        --round2-k-list 5,10,20,40,75 `
        --skip-round3 `
        --minimal-artifacts `
        --resume
    if ($LASTEXITCODE -ne 0) { throw "MoNuSAC zero-shot candidate generation failed" }
}
$actualCandidates = (Import-Csv -LiteralPath $candidateCsv).Count
if ($actualCandidates -ne $expectedCandidates) {
    throw "Expected $expectedCandidates MoNuSAC candidates, found $actualCandidates"
}

& $PythonExe -B code\benchmarks\apply_saved_prompt_policy.py `
    --candidates $candidateCsv `
    --model-dir $policyDir `
    --out-dir $outDir
if ($LASTEXITCODE -ne 0) { throw "Frozen MoNuSeg V7 inference on MoNuSAC failed" }

& $PythonExe -B code\benchmarks\analyze_prompt_policy_results.py `
    --fixed (Join-Path $outDir "fixed_predictions.csv") `
    --learned (Join-Path $outDir "anchored_predictions.csv") `
    --out-dir (Join-Path $outDir "paired_analysis")
if ($LASTEXITCODE -ne 0) { throw "MoNuSAC zero-shot paired analysis failed" }

& $PythonExe -B code\benchmarks\analyze_monusac_patient_level.py `
    --fixed (Join-Path $outDir "fixed_predictions.csv") `
    --learned (Join-Path $outDir "anchored_predictions.csv") `
    --assignments runs\research_direction\prompt_policy_splits\monusac_strict_patient_disjoint_ref1\case_assignments.csv `
    --out-dir (Join-Path $outDir "patient_level_analysis")
if ($LASTEXITCODE -ne 0) { throw "MoNuSAC patient-level analysis failed" }
