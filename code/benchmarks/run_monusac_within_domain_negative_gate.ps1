param(
    [string]$GpuPython = "",
    [string]$PolicyPython = ""
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$protocolDir = Join-Path $root "runs\research_direction\prompt_policy_splits\monusac_strict_patient_disjoint_ref1"
$runRoot = Join-Path $root "runs\research_direction\monusac_prompt_policy\within_domain_negative_gate"
$assignments = Join-Path $protocolDir "case_assignments.csv"

if (-not $GpuPython) {
    $workspacePython = Join-Path $root "conda_envs\sam_gpu\python.exe"
    $GpuPython = if (Test-Path -LiteralPath $workspacePython) { $workspacePython } else { "python" }
}
if (-not $PolicyPython) {
    $PolicyPython = if (Test-Path -LiteralPath "D:\conda\python.exe") { "D:\conda\python.exe" } else { $GpuPython }
}

Set-Location $root
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$candidatePaths = @{}
$negativePaths = @{}
foreach ($split in @("train", "val", "test")) {
    $manifest = Join-Path $protocolDir "episodes_$split.csv"
    $episodeCount = (Import-Csv -LiteralPath $manifest).Count
    if ($split -eq "test") {
        $candidateDir = Join-Path $root "runs\research_direction\monusac_prompt_policy\negative_gate_ref1_candidates"
        $negativeDir = Join-Path $root "runs\research_direction\monusac_prompt_policy\negative_gate_ref1_grid_v2"
    } else {
        $candidateDir = Join-Path $runRoot "candidates_$split"
        $negativeDir = Join-Path $runRoot "negative_$split"
    }
    $candidateCsv = Join-Path $candidateDir "candidates.csv"
    $candidatePaths[$split] = $candidateCsv
    $negativePaths[$split] = Join-Path $negativeDir "per_case.csv"

    $expectedCandidates = 2 * $episodeCount
    $actualCandidates = if (Test-Path -LiteralPath $candidateCsv) { (Import-Csv -LiteralPath $candidateCsv).Count } else { 0 }
    if ($actualCandidates -ne $expectedCandidates) {
        & $GpuPython -B code\benchmarks\run_monuseg_dense_prompt_experiment.py `
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
        if ($LASTEXITCODE -ne 0) { throw "MoNuSAC $split candidate generation failed" }
    }
    if ((Import-Csv -LiteralPath $candidateCsv).Count -ne $expectedCandidates) {
        throw "MoNuSAC $split candidate count mismatch"
    }

    & $GpuPython -B code\benchmarks\evaluate_negative_prompt_ablation.py `
        --manifest $manifest `
        --candidates $candidateCsv `
        --fixed-action contrast_topk_K40_lam0.75 `
        --negative-counts 0,5,10,20 `
        --sam-device auto `
        --minimal-artifacts `
        --resume `
        --out-dir $negativeDir
    if ($LASTEXITCODE -ne 0) { throw "MoNuSAC $split negative-budget grid failed" }
    if ((Import-Csv -LiteralPath $negativePaths[$split]).Count -ne 4 * $episodeCount) {
        throw "MoNuSAC $split negative-grid count mismatch"
    }
}

$policyDir = Join-Path $runRoot "policy"
& $PolicyPython -B code\benchmarks\train_negative_budget_policy.py `
    --train-negative $negativePaths["train"] `
    --val-negative $negativePaths["val"] `
    --test-negative $negativePaths["test"] `
    --train-candidates $candidatePaths["train"] `
    --val-candidates $candidatePaths["val"] `
    --test-candidates $candidatePaths["test"] `
    --out-dir $policyDir
if ($LASTEXITCODE -ne 0) { throw "MoNuSAC within-domain policy training failed" }

& $PolicyPython -B code\benchmarks\analyze_prompt_policy_results.py `
    --fixed (Join-Path $policyDir "test_fixed_predictions.csv") `
    --learned (Join-Path $policyDir "test_learned_predictions.csv") `
    --out-dir (Join-Path $policyDir "paired_analysis")
if ($LASTEXITCODE -ne 0) { throw "MoNuSAC within-domain paired analysis failed" }

& $PolicyPython -B code\benchmarks\analyze_monusac_patient_level.py `
    --fixed (Join-Path $policyDir "test_fixed_predictions.csv") `
    --learned (Join-Path $policyDir "test_learned_predictions.csv") `
    --assignments $assignments `
    --out-dir (Join-Path $policyDir "patient_level_analysis")
if ($LASTEXITCODE -ne 0) { throw "MoNuSAC within-domain patient analysis failed" }
