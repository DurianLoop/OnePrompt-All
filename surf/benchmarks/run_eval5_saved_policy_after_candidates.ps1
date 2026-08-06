param(
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$policyDir = Join-Path $root "runs\research_direction\prompt_policy\adaptive_policy_models_v7_multiref5"
$valCandidates = Join-Path $root "runs\research_direction\prompt_policy_oracle\val_multiref5_eval5_dinov2l_samvitb\candidates.csv"
$testCandidates = Join-Path $root "runs\research_direction\prompt_policy_oracle\test_multiref5_eval5_dinov2l_samvitb\candidates.csv"

while (-not (Test-Path (Join-Path $policyDir "model.json")) -or -not (Test-Path $valCandidates) -or -not (Test-Path $testCandidates)) {
    Write-Host "[wait] v7 model or five-reference evaluation candidates are incomplete"
    Start-Sleep -Seconds $PollSeconds
}

Set-Location $root
$evalRoot = "runs\research_direction\prompt_policy\adaptive_policy_models_v7_multiref5_eval5"

python -B code\benchmarks\apply_saved_prompt_policy.py `
    --candidates $valCandidates `
    --model-dir $policyDir `
    --out-dir "$evalRoot\val"
if ($LASTEXITCODE -ne 0) { throw "Frozen v7 validation inference failed" }

python -B code\benchmarks\apply_saved_prompt_policy.py `
    --candidates $testCandidates `
    --model-dir $policyDir `
    --out-dir "$evalRoot\test"
if ($LASTEXITCODE -ne 0) { throw "Frozen v7 test inference failed" }

python -B code\benchmarks\analyze_prompt_policy_results.py `
    --fixed "$evalRoot\val\fixed_predictions.csv" `
    --learned "$evalRoot\val\anchored_predictions.csv" `
    --out-dir "$evalRoot\val\paired_analysis"
if ($LASTEXITCODE -ne 0) { throw "Five-reference validation paired analysis failed" }

python -B code\benchmarks\analyze_prompt_policy_results.py `
    --fixed "$evalRoot\test\fixed_predictions.csv" `
    --learned "$evalRoot\test\anchored_predictions.csv" `
    --out-dir "$evalRoot\test\paired_analysis"
if ($LASTEXITCODE -ne 0) { throw "Five-reference test paired analysis failed" }

python -B code\benchmarks\analyze_reference_robustness.py `
    --fixed "$evalRoot\val\fixed_predictions.csv" `
    --learned "$evalRoot\val\anchored_predictions.csv" `
    --out-dir "$evalRoot\val\reference_robustness"
if ($LASTEXITCODE -ne 0) { throw "Five-reference validation robustness analysis failed" }

python -B code\benchmarks\analyze_reference_robustness.py `
    --fixed "$evalRoot\test\fixed_predictions.csv" `
    --learned "$evalRoot\test\anchored_predictions.csv" `
    --out-dir "$evalRoot\test\reference_robustness"
if ($LASTEXITCODE -ne 0) { throw "Five-reference test robustness analysis failed" }
