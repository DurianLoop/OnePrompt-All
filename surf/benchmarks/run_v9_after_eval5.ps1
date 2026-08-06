param(
    [int]$PollSeconds = 30
)

Write-Warning "Deprecated full-grid V9 entrypoint; forwarding to the budget-faithful top-3 shortlist experiment."
& (Join-Path $PSScriptRoot "run_v9_shortlist_consistency.ps1")
exit $LASTEXITCODE

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$trainCandidates = Join-Path $root "runs\research_direction\prompt_policy_oracle\train_multiref5_reduced_cpu_dinov2l_samvitb\candidates.csv"
$valCandidates = Join-Path $root "runs\research_direction\prompt_policy_oracle\val_multiref5_eval5_dinov2l_samvitb\candidates.csv"
$testCandidates = Join-Path $root "runs\research_direction\prompt_policy_oracle\test_multiref5_eval5_dinov2l_samvitb\candidates.csv"
$negativePolicySummary = Join-Path $root "runs\research_direction\prompt_policy\adaptive_negative_budget_v1\paired_analysis\paired_summary.json"

while (-not (Test-Path $valCandidates) -or -not (Test-Path $testCandidates) -or -not (Test-Path $negativePolicySummary)) {
    Write-Host "[wait] v9 is waiting for eval5 and the focused negative-budget policy"
    Start-Sleep -Seconds $PollSeconds
}

if ((Import-Csv $trainCandidates).Count -ne 2320) { throw "Incomplete train candidates" }
if ((Import-Csv $valCandidates).Count -ne 640) { throw "Incomplete five-reference val candidates" }
if ((Import-Csv $testCandidates).Count -ne 1120) { throw "Incomplete five-reference test candidates" }

Set-Location $root
python -B code\benchmarks\materialize_candidate_masks.py `
    --candidates $trainCandidates `
    --episodes runs\research_direction\prompt_policy_splits\monuseg_strict_multiref5\episodes_train.csv `
    --summary runs\research_direction\prompt_policy_oracle\train_multiref5_reduced_cpu_dinov2l_samvitb\mask_materialization.json
if ($LASTEXITCODE -ne 0) { throw "Train mask materialization failed" }

python -B code\benchmarks\materialize_candidate_masks.py `
    --candidates $valCandidates `
    --episodes runs\research_direction\prompt_policy_splits\monuseg_strict_multiref5_eval5\episodes_val.csv `
    --summary runs\research_direction\prompt_policy_oracle\val_multiref5_eval5_dinov2l_samvitb\mask_materialization.json
if ($LASTEXITCODE -ne 0) { throw "Validation mask materialization failed" }

python -B code\benchmarks\materialize_candidate_masks.py `
    --candidates $testCandidates `
    --episodes runs\research_direction\prompt_policy_splits\monuseg_strict_multiref5_eval5\episodes_test.csv `
    --summary runs\research_direction\prompt_policy_oracle\test_multiref5_eval5_dinov2l_samvitb\mask_materialization.json
if ($LASTEXITCODE -ne 0) { throw "Test mask materialization failed" }

powershell -NoProfile -ExecutionPolicy Bypass -File code\benchmarks\run_prompt_policy_v9_consistency.ps1
if ($LASTEXITCODE -ne 0) { throw "V9 training failed" }

$policyDir = "runs\research_direction\prompt_policy\adaptive_policy_models_v9_consistency_multiref5"
$evalRoot = "runs\research_direction\prompt_policy\adaptive_policy_models_v9_consistency_multiref5_eval5"
python -B code\benchmarks\apply_saved_prompt_policy.py --candidates $valCandidates --model-dir $policyDir --out-dir "$evalRoot\val"
if ($LASTEXITCODE -ne 0) { throw "Frozen V9 validation inference failed" }
python -B code\benchmarks\apply_saved_prompt_policy.py --candidates $testCandidates --model-dir $policyDir --out-dir "$evalRoot\test"
if ($LASTEXITCODE -ne 0) { throw "Frozen V9 test inference failed" }

python -B code\benchmarks\analyze_prompt_policy_results.py --fixed "$evalRoot\val\fixed_predictions.csv" --learned "$evalRoot\val\anchored_predictions.csv" --out-dir "$evalRoot\val\paired_analysis"
if ($LASTEXITCODE -ne 0) { throw "V9 validation paired analysis failed" }
python -B code\benchmarks\analyze_prompt_policy_results.py --fixed "$evalRoot\test\fixed_predictions.csv" --learned "$evalRoot\test\anchored_predictions.csv" --out-dir "$evalRoot\test\paired_analysis"
if ($LASTEXITCODE -ne 0) { throw "V9 test paired analysis failed" }
