param(
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$baselineSummary = Join-Path $root "runs\research_direction\baselines\gbmseg_style_adapted_test14\summary.json"

while (-not (Test-Path -LiteralPath $baselineSummary)) {
    Write-Host "[wait] controlled baselines have not finished"
    Start-Sleep -Seconds $PollSeconds
}

Set-Location $root
$common = @(
    "--case-limit", "0",
    "--k-list", "40",
    "--lambda-list", "0.25,0.75,1.5",
    "--round2-k-list", "5,10,20,40,75",
    "--skip-round3",
    "--minimal-artifacts",
    "--resume"
)

python -B code\benchmarks\run_monuseg_dense_prompt_experiment.py `
    --manifest runs\research_direction\prompt_policy_splits\monuseg_strict_multiref5_eval5\episodes_val.csv `
    --role-filter val `
    --out-dir runs\research_direction\prompt_policy_oracle\val_multiref5_eval5_dinov2l_samvitb `
    @common
if ($LASTEXITCODE -ne 0) {
    throw "Five-reference validation candidates failed with exit code $LASTEXITCODE"
}

python -B code\benchmarks\run_monuseg_dense_prompt_experiment.py `
    --manifest runs\research_direction\prompt_policy_splits\monuseg_strict_multiref5_eval5\episodes_test.csv `
    --role-filter test `
    --out-dir runs\research_direction\prompt_policy_oracle\test_multiref5_eval5_dinov2l_samvitb `
    @common
if ($LASTEXITCODE -ne 0) {
    throw "Five-reference test candidates failed with exit code $LASTEXITCODE"
}
