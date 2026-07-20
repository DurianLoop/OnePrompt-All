param(
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$negativeSummary = Join-Path $root "runs\research_direction\mechanisms\negative_prompt_ablation_test14\summary.csv"

while (-not (Test-Path -LiteralPath $negativeSummary)) {
    Write-Host "[wait] negative-point ablation has not finished"
    Start-Sleep -Seconds $PollSeconds
}

Set-Location $root
python -B code\benchmarks\run_gbmseg_style_baseline.py `
    --manifest runs\research_direction\prompt_policy_splits\monuseg_strict_official\episodes_test.csv `
    --dino-weights assets\checkpoints\dinov2_vitl14_pretrain.pth `
    --dino-size vit_large `
    --sam-checkpoint assets\checkpoints\sam_vit_b.pth `
    --sam-model-type vit_b `
    --device auto `
    --out-dir runs\research_direction\baselines\gbmseg_style_adapted_test14

if ($LASTEXITCODE -ne 0) {
    throw "GBMSeg-style adapted baseline failed with exit code $LASTEXITCODE"
}
