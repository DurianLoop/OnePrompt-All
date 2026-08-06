param(
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$trainCandidates = Join-Path $root "runs\research_direction\prompt_policy_oracle\train_multiref5_reduced_cpu_dinov2l_samvitb\candidates.csv"

while (-not (Test-Path -LiteralPath $trainCandidates)) {
    Write-Host "[wait] dense candidate generation is still using the GPU"
    Start-Sleep -Seconds $PollSeconds
}

Set-Location $root
python -B code\benchmarks\evaluate_negative_prompt_ablation.py `
    --manifest runs\research_direction\prompt_policy_splits\monuseg_strict_official\episodes_test.csv `
    --candidates runs\research_direction\prompt_policy_oracle\test_grid_dinov2l_samvitb_fixedr3\candidates.csv `
    --fixed-action contrast_topk_K40_lam0.75 `
    --negative-counts 0,5,10,20 `
    --sam-checkpoint assets\checkpoints\sam_vit_b.pth `
    --sam-model-type vit_b `
    --sam-device auto `
    --out-dir runs\research_direction\mechanisms\negative_prompt_ablation_test14

if ($LASTEXITCODE -ne 0) {
    throw "Negative-point ablation failed with exit code $LASTEXITCODE"
}

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
