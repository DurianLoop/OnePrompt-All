$ErrorActionPreference = 'Stop'

# Main-table baselines for the exact OnePrompt-All protocol:
# 2,220 images × 5 independent foreground clicks.  The runs are deliberately
# sequential because SAM3 and DINO share one GPU.
$python = 'D:\SAM\conda_envs\sam_gpu\python.exe'
$runner = 'D:\SAM\surf\research\evaluate_sam3_multi_click.py'
$dataset = 'D:\SAM\surf\datasets\oneprompt_all\surf_v2_unified\oneprompt_multiinstance_2220'
$sam3 = 'D:\SAM\assets\checkpoints\sam3_modelscope\sam3.pt'
$dino = 'D:\SAM\assets\checkpoints\dinov2_vitb14_pretrain.pth'
$runRoot = 'D:\SAM\runs\oneprompt_all'

function Invoke-OnePromptBaseline([string] $name, [string[]] $extra) {
    $out = Join-Path $runRoot $name
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    $args = @(
        '-B', $runner,
        '--dataset', $dataset,
        '--checkpoint', $sam3,
        '--dino-checkpoint', $dino,
        '--dino-size', 'vit_base',
        '--prompts-per-image', '5',
        '--seed', '20260730',
        '--threshold', '0.45',
        '--nms-iou', '0.5',
        '--min-area-ratio', '0.002',
        '--negative-points', '0',
        '--negative-policy', 'fixed',
        '--out-dir', $out
    ) + $extra
    & $python @args
    if ($LASTEXITCODE -ne 0) { throw "Baseline '$name' failed with exit code $LASTEXITCODE" }
}

# Baseline 1: SAM3 confidences only; DINO is loaded solely because the shared
# evaluator has a common interface, but its weight is exactly zero.
Invoke-OnePromptBaseline 'main_table_sam3_only_2220_5click' @('--dino-nms-weight', '0')

# Baseline 2: fixed hard DINO veto.  This tests the tempting but recall-harming
# alternative to OnePrompt-All's soft DINO evidence.
Invoke-OnePromptBaseline 'main_table_sam3_dino_hardfilter_022_2220_5click' @(
    '--dino-nms-weight', '0',
    '--candidate-filter-margin', '0.22'
)
