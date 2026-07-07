$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$env:nnUNet_raw = Join-Path $repo "runs\nnunet\nnUNet_raw"
$env:nnUNet_preprocessed = Join-Path $repo "runs\nnunet\nnUNet_preprocessed"
$env:nnUNet_results = Join-Path $repo "runs\nnunet\nnUNet_results"
$env:nnUNet_n_proc_DA = "0"
$env:nnUNet_def_n_proc = "1"
$env:MPLCONFIGDIR = Join-Path $repo "runs\nnunet\matplotlib_cache"

New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null

& (Join-Path $repo "conda_envs\sam_gpu\Scripts\nnUNetv2_train.exe") 902 2d 0 -tr nnUNetTrainer_250epochs -device cuda
