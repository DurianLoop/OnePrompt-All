$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$env:nnUNet_raw = Join-Path $repo "runs\nnunet\nnUNet_raw"
$env:nnUNet_preprocessed = Join-Path $repo "runs\nnunet\nnUNet_preprocessed"
$env:nnUNet_results = Join-Path $repo "runs\nnunet\nnUNet_results"
$env:nnUNet_def_n_proc = "1"
$env:MPLCONFIGDIR = Join-Path $repo "runs\nnunet\matplotlib_cache"

New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null

& (Join-Path $repo "conda_envs\sam_gpu\Scripts\nnUNetv2_plan_and_preprocess.exe") -d 902 -c 2d -npfp 1 -np 1 --clean
