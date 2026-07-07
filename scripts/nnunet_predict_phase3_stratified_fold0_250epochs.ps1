$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$env:nnUNet_raw = Join-Path $repo "runs\nnunet\nnUNet_raw"
$env:nnUNet_preprocessed = Join-Path $repo "runs\nnunet\nnUNet_preprocessed"
$env:nnUNet_results = Join-Path $repo "runs\nnunet\nnUNet_results"
$env:nnUNet_def_n_proc = "1"
$env:MPLCONFIGDIR = Join-Path $repo "runs\nnunet\matplotlib_cache"

$inputDir = Join-Path $env:nnUNet_raw "Dataset902_OCSAMPhase3Stratified\imagesTs"
$outputDir = Join-Path $repo "runs\nnunet\predictions\Dataset902_OCSAMPhase3Stratified_fold0_250epochs"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null

& (Join-Path $repo "conda_envs\sam_gpu\Scripts\nnUNetv2_predict.exe") -i $inputDir -o $outputDir -d 902 -c 2d -f 0 -tr nnUNetTrainer_250epochs -chk checkpoint_best.pth -device cuda -npp 1 -nps 1
