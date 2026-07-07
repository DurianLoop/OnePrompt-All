$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot

& (Join-Path $repo "conda_envs\sam_gpu\python.exe") `
  (Join-Path $repo "code\benchmarks\prepare_nnunet_phase3_dataset.py") `
  --out (Join-Path $repo "runs\nnunet\nnUNet_raw\Dataset902_OCSAMPhase3Stratified") `
  --dataset-name OCSAMPhase3Stratified `
  --split-mode stratified `
  --seed 2026 `
  --train-ratio 0.8
