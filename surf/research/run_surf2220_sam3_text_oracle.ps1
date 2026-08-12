$ErrorActionPreference = 'Stop'
$python = 'D:\SAM\conda_envs\sam_gpu\python.exe'
& $python -B 'D:\SAM\surf\research\evaluate_sam3_oracle_and_point.py' `
  --dataset 'D:\SAM\surf\datasets\oneprompt_all\surf_v2_unified\oneprompt_multiinstance_2220' `
  --checkpoint 'D:\SAM\assets\checkpoints\sam3_modelscope\sam3.pt' `
  --out-dir 'D:\SAM\runs\oneprompt_all\surf2220_sam3_text_oracle' `
  --mode text --threshold 0.45 --nms-iou 0.5 --min-area-ratio 0.002 --seed 20260730
