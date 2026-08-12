$ErrorActionPreference = 'Stop'
$python = 'D:\SAM\conda_envs\sam_gpu\python.exe'
$runner = 'D:\SAM\surf\research\evaluate_sam3_oracle_and_point.py'
$dataset = 'D:\SAM\surf\datasets\oneprompt_all\benchmark220'
$checkpoint = 'D:\SAM\assets\checkpoints\sam3_modelscope\sam3.pt'
$root = 'D:\SAM\runs\oneprompt_all'

& $python -B $runner --dataset $dataset --checkpoint $checkpoint --out-dir "$root\benchmark220_sam3_text_oracle" --mode text --threshold 0.45 --nms-iou 0.5 --min-area-ratio 0.002 --seed 20260730
& $python -B $runner --dataset $dataset --checkpoint $checkpoint --out-dir "$root\benchmark220_sam3_point_single" --mode point --threshold 0.45 --prompts-per-image 5 --seed 20260730
