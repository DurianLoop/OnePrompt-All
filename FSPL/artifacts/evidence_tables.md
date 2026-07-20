# Evidence Tables (Auto-generated)

Only completed experiment summaries are rendered. `pending` means no result artifact exists.

## MoNuSAC fixed-rule ablation (seed 20260716; 303 episodes/rule)

| Rule | Mean Dice |
|---|---:|
| Top-1 / 0 negative | 0.0907 |
| Top-3 / 0 negative | 0.3266 |
| Top-5 / 0 negative | 0.3814 |
| Top-5 / 3 negative | 0.4196 |
| Spread radius 3 / 3 negative | 0.4526 |

## Fixed spatial-layout replication

| Combination / reference seed | Top-5 + 3 neg. | Spread r=3 + 3 neg. | Gain |
|---|---:|---:|---:|
| DINOv2-L + SAM-B / 20260716 | 0.4196 | 0.4526 | +0.0330 |
| DINOv2-L + SAM-B / 20260717 | 0.4008 | 0.4362 | +0.0354 |
| DINOv2-B + SAM-B / 20260716 | 0.3835 | 0.4344 | +0.0508 |
| DINOv2-L + SAM-L / 20260716 | 0.4507 | 0.4659 | +0.0152 |
| DINOv2-L + SAM-H / 20260716 | 0.4289 | 0.4439 | +0.0149 |
