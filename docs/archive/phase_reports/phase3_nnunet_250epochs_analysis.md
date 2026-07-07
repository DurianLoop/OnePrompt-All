# Phase3 nnU-Net 250-epoch 结果分析

## 结论摘要

- 本次实际完成的是 `nnUNetTrainer_250epochs` 的 fold0 训练与 test prediction。
- 训练内部 validation Dice: `0.8057`；held-out test Dice: `0.6507`。
- 当前 held-out test 的 20 个 case 全部来自 `MonuSeg`，因为原始转换脚本按 manifest 顺序做 80/20 split，前 50 个是 IDRID，后 50 个是 MonuSeg。这个结果可以作为 MonuSeg 小测试集参考，但还不能作为跨数据集最终强基线。
- 同 20 个 case 上，promptable SAM 系列最好约 `0.40 Dice`，nnU-Net 达到 `0.6507`，监督式模型明显更强。
- 主要失败来自小目标/小前景 case 的面积过预测，后续 OCSAM-X 可以把 `前景面积校准` 和 `小目标置信抑制` 作为明确改进点。

## 训练概况

| Item | Value |
| --- | --- |
| Trainer | nnUNetTrainer_250epochs |
| Fold | 0 |
| Logged epochs | 250 |
| First epoch pseudo Dice | 0.6583 |
| Last epoch pseudo Dice | 0.868 |
| Best instant pseudo Dice | 0.9025 @ epoch 232 |
| Best EMA pseudo Dice | 0.8787000179290771 @ epoch 243 |
| Internal validation Dice | 0.8056549085638038 |
| Internal validation IoU | 0.678984902535874 |

## Held-out Test 结果

| Split | Cases | Mean Dice | Median Dice | Min Dice | Max Dice | Mean IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MonuSeg test | 20 | 0.650662 | 0.724191 | 0.18077 | 0.809826 | 0.502414 |

## 同 20 Case 对照

| model | best_protocol | mean_dice_same20 | median_dice |
| --- | --- | --- | --- |
| nnU-Net 2D | supervised_250ep_best_ckpt | 0.650662 | 0.724191 |
| sam2_tiny | gt_box | 0.401613 | 0.412277 |
| medsam_vit_b | gt_box | 0.399993 | 0.432681 |
| sam_vit_h | gt_box | 0.398683 | 0.411742 |
| sam_vit_b | gt_box | 0.398472 | 0.411658 |
| medsa_microscope | auto_medical_point_v1 | 0.397811 | 0.428404 |
| medsa_opticdisc | auto_medical_point_v1 | 0.380021 | 0.396153 |

## 失败样本

| case_id | source_case_id | dice | pred_gt_ratio | pred_pixels | gt_pixels |
| --- | --- | --- | --- | --- | --- |
| OCSAM_0096 | MonuSeg_10_4 | 0.18077 | 9.553917 | 10366 | 1085 |
| OCSAM_0092 | MonuSeg_09_6 | 0.244237 | 6.070485 | 2756 | 454 |
| OCSAM_0094 | MonuSeg_10_2 | 0.399178 | 0.634906 | 4158 | 6549 |
| OCSAM_0091 | MonuSeg_09_5 | 0.586881 | 0.71599 | 7684 | 10732 |
| OCSAM_0089 | MonuSeg_09_3 | 0.619807 | 1.138292 | 12610 | 11078 |

## 表现较好样本

| case_id | source_case_id | dice | pred_gt_ratio |
| --- | --- | --- | --- |
| OCSAM_0097 | MonuSeg_11_1 | 0.809826 | 0.981346 |
| OCSAM_0093 | MonuSeg_10_1 | 0.779982 | 1.064635 |
| OCSAM_0080 | MonuSeg_07_1 | 0.774476 | 1.106349 |
| OCSAM_0082 | MonuSeg_07_3 | 0.761924 | 1.133476 |
| OCSAM_0084 | MonuSeg_08_2 | 0.759567 | 1.082334 |

## 对 Phase3 的影响

1. nnU-Net 已经证明：在 MonuSeg 这类监督式细胞核分割上，任务专用训练模型明显强于当前零训练 promptable SAM 系列。
2. 这并不否定 OCSAM-X；它说明我们的论文 claim 不能写成全面超过监督式 SOTA，而应该强调 promptable、少交互、跨模型选择、面积校准和不确定性控制。
3. 当前 split 不合理，必须重做 stratified split 或 dataset-wise protocol：IDRID 和 MonuSeg 都要有 train/val/test，而不是 test 只有 MonuSeg。
4. 下一步最有价值的是做一个 `nnU-Net prediction -> OCSAM failure oracle` 分析：哪些 case SAM 低、nnU-Net 高，说明 prompt/模型选择还能补；哪些 case 两者都低，说明数据或任务定义需要清洗。

## 产物路径

- Test metrics: `runs/nnunet/analysis/Dataset901_OCSAMPhase3_fold0_250epochs/test_metrics.csv`
- Same-20 promptable summary: `runs/nnunet/analysis/Dataset901_OCSAMPhase3_fold0_250epochs/promptable_same20_summary.csv`
- Training curve summary: `runs/nnunet/analysis/Dataset901_OCSAMPhase3_fold0_250epochs/training_curve_summary.json`
- Predictions: `runs/nnunet/predictions/Dataset901_OCSAMPhase3_fold0_250epochs`
