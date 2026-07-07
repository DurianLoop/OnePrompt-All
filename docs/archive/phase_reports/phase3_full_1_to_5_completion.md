# Phase3 1-5 完整推进结果

这份报告对应用户要求的 1-5：扩数据、补本地强基线、真实 prompt jitter、统一主表、claim 判断。

## 1. 扩数据到 100-200 cases

- Manifest: `runs/phase3_full/manifests/phase3_100_cases.csv`
- Cases: 100
- Prompts: `runs/phase3_full/prompts/phase3_prompts_with_jitter.jsonl`
- 数据集分布：{'IDRID': 50, 'MonuSeg': 50}

## 2. 强基线接入

| 模型 | Mean Dice | 有效行数 |
| --- | ---: | ---: |
| medsam_vit_b | 0.326 | 1200 |
| sam2_tiny | 0.2968 | 1900 |
| sam_vit_b | 0.2973 | 1900 |
| sam_vit_h | 0.3154 | 1900 |

本轮补上了本地可立即复用且能稳定运行的强基线：`sam_vit_h`，并保留 `sam_vit_b`、`medsam_vit_b`、`sam2_tiny`。`lite_medsam` 在 smoke 中未形成有效主表结果；Matcher 仍应作为 one-shot/support baseline 单独接入，不应伪装成普通 box/point prompt。

## 3. 真实 prompt jitter uncertainty

- 真实 jitter 结果表：`runs/phase3_full/analysis/real_jitter_uncertainty_selection.csv`
- Oracle jitter selector 平均 delta vs auto_box: 0.0655
- Deployable stability selector 平均 delta vs auto_box: -0.0145
- Deployable stability selector 表：`runs/phase3_full/analysis/real_jitter_stability_selector_v1.csv`

## 4. 统一主表

- Metrics: `runs/phase3_full/metrics/phase2_lite_metrics.csv`
- OK rows: 6900 / 7600
- Status counts: {'ok': 6900, 'unsupported': 700}

| 模型 | Prompt | Mean Dice | 有效行数 |
| --- | --- | ---: | ---: |
| medsam_vit_b | auto_box | 0.3714 | 100 |
| medsam_vit_b | auto_medical_box_v1 | 0.3618 | 100 |
| medsam_vit_b | gt_box | 0.2381 | 100 |
| medsam_vit_b | jitter_box_00 | 0.3529 | 100 |
| sam2_tiny | auto_box | 0.3963 | 100 |
| sam2_tiny | auto_medical_box_v1 | 0.3454 | 100 |
| sam2_tiny | gt_box | 0.2184 | 100 |
| sam2_tiny | jitter_box_00 | 0.378 | 100 |
| sam_vit_b | auto_box | 0.4045 | 100 |
| sam_vit_b | auto_medical_box_v1 | 0.3782 | 100 |
| sam_vit_b | gt_box | 0.2211 | 100 |
| sam_vit_b | jitter_box_00 | 0.397 | 100 |
| sam_vit_h | auto_box | 0.4481 | 100 |
| sam_vit_h | auto_medical_box_v1 | 0.3975 | 100 |
| sam_vit_h | gt_box | 0.2307 | 100 |
| sam_vit_h | jitter_box_00 | 0.4229 | 100 |

## 5. Claim 判断

- SAM ViT-B auto_box 平均 Dice: 0.4045
- Cross-model stability selector delta: -0.0062
- Cross-model real-jitter oracle upper bound delta: 0.1217

当前证据说明存在上限空间，但不看 GT 的 selector 仍不够强，应继续补强 reliability predictor 和 P0 医学强基线。

可写的稳健 claim：OCSAM-X 通过结构化自动 prompt、真实 prompt-induced uncertainty 和跨模型互补性分析，构建了一个可复现的医学 SAM prompt adaptation 实验框架；在当前扩展数据上可观察到上限空间，但最终 SOTA 结论仍需 SAM-Med2D / Med-SA / MedSAM2 / nnU-Net 等 P0 baseline 验证。
