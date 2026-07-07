# Phase 3 Connection Point

## 判断

Phase2-lite 已经达到 Phase3 的连接点：我们不再缺少“能跑的模型”，而是已经有证据说明下一步该研究什么。

核心结论是：

```text
医学 prompt 不能只做一种固定策略，应该做 data-aware prompt selector。
```

也就是说，Phase3 的第一个方法模块不是再换一个更大的 SAM，而是：

```text
Auto Medical Prompt Generator + Prompt Selector
```

## 证据

### Dataset-level delta

| dataset | box_v1 vs auto_box | wins | point_v1 vs centroid | wins | best_v1 vs best_existing | wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IDRID | -0.0607 | 25/60 | -0.0289 | 27/40 | -0.0580 | 24/60 |
| MonuSeg | 0.0012 | 2/60 | 0.0844 | 26/40 | 0.0081 | 12/60 |

### Model x Dataset delta

| model | dataset | box_v1 vs auto_box | point_v1 vs centroid | best_v1 vs best_existing |
| --- | --- | ---: | ---: | ---: |
| medsam_vit_b | IDRID | -0.0342 | nan | -0.0666 |
| medsam_vit_b | MonuSeg | 0.0003 | nan | -0.0342 |
| sam2_tiny | IDRID | -0.1016 | -0.0259 | -0.0967 |
| sam2_tiny | MonuSeg | -0.0000 | 0.0661 | -0.0201 |
| sam_vit_b | IDRID | -0.0463 | -0.0318 | -0.0106 |
| sam_vit_b | MonuSeg | 0.0033 | 0.1027 | 0.0787 |

## Phase3 方法定义

Phase3 不应写成“提出一个新的 SAM”。更稳的定义是：

```text
给定医学图像和任务域，自动生成多个候选 prompt，预测每个 prompt 的可靠性，选择或融合最适合的 prompt，再调用 SAM-family 模型。
```

建议模块拆成三部分：

1. Candidate generation：为 IDRID 生成红色/亮性病灶候选，为 MonuSeg 生成暗紫色细胞核候选。
2. Prompt selector：在 `auto_box`、`auto_medical_box_v1`、`auto_medical_point_v1` 之间选择最可能可靠的 prompt。
3. Optional refinement：只在失败类型明确时加入轻量 adapter 或后处理，不急着训练大模型。

## 下一步实验

1. 实现 `prompt_selector_v1`，先用无需训练的规则：dataset、candidate_area_ratio、box aspect ratio、positive/negative point count、SAM score。
2. 在同一 40-case 上跑 selector，比较 `selected_prompt` 是否超过单一 prompt 均值。
3. 扩到 100-case，再加入 SAM ViT-H / Matcher / SAM3 作为补充模型，而不是改变主线。

## 产物位置

- Comparison table: `runs/phase2_lite/analysis/auto_prompt_v1_comparison.csv`
- Dataset summary: `runs/phase2_lite/analysis/auto_prompt_v1_by_dataset.csv`
- Model x dataset summary: `runs/phase2_lite/analysis/auto_prompt_v1_by_model_dataset.csv`
