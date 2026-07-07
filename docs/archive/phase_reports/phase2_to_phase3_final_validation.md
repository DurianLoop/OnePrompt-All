# Phase2 to Phase3 Final Validation

## 1. 验证目标

本轮持续实验要回答一个问题：

```text
“医学图像需要 data-aware prompt selector” 这个方向是否真正可行？
```

这里的“可行”不是指当前规则已经达到最终效果，而是指它满足三个条件：

1. 固定单一 prompt 不是最优策略。
2. 只使用非 GT 信息的 selector 能带来正收益。
3. oracle 上限明显高于当前 selector，说明 Phase3 还有继续研究空间。

实验结论：方向可行。

## 2. 当前实验设置

数据：

- IDRID：20 cases。
- MonuSeg：20 cases。

模型：

- SAM ViT-B。
- MedSAM ViT-B。
- SAM2 Tiny。

Prompt 协议：

- `gt_box`：由 GT mask 得到的 tight box。
- `perturbed_box`：扰动后的 GT box。
- `centroid_point`：GT 前景质心点。
- `grid_point`：GT 前景 2x2 网格点。
- `auto_box`：phase2-lite v0 图像启发式 box。
- `auto_medical_box_v1`：医学启发式候选 box。
- `auto_medical_point_v1`：医学启发式正负点 prompt。

总评估规模：

```text
40 cases x 7 prompt protocols x 3 models = 840 rows
```

其中 MedSAM 不支持 point prompt，因此对应 point rows 记录为 `unsupported`。

## 3. Auto Medical Prompt v1 设计

`auto_medical_prompt_v1` 不使用 GT mask，而是根据数据域生成候选 prompt。

### IDRID

目标是眼底病灶候选。

使用信号：

- red lesion saliency。
- bright lesion saliency。
- fundus/tissue mask。
- robust candidate percentile box。

输出：

- `auto_medical_box_v1`。
- `auto_medical_point_v1`，包含正点和一个背景负点。

### MonuSeg

目标是病理图像中的细胞核候选。

使用信号：

- dark nuclei signal。
- purple stain signal。
- tissue mask。
- dense candidate point sampling。

输出：

- `auto_medical_box_v1`。
- `auto_medical_point_v1`，包含正点和背景负点。

## 4. 单一 Prompt 结果

整体 mean Dice：

| prompt | mean Dice |
| --- | ---: |
| `auto_box` | 0.3118 |
| `auto_medical_box_v1` | 0.2821 |
| `auto_medical_point_v1` | 0.2293 |
| `centroid_point` | 0.2016 |
| `gt_box` | 0.1989 |
| `grid_point` | 0.1856 |
| `perturbed_box` | 0.1718 |

第一层观察：

- `auto_box` 仍然是最强单一 prompt。
- `auto_medical_box_v1` 不是全局替代品。
- `auto_medical_point_v1` 虽然整体不如 `auto_box`，但在 MonuSeg 上有明确优势。

这说明 Phase3 不应该继续寻找“一条规则吃掉所有数据集”的 prompt，而应该做 selector。

## 5. 数据集分解

| dataset | prompt | mean Dice |
| --- | --- | ---: |
| IDRID | `auto_box` | 0.3725 |
| IDRID | `auto_medical_box_v1` | 0.3118 |
| IDRID | `auto_medical_point_v1` | 0.1740 |
| MonuSeg | `auto_box` | 0.2512 |
| MonuSeg | `auto_medical_box_v1` | 0.2524 |
| MonuSeg | `auto_medical_point_v1` | 0.2847 |

关键结论：

- IDRID 当前最适合 lesion candidate box。
- MonuSeg 当前最适合 structured point prompt。
- 医学 prompt 策略存在明显 domain dependency。

## 6. Selector 验证

验证了六类 selector/policy：

| policy | 是否使用 GT | 含义 |
| --- | --- | --- |
| `single_auto_box` | 否 | 所有场景固定用 `auto_box` |
| `single_auto_medical_box_v1` | 否 | 所有场景固定用 `auto_medical_box_v1` |
| `data_aware_selector_v1` | 否 | IDRID 选 `auto_box`；MonuSeg 对 SAM/SAM2 选 `auto_medical_point_v1`，对 MedSAM 选 box |
| `score_selector_v1` | 否 | 在 auto-family prompt 中选择原始 SAM score 最高者 |
| `oracle_auto_family` | 是 | 在 auto-family prompt 中选择 Dice 最好者，只作上限 |
| `oracle_all_prompts` | 是 | 在所有 prompt 中选择 Dice 最好者，只作上限 |

整体结果：

| policy | mean Dice | delta vs `auto_box` |
| --- | ---: | ---: |
| `single_auto_box` | 0.3118 | 0.0000 |
| `data_aware_selector_v1` | 0.3234 | +0.0115 |
| `score_selector_v1` | 0.2846 | -0.0273 |
| `oracle_auto_family` | 0.3584 | +0.0466 |
| `oracle_all_prompts` | 0.3755 | +0.0636 |

这个结果很重要：

1. `data_aware_selector_v1` 在不使用 GT 的情况下超过固定 `auto_box`。
2. `score_selector_v1` 反而变差，说明原始 SAM score 不能直接作为医学 prompt 可靠性评分。
3. `oracle_auto_family` 比当前 selector 高出约 0.035 Dice，说明还有足够空间设计更好的 reliability score。

## 7. 结论是否成立

结论成立，但需要精确定义。

不成立的说法：

```text
我们提出的 auto_medical_prompt_v1 全面优于 auto_box。
```

这个说法不成立，因为 IDRID 上 `auto_box` 仍然更强。

成立的说法：

```text
医学图像 promptable segmentation 需要 data-aware prompt generation 和 prompt selection。
固定单一 prompt 策略无法同时适配 IDRID 和 MonuSeg。
一个简单的 data-aware selector 已经能超过固定 auto_box，而 oracle 上限说明仍有改进空间。
```

这就是 Phase3 的连接点。

## 8. Phase3 方法定义

Phase3 的方法不应该写成“提出一个新的 SAM”，而应写成：

```text
给定医学图像，自动生成多个候选 prompt，估计每个 prompt 的可靠性，
再选择或融合 prompt 调用 SAM-family 模型，以提高医学图像 promptable segmentation 的准确率和稳定性。
```

建议方法名暂定：

```text
OCSAM: Organized Candidate Prompting for Medical Segment Anything Models
```

核心模块：

1. Domain-aware candidate generation。
2. Prompt reliability scoring。
3. Prompt selector。
4. Optional mask refinement。

## 9. 下一步实验

已进一步实现并验证 `prompt_selector_v2` 和 `prompt_selector_v3`。

`prompt_selector_v2` 是一个 leave-one-case-out 的轻量 reliability scorer。它在训练折中使用 Dice 学习 prompt 可靠性，但在 held-out case 上不使用 GT，只根据非 GT 特征选择 prompt。

使用特征：

- dataset one-hot。
- model one-hot。
- prompt protocol one-hot。
- SAM/SAM2 raw score。
- score missing flag。
- predicted mask area ratio。
- small/large mask flags。

验证结果：

| method | mean Dice | delta vs `auto_box` |
| --- | ---: | ---: |
| `single_auto_box` | 0.3118 | 0.0000 |
| `data_aware_selector_v1` | 0.3234 | +0.0115 |
| `prompt_selector_v2` | 0.3263 | +0.0145 |
| `oracle_auto_family` | 0.3584 | +0.0466 |

这说明 learned reliability selector 在 40-case 小基准上已经超过手工 data-aware 规则，但距离 oracle 仍有明显差距。

继续加入论文常见的 prompt-induced uncertainty / mask stability 组件后，得到 `prompt_selector_v3`：

| method | mean Dice | delta vs `auto_box` |
| --- | ---: | ---: |
| `single_auto_box` | 0.3118 | 0.0000 |
| `data_aware_selector_v1` | 0.3234 | +0.0115 |
| `prompt_selector_v2` | 0.3263 | +0.0145 |
| `prompt_selector_v3` | 0.3279 | +0.0160 |
| `oracle_auto_family` | 0.3584 | +0.0466 |

`prompt_selector_v3` 的新增特征包括：

- auto-family masks 之间的 mean IoU。
- auto-family masks 之间的 minimum IoU。
- prompt uncertainty = 1 - mask agreement。
- mask area deviation from median candidate。
- mask area deviation from `auto_box`。

这一步验证了：prompt-induced uncertainty / mask stability 是有效组件，虽然当前 40-case 上提升幅度仍然有限。

因此 Phase3 小里程碑已经完成：

```text
固定自动 prompt -> data-aware selector -> leave-one-case-out reliability selector
-> uncertainty-aware reliability selector
```

都已经实现并验证。

下一步不是继续加大模型，而是增强 reliability signal 和扩大验证集。

建议做 `prompt_selector_v4` 或扩展数据规模。

输入特征：

- dataset/domain。
- candidate area ratio。
- box width/height/aspect ratio。
- positive point count。
- negative point count。
- predicted mask area ratio。
- SAM/SAM2 raw score。
- multi-mask stability。
- prompt-induced uncertainty。
- perturbation-based uncertainty。

输出：

- selected prompt。
- confidence。
- selection reason。

目标：

```text
超过 data_aware_selector_v1，并尽量逼近 oracle_auto_family。
```

当前可量化目标：

| method | mean Dice |
| --- | ---: |
| `single_auto_box` | 0.3118 |
| `data_aware_selector_v1` | 0.3234 |
| `prompt_selector_v2` | 0.3263 |
| `prompt_selector_v3` | 0.3279 |
| `prompt_selector_v4` target | > 0.3279 |
| `oracle_auto_family` upper bound | 0.3584 |

如果 `prompt_selector_v4` 能在 40-case 上稳定超过 0.3279，或 v3 在 100-case 上仍保持正收益，就加入 SAM ViT-H、Matcher、SAM3 作为补充实验。

## 10. 文件位置

代码：

- `code/benchmarks/phase2_lite.py`
- `code/benchmarks/analyze_phase2_lite.py`
- `code/benchmarks/summarize_phase3_connection.py`
- `code/benchmarks/evaluate_prompt_selector.py`
- `code/benchmarks/evaluate_prompt_selector_v2.py`
- `code/benchmarks/evaluate_prompt_selector_v3.py`

结果：

- `runs/phase2_lite/metrics/phase2_lite_metrics.csv`
- `runs/phase2_lite/analysis/summary_by_protocol.csv`
- `runs/phase2_lite/analysis/auto_prompt_v1_comparison.csv`
- `runs/phase2_lite/analysis/prompt_selector_v1_results.csv`
- `runs/phase2_lite/analysis/prompt_selector_v1_summary_by_policy.csv`
- `runs/phase2_lite/analysis/prompt_selector_v2_results.csv`
- `runs/phase2_lite/analysis/prompt_selector_v2_summary_by_policy.csv`
- `runs/phase2_lite/analysis/prompt_selector_v3_results.csv`
- `runs/phase2_lite/analysis/prompt_selector_v3_summary_by_policy.csv`

文档：

- `docs/phase2_medical_prompt_robustness_target.md`
- `docs/phase3_connection_point.md`
- `docs/prompt_selector_feasibility_validation.md`
- `docs/prompt_selector_v2_validation.md`
- `docs/prompt_selector_v3_validation.md`
- `docs/phase2_to_phase3_final_validation.md`
