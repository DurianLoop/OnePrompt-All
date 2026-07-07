# Prompt Selector Feasibility Validation

## 结论

Phase3 连接点是可行的，但不是以“单个新 prompt 规则碾压所有场景”的方式可行。

验证结果支持一个更稳的方向：

```text
Auto Medical Prompt Generator + Data-aware Prompt Selector
```

一个只根据数据域和模型能力做选择的 `data_aware_selector_v1` 已经超过固定 `auto_box`，说明 selector 路线有实际收益；同时 oracle 上限明显更高，说明还有可研究空间。

## Overall Selector Results

| policy | mean Dice | delta vs auto_box | wins/ties/losses | low Dice < 0.1 | high Dice >= 0.5 |
| --- | ---: | ---: | --- | ---: | ---: |
| data_aware_selector_v1 | 0.3234 | +0.0115 | 19/79/22 | 28 | 29 |
| oracle_all_prompts | 0.3755 | +0.0636 | 69/51/0 | 13 | 34 |
| oracle_auto_family | 0.3584 | +0.0466 | 48/72/0 | 21 | 34 |
| score_selector_v1 | 0.2846 | -0.0273 | 13/90/17 | 39 | 27 |
| single_auto_box | 0.3118 | +0.0000 | 0/120/0 | 31 | 30 |
| single_auto_medical_box_v1 | 0.2821 | -0.0297 | 27/57/36 | 29 | 25 |

## Key Numbers

- Fixed `auto_box`: mean Dice 0.3118.
- `data_aware_selector_v1`: mean Dice 0.3234, delta +0.0115.
- Auto-family oracle: mean Dice 0.3584, delta +0.0466.
- All-prompt oracle: mean Dice 0.3755, delta +0.0636.

## Dataset Breakdown

| policy | dataset | mean Dice | delta vs auto_box | wins/ties/losses |
| --- | --- | ---: | ---: | --- |
| data_aware_selector_v1 | IDRID | 0.3725 | +0.0000 | 0/60/0 |
| data_aware_selector_v1 | MonuSeg | 0.2743 | +0.0231 | 19/19/22 |
| oracle_all_prompts | IDRID | 0.4352 | +0.0627 | 30/30/0 |
| oracle_all_prompts | MonuSeg | 0.3157 | +0.0645 | 39/21/0 |
| oracle_auto_family | IDRID | 0.4278 | +0.0553 | 28/32/0 |
| oracle_auto_family | MonuSeg | 0.2891 | +0.0378 | 20/40/0 |
| score_selector_v1 | IDRID | 0.3158 | -0.0566 | 11/32/17 |
| score_selector_v1 | MonuSeg | 0.2533 | +0.0020 | 2/58/0 |
| single_auto_box | IDRID | 0.3725 | +0.0000 | 0/60/0 |
| single_auto_box | MonuSeg | 0.2512 | +0.0000 | 0/60/0 |
| single_auto_medical_box_v1 | IDRID | 0.3118 | -0.0607 | 25/0/35 |
| single_auto_medical_box_v1 | MonuSeg | 0.2524 | +0.0012 | 2/57/1 |

## Interpretation

- IDRID: 固定 `auto_box` 仍然很强，说明眼底病灶任务当前更需要稳定的 lesion candidate box，而不是多点 prompt。
- MonuSeg: `auto_medical_point_v1` 对 SAM/SAM2 有明显收益，说明密集细胞核任务更适合结构化正负点 prompt。
- MedSAM: 由于不支持 point prompt，selector 必须显式处理模型能力，不能把所有 prompt 强行纳入同一比较。
- `score_selector_v1` 的效果如果低于 data-aware 规则，说明原始 SAM score 不能直接当医学 prompt 可靠性评分，需要学习或设计专门的 reliability score。

## Phase3 Method To Implement

Phase3 的方法应定义为：

```text
给定医学图像，先生成多个候选 prompt，再预测每个 prompt 的可靠性，最后选择或融合 prompt 调用 SAM-family 模型。
```

最小可发表版本可以包含：

1. Domain-aware candidate generation：IDRID lesion box，MonuSeg nuclei point。
2. Prompt reliability scoring：不用 GT，使用 candidate area、box aspect ratio、positive/negative point count、SAM stability score、mask area ratio。
3. Prompt selector：选择 box 或 point prompt，并记录选择理由。
4. Evaluation：和固定 GT box、auto box、centroid/grid point 比较准确率、失败率和交互成本。

## Next Experiment

下一步不是继续增加模型，而是实现 `prompt_selector_v2`：

- 输入：candidate metadata + SAM/SAM2 raw scores + predicted mask area。
- 输出：selected prompt + confidence + reason。
- 目标：逼近 `oracle_auto_family`，同时保持不使用 GT。

如果 `prompt_selector_v2` 能在 40-case 上稳定超过 `data_aware_selector_v1`，再扩到 100-case 并加入 SAM ViT-H / Matcher / SAM3。
