# 论文就绪判定：Response-Gated Reference Prompting

更新：2026-07-16

## 1. 判定

**已经达到“可以正式写论文”的条件，但尚未达到“可以立即投稿”的条件。**

已具备：明确问题、收缩后的主方法、严格 nested-CV 主结果、机制消融、oracle gap、患者隔离外部验证、跨域失败边界、成功/失败定性图、完整可复现脚本和 112 项通过的单元测试。

投稿前的主要硬缺口：

1. 最接近工作 FoB 尚未完成 MoNuSeg 同协议适配；
2. 完整 DINO matcher + segmenter 的端到端部署计时尚未形成；segmenter-only 成本表已完成。

## 2. 可以写进摘要的主结果

| 命题 | 严格证据 | 判定 |
| --- | --- | --- |
| reference background 改善 matching | MoNuSeg nested-CV：Dice 0.473325 -> 0.538050，delta=+0.064725，CI=[0.046994,0.084141]，36/0/1 | 成立 |
| learned gate 优于每折 validation anchor | 37 query：0.537458 -> 0.547555，delta=+0.010097，CI=[0.003870,0.018777]，15/21/1，p=0.000519 | 成立 |
| learned gate 优于全局固定 N5 | 0.538372 -> 0.547555，delta=+0.009183，CI=[0.000437,0.019333]，14/21/2，p=0.004181 | 成立 |
| gate 缩小 oracle gap | oracle=0.552270；fixed-to-oracle gap=0.013898；learned gap=0.004715；关闭 66.1% | 成立 |
| 零样本 gate 能在 MoNuSAC 纠正 N0 | patch delta=+0.007904，CI=[0.001635,0.015518]；patient delta=+0.007617，CI=[0.001808,0.014621] | 成立 |
| MoNuSAC 域内 gate 能纠正 N0 | patch delta=+0.005951，CI 跨 0；patient delta=+0.005559，CI=[0.000625,0.011330] | 患者级成立，patch 级趋势 |
| gate 在外部域优于任意固定预算 | MoNuSAC global fixed N5=0.544167，高于 zero-shot=0.536154 和 domain=0.534201 | 不成立 |
| shared policy 跨 SAM 版本普适 | fixed fg-bg 对 ViT-B/L/H/SAM2.1 的增益差异大，后两者非正 | 不成立 |
| learned gate 跨第二 segmenter 保持收益 | SAM-Med2D nested-CV：delta=-0.000330，CI=[-0.000872,0.000024] | 不成立 |

## 3. 主论文定位

建议采用方法与机制结合的定位：

**Learning When to Accept Background Prompts**，而不是 **Universal Prompt Policy Learning**。

核心贡献顺序：

1. dense all-instance intent 的 reference fg/bg matching；
2. 冻结 matcher/SAM 的两候选响应 gate；
3. nested-CV 与患者级严格评估；
4. 外部域和 segmenter 的失败边界。

不能使用的表述：

- 首次使用背景 prompt；
- 首次自动生成点；
- model-agnostic universal policy；
- 在 MoNuSeg 上优于 FoB；
- SOTA。

## 4. 为什么域内 MoNuSAC 没有推翻主线

MoNuSAC validation 在 46 patches/9 patients 上选择 N0，而 official-test 事后均值显示 N5 更高。域内 gate 相对 validation-frozen N0 有患者级正收益，但没有超过 zero-shot gate 或 global N5。这一结果否定了“每域训练一次就能自动找到域最优参数”的强假设，却保留了更窄且可发表的结论：SAM response 中存在可学习的病例级纠错信号，但 anchor calibration 受小验证集和领域分布影响。

## 5. 最接近工作 FoB

FoB 官方仓库和权重已审计：

- 代码：`external/FoB_SAM/`，commit `40d0e23...`
- 权重：仅 CHAOST2/SABS
- 无 MoNuSeg/MoNuSAC loader 或 checkpoint
- 官方默认 SAM ViT-H，与本文主执行器不同

因此当前只在相关工作中正面定位，不做数值胜负声明。投稿前应按 nested-CV inner-train/inner-val/outer-test 适配，详见 `docs/research/fob_baseline_reproducibility_audit_20260716.md`。

## 6. 论文资产

- 中文工作稿：`docs/research/prompt_policy_paper_draft_zh_20260715.md`
- 最终主图：`runs/research_direction/paper_artifacts_20260715/final_main_results.png`
- 主表：`.../final_performance_table.csv`
- 成对效应：`.../final_paired_effects.csv`
- MoNuSeg 定性：`.../monuseg_nested_gate_qualitative.png`
- MoNuSAC 定性：`.../monusac_within_domain_gate_qualitative.png`
- FoB 审计：`docs/research/fob_baseline_reproducibility_audit_20260716.md`

## 7. 投稿前执行门槛

| 状态 | 项目 |
| --- | --- |
| [x] | MoNuSeg 37-query 5-fold nested-CV |
| [x] | FG-only vs FG-BG 严格消融 |
| [x] | N={0,5,10,20} 固定预算曲线 |
| [x] | learned vs fold anchor / global N5 / oracle gap |
| [x] | CPU/GPU 数值一致性审计 |
| [x] | MoNuSAC zero-shot patient-disjoint 验证 |
| [x] | MoNuSAC within-domain patient-disjoint 验证 |
| [x] | 成功与失败定性图，含 reference image/mask |
| [x] | FoB 官方发布物审计 |
| [ ] | FoB 病理核同协议至少一折可运行适配 |
| [x] | 第二 segmenter learned-gate 复核：SAM-Med2D 严格负结果 |
| [x] | segmenter-only runtime/VRAM/参数量表 |
| [ ] | 英文稿与 BibTeX 冻结 |

完成前三个未勾项后，达到投稿前内部审稿条件。
