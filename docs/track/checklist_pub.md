# 发表主线 Checklist

更新：2026-07-16  
当前阶段：**已经可以写中文论文；正在补投稿前强基线、跨模型复核和运行成本。**

## 全局进度

| 阶段 | 状态 | 一句话结果 |
| --- | --- | --- |
| P0 问题与协议 | [x] | dense all-nuclei reference prompting；MoNuSeg nested-CV + MoNuSAC patient-disjoint |
| P1 Matcher 机制 | [x] | FG-BG 比 FG-only +0.064725 Dice，CI 不跨 0 |
| P2 Prompt gate | [x] | MoNuSeg learned gate 比 fold anchor +0.010097，比 global N5 +0.009183 |
| P3 外部验证 | [x] | MoNuSAC gate 能纠正 N0，但没有超过 post-hoc global N5 |
| P4 论文资产 | [~] | 中文稿、主表、主图、定性图、第二 segmenter 和 segmenter 成本表已完成；FoB 待补 |

## P0：严格协议

- [x] 核验 MoNuSeg 官方规模：37 train + 14 test = 51 case。
- [x] 建立 37 source-train 的 5-fold nested-CV。
- [x] 每个 outer query 使用 5 个当前折 train-only references。
- [x] 五折后 37 个 case 各测试一次，共 185 reference episodes。
- [x] 按 query 聚合，避免把 185 episodes 当独立样本。
- [x] 建立 MoNuSAC patient-disjoint 154 train / 46 val / 101 test。
- [x] 排除与 official-test patient 重叠的 9 个 train patches。
- [x] 按 25 个 test patients 聚合外部统计。

## P1：控制变量后的 Matcher

- [x] 固定 backbone：DINOv2-L/14。
- [x] 固定正点：K=40 top-k。
- [x] 固定背景对比：lambda=0.75。
- [x] 用 reference mask 内特征构造前景 prototype。
- [x] 用 reference mask 外特征构造背景 prototype。
- [x] 验证 FG-only -> FG-BG：0.473325 -> 0.538050。
- [x] 生成 N={0,5,10,20} 曲线。
- [x] 发现 N 越大 object recall 越高，但预测面积增加、Dice 下降。

## P2：Negative-Response Gate

- [x] 动作只保留 N0/N5/N10/N20，冻结 matcher、K、lambda、layout、SAM。
- [x] validation 选择固定 anchor。
- [x] pre-SAM 选择一个 challenger，并强制保留 anchor。
- [x] post-SAM 使用 predicted-area delta、SAM-score delta 等无 GT 特征。
- [x] 测试预算固定为两次 SAM。
- [x] MoNuSeg nested-CV：anchor 0.537458 -> gate 0.547555。
- [x] 对 global N5：0.538372 -> 0.547555。
- [x] per-query oracle=0.552270，gate 关闭 66.1% gap。
- [x] 审计 49 次实际切换：42 成功、7 失败。
- [x] CPU/GPU 128 次同提示审计通过，max Dice diff=1.45e-4。

## P3：泛化与边界

- [x] MoNuSAC frozen MoNuSeg gate：patch +0.007904，patient +0.007617。
- [x] MoNuSAC within-domain gate：patch +0.005951，patient +0.005559。
- [x] 明确负结论：within-domain 没有超过 zero-shot gate。
- [x] 明确负结论：两种 gate 都没有超过 post-hoc global N5=0.544167。
- [x] 固定 fg-bg 跨 segmenter 审计：只在 SAM ViT-B 强正向，不能称 model-agnostic。
- [x] SAM-Med2D 同协议 5-fold gate：delta=-0.000330，CI 跨 0，确认 learned policy 也依赖执行器。
- [x] Joint vs per-point union smoke：当前 dense all-nuclei 主线保留 joint。

## P4：论文与强基线

### 已完成

- [x] 中文论文工作稿已按最终主线重写。
- [x] 最终性能表：`final_performance_table.csv`。
- [x] 最终成对效应表：`final_paired_effects.csv`。
- [x] 最终主图：`final_main_results.png`。
- [x] MoNuSeg 定性成功/失败图，包含 reference image/mask。
- [x] MoNuSAC 定性成功/失败图，包含 reference image/mask。
- [x] FoB 官方仓库已克隆并固定 commit。
- [x] FoB 发布数据、权重和 MoNuSeg 不兼容边界已审计。
- [x] SAM ViT-B / SAM-Med2D 参数量、embedding、两候选 decode 和峰值显存表。
- [x] 当前 benchmark 单元测试全量通过：112 passed。

### 投稿前待完成

- [ ] **P0：FoB MoNuSeg 适配。** 先完成一个 outer fold 的 loader、训练和 test smoke，记录成本；可行后扩 5-fold。
- [x] **第二 segmenter learned gate。** SAM-Med2D 37-query nested-CV 已完成，结果为严格非正向。
- [~] **P1：成本表。** segmenter-only 已完成；完整 DINO matching 端到端部署计时待补。
- [ ] **P1：参考图敏感性。** 对 learned gate 报告 5-reference 方差和最差 reference，而不只报告均值。
- [ ] **P1：英文稿。** 转换摘要、方法、实验和图注，补 BibTeX。
- [ ] **P2：内部审稿。** 检查所有主数字可从证据索引一键回溯。
- [ ] **P2：投稿定位。** 根据 FoB 结果决定方法论文或轻量机制/诊断论文。

## 当前论文结论

可以写：

> Reference 前景/背景证据显著改善 dense all-nuclei matching；学习是否接受背景负点引起的 SAM 响应，在 MoNuSeg nested-CV 中稳定优于固定预算并关闭 66.1% oracle gap。外部 MoNuSAC 仍存在正向纠错信号，但最优预算具有领域依赖。

不能写：

- universal prompt policy；
- model-agnostic；
- 每个领域训练一次即可最优；
- 优于 FoB 官方实现；
- SOTA。

## 入口

- 中文论文：`docs/research/prompt_policy_paper_draft_zh_20260715.md`
- 就绪判定：`docs/research/paper_readiness_decision_20260715.md`
- 证据索引：`docs/research/paper_evidence_index_20260715.md`
- FoB 审计：`docs/research/fob_baseline_reproducibility_audit_20260716.md`
- 论文资产：`runs/research_direction/paper_artifacts_20260715/`
