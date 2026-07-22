# Phase R 最终报告 (Phase R Final Report)

日期: 2026-07-22  |  Phase: R  |  Final State: `PHASER_REFERENCE_STABILITY_NOT_FEASIBLE_DETERMINISTIC`

## 1. 摘要 (Executive Summary)

Phase R 旨在验证「参考图像稳定性 (reference stability)」是否可通过**确定性共识 (deterministic consensus)** 或**可靠性加权 (reliability weighting)** 来缓解。最终结论: **参考不稳定性被确认存在 (R1 PASSED), 但确定性共识无法降低参考诱导的方差 (R2 FAILED on cond3)**, 因此参考稳定性在确定性方法下不可行。

- **Gate R1**: PASSED — `PHASER_GATE_R1_REFERENCE_INSTABILITY_CONFIRMED` (88.75% queries best-worst gap >= 0.05, mean ref std = 0.0737)
- **Gate R2**: FAILED — `PHASER_GATE_R2_CONSENSUS_NO_GAIN` (7/8 conditions passed; cond3 variance reduction = -5.06%, need >= +25%)
- **Gate R3 / R4 / GlaS**: BLOCKED (R2 未通过)
- **Formal test**: LOCKED_UNUSED
- **计算预算**: 0.2153 / 85 GPU-hours (剩余 84.7847); 总 SAM calls = 10338
- **关键发现**: B2_mean 共识将 mean Dice 从 0.4297 提升至 0.4756 (+0.0460), 并将 catastrophic failure rate 从 12.20% 降至 7.32%, **但方差反而增加 5.06%**, 未达到 R2 的方差缩减条件。

## 2. 背景与动机 (Background)

Phase Y 发现 `REFERENCE_INSTABILITY` 是主要瓶颈: 不同有效参考图像对同一 query 产生显著不同的分割结果。Phase Y 的 stable oracle gap 仅 0.0061 (看似很小), 但这是**聚合层面的伪影** — per-query best-worst Dice gap 均值达 0.1874, 82.9% 的 query gap >= 0.05。

Phase R 的核心假设: 通过对多个参考的 contrast map 进行**共识聚合 (consensus aggregation)**, 可以在不增加 SAM 调用次数的前提下降低参考诱导的方差并提升分割可靠性。Phase R 使用 3 个 prompt config (P20-N1, P30-N0, P50-N0) 而非 Phase Y 的 210-action bank, 以聚焦参考效应本身。

## 3. 数据与协议 (Data and Protocol)

**MoNuSAC 数据集角色:**

| Split | 用途 | Cases |
|---|---|---|
| Train | R1 不稳定性测量 + R2 train 共识选择 | 80 |
| Validation | R2 验证 (consensus 确认) | 41 |
| Holdout | R4 确认 (LOCKED) | 40 |
| Reference pool | 6 refs/query (A=3, B=3) | 80 cases / 18 patients |

**参考重采样协议:** 每个 query 配 6 个参考 (set A=3, set B=3, 患者不重叠)。Train 使用 5 个 resampling seeds; validation 使用固定 A/B 分配 (`validation_AB_fixed.csv`) 以避免多次观测。参考池充分性已确认 (min available train=70, val=80, feasible_for_5_seeds=True)。

## 4. 冻结的 Pipeline (Frozen Pipeline)

- **特征提取**: DINOv2-L (CPU 复用 Phase Y 的 matching maps 与 candidate pools)
- **分割模型**: SAM-B (single call per prediction)
- **Prompt configs (3)**: P20-N1 (20 pos / 1 neg), P30-N0 (30 pos / 0 neg), P50-N0 (50 pos / 0 neg)
- **共识方法 (B0-B7)**: 冻结于 `configs/frozen/phaseR_consensus_methods.yaml`
  - B1_random (随机单参考), B2_mean, B3_median, B4_trimmed, B5_disagreement (beta), B6_quantile (tau), B7_weighted (R3 gated)
- **统计阈值**: 冻结于 `configs/frozen/phaseR_statistics.yaml`
- 所有 config / 阈值在 preregistration commit 冻结, 推理前不可修改。

## 5. Gate R1 结果 (Reference Instability Replication)

**Decision**: `PHASER_GATE_R1_REFERENCE_INSTABILITY_CONFIRMED` — **PASSED**

- Train queries: 80, seeds: 5, configs: 3
- Total SAM calls: 7200 | GPU-hours: 0.13935

| # | Condition | Result |
|---|---|---|
| 1 | >= 30% queries best-worst gap >= 0.05 | PASS (88.75%) |
| 2 | mean ref std >= 0.02 | PASS (0.0737) |
| 3 | direction consistent across 5 seeds | PASS |
| 4 | >= 1 config shows sensitivity | PASS (P20-N1) |
| 5 | not code/coordinate/leakage error | PASS |

**结论**: 参考不稳定性在 train=80 上稳定复现, 且方向跨 5 seeds 一致。R1 通过, 进入 R2。

## 6. Gate R2 结果 (Deterministic Consensus Feasibility)

**Decision**: `PHASER_GATE_R2_CONSENSUS_NO_GAIN` — **FAILED**

- Validation queries: 41 | Best config: P30-N0 | Best method: B2_mean (beta=0.25, tau=0.83)
- Total SAM calls: 3138 | GPU-hours: 0.07546

**关键指标:**

| Metric | Consensus (B2_mean) | B1_random | Delta |
|---|---|---|---|
| Mean Dice | 0.4756 | 0.4297 | +0.0460 |
| Variance | 0.05706 | 0.05431 | -5.06% |
| CFR (Dice<0.1) | 7.32% | 12.20% | -4.88 pp |
| A/B overlap | 0.6075 | 0.5942 | +0.0133 |
| Bootstrap 95% CI | [0.0125, 0.0835] | — | — |

**8 个条件 (7 passed, 1 failed):**

| # | Condition | Result |
|---|---|---|
| 1 | Delta Dice >= 0.010 | PASS |
| 2 | Bootstrap CI lower > 0 | PASS |
| 3 | Variance reduction >= 25% | **FAIL** (-5.06%) |
| 4 | Catastrophic failure not increase | PASS |
| 5 | A/B prediction overlap improve | PASS |
| 6 | Single SAM call | PASS |
| 7 | No query GT used | PASS |
| 8 | >= 2 of 3 configs consistent | PASS (3/3) |

**为何 cond3 失败**: 共识聚合将 per-query 方差从 0.0543 (B1) 增至 0.0571 (consensus), 即方差**增加 5.06%** 而非缩减 >= 25%。尽管 Dice 提升 (+0.0460) 和 catastrophic failure 下降, R2 的核心假设 (确定性共识降低参考诱导方差) 不成立。共识平均掉了部分随机噪声, 但也放大了系统性参考偏差 (systematic reference bias), 导致方差不降反升。

## 7. Gate R3 状态 (Reference Reliability Learnability)

**Status**: `BLOCKED` — R2 未通过。Directive section 12 规定「只有 Gate R2 通过才训练」。可靠性加权共识 (B7) 无法执行。

## 8. Gate R4 状态 (Confirmatory Holdout)

**Status**: `BLOCKED` — R2 和 R3 均未通过。Directive section 13 规定「只有 R2 或 R3 通过」才可访问 holdout。Holdout 40 cases 保持锁定。

## 9. GlaS 诊断 (Secondary Diagnostic)

**Status**: `BLOCKED` — MoNuSAC R2 未通过。Directive section 14 规定「只有 MoNuSAC R2 通过后运行 GlaS」。GlaS 跨数据集诊断无法执行。

## 10. Formal test 状态

**Status**: `LOCKED_UNUSED` — Directive section 22 规定「不访问 formal test」。MoNuSAC test / GlaS testA/testB 全程锁定未使用。

## 11. 计算预算 (Compute Budget)

| Stage | SAM calls | GPU-hours |
|---|---|---|
| R0 audit | 0 | 0.00 |
| Gate R1 | 7200 | 0.13935 |
| Gate R2 | 3138 | 0.07546 |
| **Total** | **10338** | **0.2153** |

Cap: 85 GPU-hours | Used: 0.2153 | Remaining: 84.7847 GPU-hours (使用率 0.25%)。Phase R 远低于预算上限, 主要因 DINOv2 特征 CPU 复用与单次 SAM 调用设计。

## 12. 表格与图索引 (Table & Figure Index)

### 18 Tables

- Table 01: `table_01_dataset_roles.md`
- Table 02: `table_02_reference_resampling.md`
- Table 03: `table_03_historical_phaseY_correction.md`
- Table 04: `table_04_single_reference_instability.md`
- Table 05: `table_05_matching_map_disagreement.md`
- Table 06: `table_06_candidate_overlap.md`
- Table 07: `table_07_prompt_coordinate_overlap.md`
- Table 08: `table_08_prediction_overlap.md`
- Table 09: `table_09_deterministic_consensus_comparison.md`
- Table 10: `table_10_variance_reduction.md`
- Table 11: `table_11_catastrophic_failure.md`
- Table 12: `table_12_reference_utility_prediction.md`
- Table 13: `table_13_weighted_consensus.md`
- Table 14: `table_14_holdout_confirmation.md`
- Table 15: `table_15_glas_diagnostic.md`
- Table 16: `table_16_calls_runtime.md`
- Table 17: `table_17_gate_decisions.md`
- Table 18: `table_18_claim_boundaries.md`

### 20 Figures

- Figure 01: `figure_01_per_query_best_worst_gap.png`
- Figure 02: `figure_02_reference_std_distribution.png`
- Figure 03: `figure_03_matching_map_disagreement.png`
- Figure 04: `figure_04_candidate_jaccard.png`
- Figure 05: `figure_05_single_vs_mean_consensus_scatter.png`
- Figure 06: `figure_06_single_vs_stable_consensus_scatter.png`
- Figure 07: `figure_07_variance_reduction.png`
- Figure 08: `figure_08_worst_reference_improvement.png`
- Figure 09: `figure_09_catastrophic_failure_reduction.png`
- Figure 10: `figure_10_beta_sweep.png`
- Figure 11: `figure_11_quantile_sweep.png`
- Figure 12: `figure_12_reliability_prediction_calibration.png`
- Figure 13: `figure_13_predicted_vs_actual_reference_utility.png`
- Figure 14: `figure_14_reference_weights_distribution.png`
- Figure 15: `figure_15_cross_set_prompt_overlap.png`
- Figure 16: `figure_16_cross_set_prediction_overlap.png`
- Figure 17: `figure_17_monusac_success_failure_cases.png`
- Figure 18: `figure_18_glas_secondary_diagnostic.png`
- Figure 19: `figure_19_gate_flow.png`
- Figure 20: `figure_20_final_method_diagram.png`

BLOCKED 状态: Tables 12-15 (R3/R4/GlaS), Figures 6, 12-14 (R3), Figure 18 (GlaS) 均为 BLOCKED 占位。

## 13. 18 个必答问题 (Mandatory Questions)

**Q1. R1 是否复现了参考不稳定性?**
  是。88.75% queries best-worst gap >= 0.05 (阈值 30%), mean ref std=0.0737 (阈值 0.02), 跨 5 seeds 方向一致。

**Q2. 哪个 prompt config 对参考最敏感?**
  P20-N1 (20 pos / 1 neg)。在所有 config 中表现出最高的不稳定性。

**Q3. 参考不稳定性是随机的还是系统性的?**
  系统性。共识未能降低方差 (反而增加 5.06%), 表明不稳定性主要来自系统性参考偏差而非随机噪声。

**Q4. 确定性共识 (B2_mean) 是否提升 mean Dice?**
  是。从 0.4297 (B1) 提升至 0.4756 (+0.0460), bootstrap 95% CI [0.0125, 0.0835]。

**Q5. 共识是否降低了方差?**
  否。方差从 0.0543 增至 0.0571, 增加 5.06%, 未达到 >= 25% 缩减的 cond3 条件。这是 R2 失败的根本原因。

**Q6. 共识是否降低了 catastrophic failure?**
  是。CFR 从 12.20% 降至 7.32%, cond4 PASS。

**Q7. 共识是否改善了 A/B 跨参考集预测重叠?**
  是。consensus overlap 0.6075 > single 0.5942, cond5 PASS。

**Q8. 哪个共识方法最优?**
  B2_mean (在 train 上选择, beta=0.25, tau=0.83 用于 B5/B6 对比)。B3_median / B4_trimmed / B5 / B6 均未超越 B2_mean。

**Q9. beta/tau sweep 是否显示明确最优?**
  B5_disagreement 在 beta=0.25 最优; B6_quantile 在 tau=0.83 (P30-N0/P50-N0) 或 0.67 最优, 但增益有限。

**Q10. R2 是否通过?**
  否。8 个条件中 7 通过, cond3 (variance reduction >= 25%) 失败。

**Q11. R3 (可靠性学习) 是否执行?**
  否。R2 未通过, R3 BLOCKED。

**Q12. R4 (holdout 确认) 是否执行?**
  否。R2/R3 均未通过, R4 BLOCKED, holdout 40 cases 锁定。

**Q13. GlaS 跨数据集诊断是否执行?**
  否。MoNuSAC R2 未通过, GlaS BLOCKED。

**Q14. Formal test 是否被访问?**
  否。全程 LOCKED_UNUSED。

**Q15. 计算预算是否超标?**
  否。使用 0.2153 / 85 GPU-hours (0.25%), 剩余 84.7847。

**Q16. matching-map disagreement 是否与不稳定性相关?**
  matching_map_disagreement 均值较高, 表明 DINOv2-L 匹配图跨参考差异显著, 与 Dice 不稳定性一致。

**Q17. candidate/prompt overlap 跨参考是否低?**
  candidate_jaccard_mean 与 prompt_overlap_mean 均偏低, 表明参考特定的候选池与 prompt 坐标差异大。

**Q18. Phase R 的最终结论是什么?**
  `PHASER_REFERENCE_STABILITY_NOT_FEASIBLE_DETERMINISTIC` — 参考不稳定性真实存在, 但无法通过确定性共识缓解 (方差不降反升); 需要新方法 (如可靠性学习) 但被 R2 gating 阻塞。

## 14. 允许的结论 (Allowed Claims)

Per directive section 20, 以下结论是**允许的**:

1. **Valid references induce substantial segmentation variability under the current frozen matching-and-prompting pipeline.**
2. Deterministic consensus (B2_mean) improves mean Dice (+0.0460) and reduces catastrophic failures (-4.88 pp), but does NOT reduce reference-induced variance.
3. Reference instability is systematic, not purely random — consensus averaging does not reduce variance.

## 15. 永久禁止的结论 (Prohibited Claims)

Per directive section 20, 以下结论**永久禁止**:

1. ❌ "Reference instability solved / eliminated" (方差未降低)
2. ❌ "Universal across datasets" (仅 MoNuSAC 测量, GlaS BLOCKED)
3. ❌ "SOTA / state-of-the-art segmentation"
4. ❌ "Formal test success" (test LOCKED_UNUSED)
5. ❌ "Reliability-weighted consensus works" (R3 BLOCKED)
6. ❌ "Holdout-confirmed" (R4 BLOCKED)
7. ❌ "Variance reduction achieved" (cond3 FAILED, 方差增加 5.06%)

## 16. 复现命令 (Reproduction)

```bash
# One-command reproduction
git checkout 7317949
python src/adapromptset/scripts/run_phaseR_gateR1.py
python src/adapromptset/scripts/run_phaseR_gateR2.py
python src/adapromptset/scripts/run_phaseR_final_report.py
```

运行环境: `D:\SAM\conda_envs\sam_gpu` (Python + matplotlib + numpy)。

## 17. 未解决的阻塞项 (Unresolved Blockers)

1. **方差缩减机制缺失**: 确定性共识 (mean/median/trimmed/disagreement/quantile) 均未降低 per-query 方差, 反而增加 ~5%。这表明参考不稳定性包含系统性成分, 简单平均无法消除。
2. **可靠性学习 (R3) 被阻塞**: 需要学习 per-reference 可靠性权重 (B7) 来降低方差, 但 R2 gating 阻止了 R3 执行。需要一个不依赖 R2 pass 的新方法路径。
3. **Holdout 确认缺失**: 无 holdout 证据支持任何稳定性声明。
4. **跨数据集泛化未知**: GlaS 诊断 BLOCKED, 无法声称跨数据集结论。

**建议方向**: 重新设计方差缩减方法 (如 query-adaptive reliability estimation, 或基于参考特征的 gating), 并在新的 preregistration 下重新评估。
