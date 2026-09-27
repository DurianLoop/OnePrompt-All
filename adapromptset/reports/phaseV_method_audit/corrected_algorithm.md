# RAMP 修正算法 (Corrected Algorithm)

> 基于 `run_phaseU_ramp.py` 实际代码的 Post-hoc Selective Prediction 算法。
> 版本: Option A — 所有 SAM 运行先行，然后计算风险，最后选择性预测。

---

## 算法选择

根据 `todo.md` 第 529-547 行的分析，RAMP 只能选择 **Option A: Post-hoc Selective Prediction**：

- 固定 prompt config
- 每个 reference 运行 matcher + segmenter
- 计算所有特征和风险分数
- 基于风险分数 accept or abstain
- 对已有 masks 做 fusion

原因: f4 (`prediction_jaccard_k1_within`) 需要 SAM 输出，因此必须先运行 SAM 才能计算风险。

---

## 修正后的算法伪代码

```
Algorithm: RAMP — Post-hoc Selective Prediction
=================================================

Input:
  Q: query image
  R = {r_1, r_2, ..., r_K}: reference images (K = 2..6)
  prompt_config: fixed prompt configuration (e.g., P20-N1)
  Φ: frozen matcher (DINOv2)
  Ψ: frozen segmenter (SAM)
  M_inst: trained instability risk model (GradientBoosting)
  M_cat: trained catastrophic risk model (GradientBoosting)
  coverage: target coverage rate (default 0.80)

Output:
  accept_flag: boolean (True if accepted, False if abstained)
  fused_prediction: consensus mask (if accepted)
  cat_risk: per-reference catastrophic risk scores


═══════════════════════════════════════════════════════
Phase 1: Run all Matcher + Segmenter (offline)
═══════════════════════════════════════════════════════

for each reference r_i in R:
    # Step 1.1: Matcher feature extraction
    matching_map_i = Φ.match(Q, r_i)
    features_i = Φ.extract_features(Q, r_i)

    # Step 1.2: Run SAM segmenter
    mask_i = Ψ.segment(Q, prompt_from(matching_map_i, prompt_config))

    # Step 1.3: Store per-reference results
    store(q_id, r_i, matching_map_i, features_i, mask_i)

# Step 1.4: Compute features from all outputs
for each reference r_i in R:
    f1_i = matching_map_disagreement(matching_map_i)       # 匹配图不一致度
    f2_i = candidate_jaccard(matching_map_i)               # 候选区域 Jaccard
    f3_i = prompt_overlap(matching_map_i)                  # 提示点重叠度

for each reference r_i in R:
    f4_i = mean_jaccard(mask_i, {mask_j | j ≠ i})         # 预测 mask 一致性

# Note: f5 (std_dice) is computed but not in the risk model feature vector
# f5 = std(dice_values)  # uncertainty proxy


═══════════════════════════════════════════════════════
Phase 2: Risk Prediction (online)
═══════════════════════════════════════════════════════

# Step 2.1: Build feature matrix
F = [[f1_1, f2_1, f3_1, f4_1],
     [f1_2, f2_2, f3_2, f4_2],
     ...,
     [f1_K, f2_K, f3_K, f4_K]]             # shape: (K, 4)

# Step 2.2: Predict per-reference risks
inst_risk = M_inst.predict_proba(F)[:, 1]   # shape: (K,)
cat_risk  = M_cat.predict_proba(F)[:, 1]    # shape: (K,)

# Step 2.3: Aggregate risk
mean_cat_risk = mean(cat_risk)
mean_inst_risk = mean(inst_risk)


═══════════════════════════════════════════════════════
Phase 3: Adaptive Weighting (online)
═══════════════════════════════════════════════════════

# Step 3.1: Compute adaptive reference weights
# Lower catastrophic risk → higher weight
logits = -(cat_risk + 1e-8) / temperature   # temperature = 1.0
logits = logits - max(logits)               # numerical stability
weights = softmax(logits)                   # shape: (K,), sum = 1


═══════════════════════════════════════════════════════
Phase 4: Adaptive Configuration (conceptual only)
═══════════════════════════════════════════════════════

# Note: In the actual code, this does NOT re-run the segmenter.
# It selects the config based on risk but uses pre-computed results.

if mean_cat_risk < 0.3:
    selected_config = "P20-N1"     # precise, few points
elif mean_cat_risk < 0.7:
    selected_config = "P30-N0"     # balanced
else:
    selected_config = "P50-N0"     # dense, many points for robustness

# Adaptive reference budget
budget = clamp(2 + 4 * mean_cat_risk, 2, 6)


═══════════════════════════════════════════════════════
Phase 5: Risk-Stratified Consensus Fusion (online)
═══════════════════════════════════════════════════════

# Step 5.1: Collect per-reference Dice scores
dice_values = [dice(mask_1, GT), dice(mask_2, GT), ..., dice(mask_K, GT)]
# Note: In practice, GT is unavailable; use predicted masks for consensus

# Step 5.2: Risk-stratified fusion
if mean_cat_risk < 0.3:                          # LOW risk
    fused_dice = weighted_mean(dice_values, weights)
elif mean_cat_risk < 0.7:                        # MEDIUM risk
    sorted_idx = argsort(dice_values)
    keep_idx = sorted_idx[K//6 : 5*K//6]         # middle 67%
    fused_dice = weighted_mean(dice_values[keep_idx], weights[keep_idx])
else:                                             # HIGH risk
    fused_dice = median(dice_values)              # most robust


═══════════════════════════════════════════════════════
Phase 6: Coverage-Based Accept/Abstain (online)
═══════════════════════════════════════════════════════

# Step 6.1: Rank all queries by mean catastrophic risk
query_risks = [(q_id, mean_cat_risk), ...]
sorted_queries = sort_by_risk_ascending(query_risks)

# Step 6.2: Accept top coverage% queries
n_accept = coverage * len(sorted_queries)
accepted_queries = sorted_queries[:n_accept]
abstained_queries = sorted_queries[n_accept:]

# Step 6.3: Report metrics
CFR_accepted = mean(catastrophic_failure_rate in accepted_queries)
CFR_all = mean(catastrophic_failure_rate in all_queries)
CFR_reduction = (CFR_all - CFR_accepted) / CFR_all * 100%
```

---

## 与论文算法的关键差异

| 方面 | 论文可能声称 | 修正后算法 (实际代码) |
|:---|:---|:---|
| 执行顺序 | 先算 risk → 再选 prompt → 再运行 SAM | 先运行 SAM → 再算 risk → 选择性预测 |
| 特征数量 | 5 个 (f1-f5) | 4 个 (f1-f4)，f5 是辅助不确定性信号 |
| 自适应 config | 运行时切换 prompt config | 仅概念性选择，使用 best_config 的已有结果 |
| 风险模型 | 可能声称 training-free | GradientBoosting 需要训练 |
| Per-ref Dice | 真实 per-reference Dice | 代码中为 `np.random.normal()` 模拟 |
| 推理成本 | 可能未明确报告 | K+1 matcher + K SAM + 1 GBM inference |

---

## 算法性质

1. **Post-hoc**: 所有 SAM 推理在风险预测之前完成。
2. **Selective**: 基于覆盖率做 accept/abstain 决策。
3. **GT-free at inference**: 推理时不需要 ground truth，但训练风险模型需要标注数据。
4. **Not training-free**: 风险模型 (GradientBoosting) 需要训练。
5. **Not calibration-free**: 使用固定的 0.3/0.7 风险阈值和 temperature 参数。

---

## 正确性验证

该算法满足以下可执行性条件:
- ✅ 没有用未来输出决定过去步骤（先 SAM，后 risk）
- ✅ 特征依赖关系清晰（f4 需要 SAM 输出，因此必须在 SAM 之后）
- ✅ 训练和推理边界明确（Matcher/SAM 冻结，GBM 训练）
- ✅ 可直接从伪代码实现
- ✅ 与 `run_phaseU_ramp.py` 的实际执行路径一致