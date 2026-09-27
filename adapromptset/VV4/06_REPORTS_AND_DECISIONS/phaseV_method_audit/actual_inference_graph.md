# RAMP 实际推理流程 (Actual Inference Graph)

> 基于 `run_phaseU_ramp.py` 的实际代码执行路径。
> 版本: Post-hoc Selective Prediction (Option A from todo.md)

---

## 整体架构

RAMP 是一个 **后验选择性预测器 (Post-hoc Selective Predictor)**：
所有 SAM 推理先完成，然后从输出中计算风险特征，最后基于风险做选择性预测。

---

## Stage 0: 离线特征预计算 (Pre-computation, 不在 RAMP 推理循环内)

```
对于每个 (query, reference, seed, prompt_config) 组合:
  ├── 运行 Matcher (DINOv2) → 提取 matching map, candidate Jaccard
  ├── 运行 SAM Segmenter → 生成 mask, 计算 Dice
  └── 聚合成 episode-level 特征, 写入 CSV
```

代码中通过 `load_episode_data()` 从 `phase2_episode_instability.csv` 加载这些预计算特征。
因此 RAMP 推理本身 **不运行 Matcher 和 SAM**，而是消费已有的特征表。

实际的 4 个特征 (定义在 `FEATURE_COLS`, 代码第 85-90 行):
| 特征 | 列名 | 来源 |
|:---|:---|:---|
| f1 | `matching_map_disagreement` | Matcher (DINOv2) 匹配图不一致度 |
| f2 | `candidate_jaccard_mean_k1` | Matcher 候选区域 Jaccard 均值 |
| f3 | `prompt_overlap_mean_k1` | Matcher 提示点空间重叠度 |
| f4 | `prediction_jaccard_k1_within` | SAM 预测 mask 间 Jaccard (mask consistency) |

代码中第 5 个不确定性信号 `std_dice_k1` 不在 FEATURE_COLS 中，但被用于:
- `evaluate_mc_dropout_baseline()` 作为 MC Dropout 不确定性代理
- 模拟 per-reference Dice 分布 (`np.random.normal(mean_dice, std_dice, ...)`)

---

## Stage 1: 特征加载与模型训练 (Train, 代码第 686-710 行)

```
输入: episodes_df (包含所有 episode 的特征)
  │
  ├── features = df[FEATURE_COLS].values          # (N, 4) 特征矩阵
  ├── inst_labels = (best_worst_gap >= 0.05)       # 不稳定性标签
  ├── cat_labels = (mean_dice < 0.1)              # 灾难性失败标签
  │
  ├── inst_model = GradientBoostingClassifier()    # 100 trees, depth=3
  │   └── inst_model.fit(features, inst_labels)
  │
  └── cat_model = GradientBoostingClassifier()     # 100 trees, depth=3
      └── cat_model.fit(features, cat_labels)
```

**注意**: 风险模型在每个 pipeline 上独立训练，不是预训练的。

---

## Stage 2: 风险预测 (Risk Prediction, 代码第 110-122 行)

```
对于每个 (query_case_id, seed) 的 reference 组:
  │
  ├── 选择 best config (按 mean_dice_k1 最大)
  ├── features = best_rows[FEATURE_COLS].values   # (N_refs, 4)
  │
  └── ramp.predict_risks(features):
        ├── inst_risk = inst_model.predict_proba(features)[:, 1]  → (N_refs,)
        └── cat_risk = cat_model.predict_proba(features)[:, 1]    → (N_refs,)
```

输出: 每个 reference 的 `inst_risk` 和 `cat_risk`。

---

## Stage 3: 自适应权重计算 (Adaptive Weights, 代码第 124-145 行)

```
mean_cat_risk = mean(cat_risk)

weights = adaptive_reference_weights(cat_risk, temperature=1.0):
  logits = -(cat_risk + 1e-8) / temperature
  logits = logits - max(logits)       # 数值稳定
  weights = softmax(logits)
  → 低风险 reference 获得更高权重
```

---

## Stage 4: 自适应配置选择 (Adaptive Config, 代码第 147-167 行)

```
selected_config = adaptive_prompt_config(mean_cat_risk):
  if mean_cat_risk < 0.3      → "P20-N1"  (精确, 少量点)
  elif mean_cat_risk < 0.7    → "P30-N0"  (平衡)
  else                        → "P50-N0"  (密集, 大批量点)
```

**注意**: 这仅用于报告，实际代码中不重新运行 segmenter，而是用 best_config 的已有结果。

---

## Stage 5: 自适应 Reference 预算 (Adaptive Budget, 代码第 169-183 行)

```
budget = adaptive_reference_budget(mean_cat_risk):
  n = MIN_REFERENCES + (MAX_REFERENCES - MIN_REFERENCES) * mean_cat_risk
  = 2 + 4 * mean_cat_risk
  → 低风险: 2 references, 高风险: 6 references
```

---

## Stage 6: 风险分层共识融合 (Risk-Stratified Consensus, 代码第 185-221 行)

```
fused_dice = risk_stratified_consensus(dice_values, weights, mean_cat_risk):
  if mean_cat_risk < 0.3:
    → weighted mean (稳定, 高效)
  elif mean_cat_risk < 0.7:
    → 修剪均值 (trim middle 67%, 去除异常值)
  else:
    → median (最鲁棒, 不受灾难性失败影响)
```

---

## Stage 7: 覆盖率选择性决策 (Coverage-based Accept/Abstain, 代码第 303-310 行)

```
per_query 按 mean_cat_risk 升序排序
n_accept = coverage * n_total           # 如 80% coverage
accepted = top n_accept 低风险 queries
abstained = 剩余高风险 queries

CFR_reduction = (CFR_all - CFR_accepted) / CFR_all * 100%
```

---

## 完整推理流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                     OFFLINE (Phase T/U 预计算)                       │
│                                                                     │
│  Query ──→ Matcher(DINOv2) ──→ f1, f2, f3                          │
│     │                                                               │
│     └──→ SAM Segmenter ──→ masks, Dice, f4                          │
│                              │                                      │
│                              └──→ phase2_episode_instability.csv    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     ONLINE (RAMP 推理)                               │
│                                                                     │
│  1. 加载预计算特征 (CSV)                                             │
│       │                                                             │
│  2. 训练风险模型 (GradientBoosting)                                  │
│       │  inst_model: f1-f4 → instability risk                       │
│       │  cat_model:  f1-f4 → catastrophic risk                      │
│       │                                                             │
│  3. 对每个 query 的每个 reference:                                   │
│       │  predict_risks(features) → inst_risk, cat_risk              │
│       │                                                             │
│  4. 计算自适应权重 (softmax over -cat_risk)                          │
│       │                                                             │
│  5. 选择 prompt config (基于 mean_cat_risk)                          │
│       │                                                             │
│  6. 风险分层共识融合                                                  │
│       │  low risk → weighted mean                                   │
│       │  med risk → trimmed mean                                    │
│       │  high risk → median                                         │
│       │                                                             │
│  7. 覆盖率选择性预测                                                  │
│       │  按 cat_risk 排序 → accept top 80% → abstain bottom 20%     │
│       │                                                             │
│  输出: fused_dice, CFR_reduction, adaptive_config_distribution       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 关键发现

1. **代码是评估脚本，不是在线推理系统**: RAMP 消费预计算的特征表，不在推理时运行 Matcher 或 SAM。

2. **4 个特征而非 5 个**: `FEATURE_COLS` 只包含 4 个特征。`std_dice_k1` 是第 5 个不确定性信号，但不在风险模型的特征向量中。

3. **自适应配置是概念性的**: 代码选择 prompt config 但不重新运行 segmenter — 它使用 best_config 的已有结果。这避免了循环依赖，但也意味着自适应配置的效果未经验证。

4. **Per-reference Dice 是模拟的**: 代码使用 `np.random.normal(mean_dice, std_dice, ...)` 生成 per-reference Dice 值，而不是实际的 per-reference Dice。这是评估近似。

5. **风险模型在每个 pipeline 上独立训练**: 使用 `GradientBoostingClassifier(100 trees, depth=3)`，在 episode 级别训练 instability 和 catastrophic 两个模型。