# RAMP 代码执行追踪 (Code Trace)

> 基于 `run_phaseU_ramp.py` 的实际 Python 代码执行路径。
> 函数按调用顺序列出，标注每个函数计算的特征和依赖关系。

---

## 调用顺序

### 1. `main()` (第 639 行)
**入口点**。解析参数，遍历 pipelines，协调全流程。

```
main()
  ├── 参数解析: --wp1-dir, --out-dir, --coverage, --smoke
  ├── 定义 pipelines (4个):
  │     L+SAM-B, L+SAM-L, L+SAM2-T, B+SAM-B
  │
  └── 对每个 pipeline:
        ├── load_episode_data()
        ├── 训练风险模型 (GradientBoosting)
        ├── evaluate_ramp_on_episodes()
        ├── evaluate_ramp_ablations()
        ├── evaluate_mc_dropout_baseline()
        ├── evaluate_deep_ensemble_baseline()
        ├── evaluate_synpo_comparator()
        ├── evaluate_matcher_comparator()
        └── evaluate_persam_comparator()
```

---

### 2. `load_episode_data(wp_dir)` (第 65 行)
**加载预计算的特征数据**。

```
输入: wp_dir (工作目录路径)
  │
  └── 读取 wp_dir/phase2_episode_instability.csv
       └── 返回 pd.DataFrame
            列: query_case_id, seed, config_name,
                 matching_map_disagreement,       # f1
                 candidate_jaccard_mean_k1,       # f2
                 prompt_overlap_mean_k1,          # f3
                 prediction_jaccard_k1_within,    # f4
                 mean_dice_k1, std_dice_k1,
                 best_worst_gap_k1,
                 catastrophic_failure_rate_k1
```

**依赖**: 预计算的 CSV 文件（由 Phase T/U 的 Matcher + SAM 运行生成）。

**注意**: 此函数不执行任何模型推理，仅读取 CSV。

---

### 3. 模型训练 (第 686-710 行, inline in `main()`)
**在 episode 级别训练两个 GradientBoosting 分类器**。

```
输入: df (episode DataFrame)
  │
  ├── features = df[FEATURE_COLS].values           # (N, 4)
  ├── inst_labels = (best_worst_gap_k1 >= 0.05)    # instability: gap >= 0.05
  ├── cat_labels = (mean_dice_k1 < 0.1)            # catastrophic: Dice < 0.1
  │
  ├── inst_model = GradientBoostingClassifier(n_estimators=100, max_depth=3, lr=0.1)
  │   └── .fit(features, inst_labels)
  │
  └── cat_model = GradientBoostingClassifier(n_estimators=100, max_depth=3, lr=0.1)
      └── .fit(features, cat_labels)
```

**依赖**: 4 个特征列 + 2 个标签列（均来自 CSV）。

**输出**: 训练好的 `inst_model` 和 `cat_model`。

---

### 4. `RAMPPredictor.__init__()` (第 96-108 行)
**初始化 RAMP 预测器**。

```
输入: inst_model, cat_model, prompt_configs, risk_thresholds
  │
  └── 存储模型引用和超参数
       self.risk_high = 0.7
       self.risk_low = 0.3
```

---

### 5. `evaluate_ramp_on_episodes(episodes_df, ramp, coverage)` (第 227-328 行)
**核心评估函数**。对每个 (query, seed) 计算 RAMP 全流程。

```
对每个 (query_case_id, seed) 分组:
  │
  ├── 选择 best_config (按 mean_dice_k1 最高)
  ├── features = best_rows[FEATURE_COLS].values   # (N_refs, 4)
  │
  ├── → ramp.predict_risks(features)              # 见 #6
  ├── mean_cat_risk = mean(cat_risks)
  ├── → ramp.adaptive_reference_weights(cat_risks) # 见 #7
  ├── → ramp.adaptive_prompt_config(mean_cat_risk) # 见 #8
  ├── → ramp.adaptive_reference_budget(mean_cat_risk) # 见 #9
  │
  ├── 模拟 per-reference Dice:
  │     per_ref_dice = np.random.normal(mean_dice, std_dice, N_refs)
  │     per_ref_dice = clip(per_ref_dice, 0, 1)
  │
  ├── → ramp.risk_stratified_consensus(per_ref_dice, weights, mean_cat_risk) # 见 #10
  │
  └── 收集结果: fused_dice, mean_cat_risk, config, budget, ...
```

然后进行覆盖率选择性预测:
```
per_query 按 mean_cat_risk 升序排序
n_accept = coverage * n_total
accepted = top n_accept
CFR_reduction = (CFR_all - CFR_accepted) / CFR_all * 100%
```

**依赖**: episodes_df (预计算特征), ramp (训练好的模型)。

---

### 6. `RAMPPredictor.predict_risks(features)` (第 110-122 行)
**预测每个 reference 的风险分数**。

```
输入: features (N_refs, 4)
  │
  ├── inst_risk = inst_model.predict_proba(features)[:, 1]   → (N_refs,)
  └── cat_risk = cat_model.predict_proba(features)[:, 1]     → (N_refs,)
```

**依赖**: 训练好的 GradientBoosting 模型。

**输出**: `inst_risk` 和 `cat_risk` 数组。

---

### 7. `RAMPPredictor.adaptive_reference_weights(cat_risks, temperature)` (第 124-145 行)
**计算自适应 reference 权重: 低灾难性风险 → 高权重**。

```
输入: cat_risks (N_refs,), temperature=1.0
  │
  ├── logits = -(cat_risks + 1e-8) / temperature
  ├── logits = logits - max(logits)      # 数值稳定
  ├── weights = exp(logits)
  └── weights = weights / sum(weights)   # 归一化
```

**依赖**: 仅 `cat_risks`。

**输出**: 归一化权重 (N_refs,)。

---

### 8. `RAMPPredictor.adaptive_prompt_config(mean_cat_risk)` (第 147-167 行)
**根据平均灾难性风险选择 prompt 配置**。

```
输入: mean_cat_risk (标量)
  │
  ├── if mean_cat_risk < 0.3:  return "P20-N1"
  ├── elif mean_cat_risk < 0.7: return "P30-N0"
  └── else:                    return "P50-N0"
```

**依赖**: 仅 `mean_cat_risk`。

**注意**: 此选择仅用于报告，不触发重新推理。

---

### 9. `RAMPPredictor.adaptive_reference_budget(mean_cat_risk)` (第 169-183 行)
**自适应确定 reference 数量**。

```
输入: mean_cat_risk (标量)
  │
  └── n = 2 + 4 * mean_cat_risk    # 线性插值 [2, 6]
      return clamp(n, 2, 6)
```

**依赖**: 仅 `mean_cat_risk`。

---

### 10. `RAMPPredictor.risk_stratified_consensus(dice_values, weights, mean_cat_risk)` (第 185-221 行)
**风险分层共识融合**。

```
输入: dice_values (N_refs,), weights (N_refs,), mean_cat_risk (标量)
  │
  ├── if mean_cat_risk < 0.3:     → weighted mean
  ├── elif mean_cat_risk < 0.7:   → trimmed mean (middle 67%)
  └── else:                        → median
```

**依赖**: `dice_values`, `weights`, `mean_cat_risk`。

---

### 11. `evaluate_ramp_ablations(episodes_df, ramp, coverage)` (第 335-407 行)
**消融实验**。比较 6 种变体:
- `RAMP-full`: 所有组件
- `RAMP-no-weights`: 均匀权重
- `RAMP-no-config`: 固定 P20-N1
- `RAMP-no-budget`: 固定 6 references
- `RAMP-no-stratify`: 总是 weighted mean
- `Baseline`: 原始 mean consensus

**与 evaluate_ramp_on_episodes 相同的循环结构**。

---

### 12. `evaluate_mc_dropout_baseline(episodes_df, n_samples, dropout_rate)` (第 414-452 行)
**MC Dropout 基线** (概念实现，使用 `std_dice_k1` 作为不确定性代理)。

---

### 13. `evaluate_deep_ensemble_baseline(episodes_dfs)` (第 455-501 行)
**Deep Ensemble 基线** (组合 SAM-B + SAM-L + SAM2-T 的预测方差)。

---

### 14-16. 外部比较器

- `evaluate_synpo_comparator()` (第 508-543 行) - SynPo 代理
- `evaluate_matcher_comparator()` (第 550-588 行) - Matcher 代理
- `evaluate_persam_comparator()` (第 595-633 行) - PerSAM 代理

---

## 代码与论文算法的差异

| 方面 | 论文声称 | 实际代码 | 影响 |
|:---|:---|:---|:---|
| 特征数量 | 5 个 (f1-f5) | 4 个 (FEATURE_COLS) | f5 不在风险模型中 |
| 推理方式 | 自适应 prompt config | 使用 best_config 的已有结果 | 自适应配置未实际执行 |
| Per-ref Dice | 真实 per-reference Dice | `np.random.normal()` 模拟 | 近似评估，非精确 |
| 风险模型 | 可能声称 training-free | GradientBoosting 需训练 | 不是 training-free |
| 训练数据 | 可能声称独立 split | 在 episode 级别全量训练 | 无 train/test split |
| Matcher/SAM | 声称在线运行 | 预计算特征，仅读取 CSV | 成本报告不匹配 |
| 特征计算 | 声称在线计算 | 从 CSV 加载 | 实际是离线批处理 |

---

## 关键发现

1. **RAMP 是评估框架，不是在线推理系统**。实际推理中 Matcher 和 SAM 的调用发生在 `phase2_episode_instability.csv` 生成阶段（Phase T/U），而非 RAMP 脚本中。

2. **风险模型在每个 pipeline 上独立训练**，且使用全量 episode 数据训练（无 train/val/test split）。

3. **自适应 prompt config 是概念性的**：代码选择 config 但使用已有 best_config 的结果，不重新运行 segmenter。这避免了循环依赖，但自适应配置的效果未经验证。

4. **Per-reference Dice 被模拟**而非真实值：代码使用 `np.random.normal(mean, std, N)` 生成，引入了随机性。

5. **4 个特征全部来自 Matcher 或 SAM 输出**，没有纯 matcher-only 特征（f1, f2, f3 需要 matcher; f4 需要 SAM）。所有特征都必须在 SAM 运行后获得，因此 RAMP 必然是 post-hoc 的。