# RAMP 特征: 在线 vs 缓存分析

> 基于 `run_phaseU_ramp.py` 中 `FEATURE_COLS` 的实际定义和 `RAMPPredictor` 的使用方式。

---

## 特征分类

### 当前代码中使用的 4 个特征 (FEATURE_COLS, 代码第 85-90 行)

| # | 特征名 | 列名 | 来源 | 在线/缓存 | 说明 |
|:---:|:---|:---|:---|:---:|:---|
| f1 | matching_map_disagreement | 匹配图不一致度 | Matcher (DINOv2) | **缓存** | 需要 query+reference 的配对匹配，但可预计算 |
| f2 | candidate_jaccard_mean_k1 | 候选区域 Jaccard | Matcher (DINOv2) | **缓存** | 需要 query+reference 的配对匹配，但可预计算 |
| f3 | prompt_overlap_mean_k1 | 提示点重叠度 | Matcher (DINOv2) | **缓存** | 需要 query+reference 的配对匹配，但可预计算 |
| f4 | prediction_jaccard_k1_within | 预测 mask 一致性 | SAM Segmenter | **缓存** | 需要 SAM 输出，但可预计算后缓存 |

### 第 5 个不确定性信号 (不在 FEATURE_COLS 中)

| # | 信号名 | 列名 | 来源 | 在线/缓存 | 说明 |
|:---:|:---|:---|:---|:---:|:---|
| f5 | std_dice_k1 | Dice 标准差 | SAM Segmenter | **缓存** | 用于 MC Dropout 基线和 per-ref Dice 模拟 |

---

## 当前代码的实际缓存策略

**所有特征都是预计算并缓存的**。当前 RAMP 脚本 (`run_phaseU_ramp.py`) 不运行任何 Matcher 或 SAM 推理，而是通过 `load_episode_data()` 从 CSV 加载所有特征。

```python
# 代码第 65-70 行
def load_episode_data(wp_dir: Path) -> pd.DataFrame:
    path = wp_dir / "phase2_episode_instability.csv"
    return pd.read_csv(path)
```

特征预计算发生在 Phase T/U 的其他脚本中（不在 `run_phaseU_ramp.py` 内）。

---

## 如果推演到真正的在线推理场景

### 必须在线计算的特征 (每个新 query)

| 特征 | 在线计算内容 | 成本 | 依赖 |
|:---|:---|:---|:---|
| f1 | 对每个 reference 运行 Matcher 得到 matching map，计算 disagreement | K 次 matcher call | query + reference 图像对 |
| f2 | 对每个 reference 的 candidate region 计算 Jaccard with query | K 次 matcher call | query + reference 图像对 |
| f3 | 对每个 reference 的 prompt points 计算与 query 的空间重叠 | K 次 matcher call | query + reference 图像对 |
| f4 | 运行所有 K 个 SAM segmenter，计算预测 mask 间的 Jaccard | K 次 SAM call | 所有 reference 的 SAM 输出 |
| f5 | 从 K 次 SAM 输出计算 Dice 标准差 | K 次 SAM call | 所有 reference 的 SAM 输出 |

**结论**: 在真正的在线推理中，所有 5 个特征都需要在线计算，因为它们都依赖于 query-reference 配对。唯一的例外是 reference 图像本身的 DINOv2 特征可以缓存（如果 reference 集固定）。

### 可以缓存的中间结果

| 缓存项 | 条件 | 缓存策略 |
|:---|:---|:---|
| Reference 的 DINOv2 特征 | reference 集固定 | 预计算一次，所有 query 复用 |
| Reference 的 SAM mask | reference 集固定 | 预计算一次，但通常不需要 |
| 风险模型 (GBM) | 训练完成后 | 序列化为 `.pkl`，所有 query 复用 |
| Reference 的 prompt points | reference 集固定 | 预计算一次 |

### 绝对不能缓存的 (每个 query 必须重新计算)

| 项 | 原因 |
|:---|:---|
| Query 的 DINOv2 特征 | 每个 query 图像不同 |
| Query-reference matching map | 配对依赖 |
| Query 的 SAM mask | 每个 query 图像不同 |
| Per-reference Dice | 需要 query 的 GT 或 SAM 输出 |

---

## 训练 vs 推理边界

| 组件 | 训练阶段 | 推理阶段 | 缓存 |
|:---|:---|:---|:---|
| Matcher (DINOv2) | 冻结，不训练 | 在线运行 | 不可缓存 (per query) |
| Segmenter (SAM) | 冻结，不训练 | 在线运行 | 不可缓存 (per query) |
| 风险预测器 (GBM) | 训练 (需要标注数据) | 推理 (predict_proba) | **可缓存** (模型文件) |
| 自适应权重 | 无训练 | 在线计算 | 不可缓存 (per query) |
| Prompt config 选择 | 无训练 | 在线计算 | 不可缓存 (per query) |
| 共识融合 | 无训练 | 在线计算 | 不可缓存 (per query) |

---

## 关键发现

1. **当前代码中所有特征都是缓存的**：RAMP 脚本是纯评估脚本，消费预计算的特征表。

2. **在真正的在线推理中，所有特征都是 query-dependent 的**：f1-f4 都需要当前 query 与 reference 的配对，无法提前缓存（除了 reference 的 DINOv2 特征）。

3. **风险模型是唯一可缓存的组件**：训练好的 GradientBoosting 模型可以序列化并在推理时加载。

4. **f4 和 f5 需要所有 SAM 输出**：这是 RAMP 必须是 post-hoc 的根本原因——必须先运行所有 SAM，才能计算 mask consistency (f4) 和 uncertainty (f5)。

5. **没有"pre-SAM risk predictor"**：当前代码的 4 个特征中，f4 需要 SAM 输出。如果只使用 f1-f3（matcher-only），可以构建 pre-SAM 风险预测器，但当前代码未实现此变体。