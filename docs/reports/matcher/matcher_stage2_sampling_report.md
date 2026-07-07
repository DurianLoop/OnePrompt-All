# Matcher Stage 2 聚类采样最小验证报告

日期：2026-06-24

## 1. 实验目的

本实验继续回应老师在 Stage 2/3 中提到的问题：

> Matcher 文章里，他们 sample N% 个匹配得到的 point prompts，然后把每次 sample 出来的 points 送进 SAM，得到一次分割，重复 m 次得到 m 个分割。

当前先完成 Stage 2 的最小闭环：不急着接 SAM3/SAM 生成 mask，而是先验证 feedback 后的 matched points 能否形成合理的 cluster，并能否稳定生成多组 point prompts。

本阶段要回答三个问题：

1. feedback 后的 top points 是否能聚成实例级候选区域？
2. 单目标 elephant case 中，聚类估计的 K 是否合理？
3. 是否可以从每个 cluster 中生成多组 1/3/5 点 prompts，作为下一步 SAM3/SAM proposal 输入？

## 2. 实验输入

输入来自 P0-2 feedback reranking：

```text
runs/research_direction/stage1_feedback/elephant_pair/
```

使用文件：

| 文件 | 用途 |
| --- | --- |
| `feedback_points.json` | feedback 后 top-k 点、相似度、selected 标记 |
| `feedback_overlay_before_after.png` | 对照 Stage 1 feedback 效果 |
| `target_image` | 用于画 cluster 和 sampled prompts overlay |

P0-2 关键结果是：feedback 后 `threshold=0.55` 的 selected points 从 9 增加到 43，并且 top-k 负点邻域候选从 16 降到 0。因此 P0-3 直接使用 feedback 后的 selected points 作为候选点集。

## 3. 可复现实验命令

检查脚本：

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\check_stage2_sampler.py
```

Stage 2 聚类采样命令：

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\sample_stage2_prompts.py --feedback-dir runs\research_direction\stage1_feedback\elephant_pair --selected-only --cluster-method dbscan --eps 55 --min-samples 3 --sample-sizes 1,3,5 --samples-per-cluster 4 --max-anchor-distance 95 --out-dir runs\research_direction\stage2_sampling\elephant_pair
```

参数解释：

| 参数 | 当前值 | 解释 |
| --- | ---: | --- |
| `selected-only` | True | 只使用 feedback 后超过 threshold 的点 |
| `cluster-method` | dbscan | 用 DBSCAN 做实例候选聚类 |
| `eps` | 55 | 点之间 55px 内视为邻近 |
| `min-samples` | 3 | 至少 3 个邻近点形成核心区域 |
| `sample-sizes` | 1,3,5 | 分别生成单点、三点、五点 prompts |
| `samples-per-cluster` | 4 | 每种点数每个 cluster 采样 4 次 |
| `max-anchor-distance` | 95 | 每个 cluster 中只保留离最高分 anchor 不太远的点 |

## 4. 聚类与采样规则

候选点选择：

```text
从 feedback_points.json 中读取 points
如果启用 selected-only，则只保留 selected_by_threshold=True 的点
否则按 top_ratio 保留前 N% 点
```

聚类规则：

```text
使用原图坐标中的欧氏距离做 DBSCAN
cluster_id >= 0 表示候选实例
cluster_id = -1 表示 noise
estimated K = 非 noise cluster 数量
```

采样规则：

```text
每个 cluster 按 rank 从高到低排序
每组 prompt 的第一个点固定为 cluster 内最高分点
其余点从 cluster 内做确定性 spread sampling
如果 cluster 点数不足 sample size，则跳过该组
```

这个规则的好处是可复现、可解释，也方便后续把每组 prompt 送进 SAM3/SAM。

## 5. 实验结果

输出目录：

```text
runs/research_direction/stage2_sampling/elephant_pair/
```

核心结果：

| 指标 | 结果 |
| --- | ---: |
| 输入 selected candidate points | 43 |
| anchor-distance 过滤后 candidate points | 40 |
| DBSCAN cluster count / estimated K | 1 |
| noise points | 0 |
| cluster sizes | 40 |
| prompt group count | 12 |

采样结果构成：

| sample size | 每个 cluster 采样次数 | prompt groups |
| ---: | ---: | ---: |
| 1 | 4 | 4 |
| 3 | 4 | 4 |
| 5 | 4 | 4 |
| total | - | 12 |

## 6. 可视化产物

| 文件 | 说明 |
| --- | --- |
| `cluster_points.csv` | 每个候选点的 cluster_id |
| `prompt_groups.csv` | 每组 sampled prompt 的点坐标和相似度 |
| `prompt_groups.json` | 完整参数、候选点、prompt groups |
| `cluster_overlay.png` | cluster 点分布图 |
| `sample_overlay.png` | sampled prompts 可视化 |
| `summary.md` | 本次实验摘要 |

视觉观察：

- feedback 后的 selected points 在 elephant 主体附近形成 1 个连通 cluster，和单目标 elephant case 的预期一致；
- `estimated K=1` 合理，说明聚类可以作为后续 K 估计的初版；
- 采样出的 prompts 覆盖了目标主体和局部边界，具备送入 SAM3/SAM 生成 proposals 的条件；
- 仍有少数点靠近水面/边界区域，说明 Stage 3 的 mask-level filtering 仍然必要。

## 7. 对老师问题的正面回答

1. “sample N% 个匹配点是否可行？”

   可行。当前用 feedback 后 selected points 作为候选点集，能够生成稳定的 cluster 和 prompt groups。若不用 selected-only，也可以通过 `--top-ratio` 控制前 N%。

2. “重复 m 次得到 m 个分割是否可行？”

   Stage 2 采样层面可行。本次每个 cluster 对 `1/3/5` 三种 sample size 各采样 4 次，共得到 12 组 prompt。下一步只需要把这些 prompt groups 逐组送进 SAM3/SAM，即可得到 m 个 mask proposals。

3. “多个物体很多时，代码怎么知道有 K 个物体？”

   初版可以用 DBSCAN 的非 noise cluster 数作为 `K_pred`。本单目标 case 得到 `K_pred=1`，符合预期。后续需要在多实例图上验证 DBSCAN 是否能把不同实例拆成多个 cluster。

4. “这一步是否已经完成 Stage 2+3？”

   还没有。当前完成的是 Stage 2 的 prompt sampling 输入准备，还没有生成 mask proposals，也没有做 Stage 3 instance-level filtering。

## 8. 下一步

下一步建议直接做 P0-3b：把 `prompt_groups.json` 中的每组点送进 SAM3/SAM，生成 mask proposals。

最小实现：

1. 读取 `prompt_groups.json`；
2. 每个 prompt group 构造成 SAM3/SAM point prompts；
3. 对每组 prompt 输出一个 mask；
4. 保存 `proposal_masks/`、`proposal_overlay.png`、`proposal_scores.csv`；
5. 比较 sample size 1/3/5 的 mask 稳定性和完整性。

接着做 P0-4：

1. 对 proposals 做 mask NMS；
2. 用 DINO/SAM feature 做 instance-level similarity；
3. 设置 threshold，保留高质量 proposals；
4. 输出最终 `K_pred` 和 retained masks。
