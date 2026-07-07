# Matcher Stage 3 Feature-level Instance Matching 最小验证报告

日期：2026-06-24

## 1. 实验目的

本实验继续推进老师提出的 Stage 3：

> instance-level 匹配，threshold 高于 threshold 留下，低于就丢弃。

上一版 P0-4 已经完成 mask-level filtering：用 SAM score、面积、bbox、cluster coverage 和 NMS 从 12 个 proposals 中保留 1 个 mask。  
本阶段 P0-4b 进一步加入 DINO/Matcher 风格的 feature-level instance matching：每个 proposal 不只看 mask 形状，还要和 reference object prototype 做特征相似度匹配。

## 2. 方法概述

本阶段复用 Stage 1 的 DINOv2 feature 提取逻辑：

```text
reference image + reference box/mask
-> DINO patch features
-> reference object prototype
```

对每个 Stage 2 SAM proposal：

```text
proposal mask
-> resize 到 DINO 518x518 patch grid
-> mask 内 patch features 做 weighted mean pooling
-> proposal prototype
-> instance_similarity = cosine(reference_proto, proposal_proto)
```

然后把 `instance_similarity` 加入 Stage 3 filtering：

```text
feature_filter_score =
    instance_similarity
  + 0.1 * sam_score
  + 0.1 * cluster_coverage
  + 0.05 * prompt_coverage
  - area_penalty
  - bbox_penalty
```

这一步比 P0-4 更接近 Matcher 论文中的 instance-level matching 思路。

## 3. 输入与命令

输入：

```text
Stage 1: runs/research_direction/stage1_matching/elephant_pair/
Stage 2 proposals: runs/research_direction/stage2_proposals/elephant_pair/
DINOv2 weights: assets/checkpoints/dinov2_vitl14_pretrain.pth
```

可复现命令：

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\feature_match_stage3_proposals.py --proposals-dir runs\research_direction\stage2_proposals\elephant_pair --stage1-dir runs\research_direction\stage1_matching\elephant_pair --dinov2-weights assets\checkpoints\dinov2_vitl14_pretrain.pth --min-instance-similarity 0.45 --min-score 0.80 --min-area-pct 5 --max-area-pct 45 --max-bbox-area-pct 45 --min-cluster-coverage 0.30 --nms-iou 0.75 --out-dir runs\research_direction\stage3_feature_matching\elephant_pair
```

运行时提示：

```text
xFormers not available
```

这是当前本地 DINO 路径的常见提示，不影响本次输出。

## 4. 实验结果

输出目录：

```text
runs/research_direction/stage3_feature_matching/elephant_pair/
```

总体结果：

| 指标 | 结果 |
| --- | ---: |
| input proposals | 12 |
| feature threshold 后 candidates | 1 |
| NMS 后 retained masks | 1 |
| `K_pred` | 1 |
| retained group id | 5 |
| retained instance similarity | 0.784939 |
| retained feature score | 0.986067 |

按 sample size 的 DINO instance similarity：

| sample size | proposal 数 | similarity mean | similarity max | feature score max |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 0.642958 | 0.772299 | 0.896890 |
| 3 | 4 | 0.701390 | 0.784939 | 0.986067 |
| 5 | 4 | 0.658832 | 0.690144 | 0.671405 |

保留的 proposal：

| group id | sample size | SAM score | instance similarity | mask area | bbox area | cluster coverage | prompt coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 3 | 0.936279 | 0.784939 | 12.7992% | 19.0690% | 0.5750 | 1.0000 |

## 5. 观察

1. Feature-level matching 仍然保留 group 5，和 P0-4 mask-level filtering 一致，说明当前 retained mask 比较稳定。

2. 3 点 prompts 的 DINO instance similarity 平均值最高，`0.701390`，并且最高值 `0.784939` 来自最终保留的 group 5。

3. 5 点 prompts 在 P0-3b 中 SAM score 很高，但 DINO instance similarity 并不最高，且经常因为面积或 bbox 过大被惩罚。这说明 “SAM score 高” 不等于 “和 reference instance 匹配”。

4. 这一步让 Stage 3 的 threshold 更接近老师说的 instance-level match：proposal 必须在 feature 上像 reference object，不能只靠 SAM 自身分数。

## 6. 对老师问题的正面回答

1. “instance-level match 能不能做？”

   可以。当前已经对每个 SAM proposal 计算了 DINO proposal prototype，并与 reference object prototype 做 cosine similarity，得到 `instance_similarity`。

2. “threshold 高于 threshold 留下，低于丢弃怎么落地？”

   当前用 `min_instance_similarity=0.45` 加上 score、面积、bbox、cluster coverage 等约束。最终 12 个 proposals 中只有 1 个通过，并输出 `K_pred=1`。

3. “代码怎么知道 K 个物体存在？”

   当前单目标 case 中，feature-aware filtering + NMS 后 retained masks 数为 1，因此 `K_pred=1`。多实例图中，下一步可以按 cluster 分组后分别做 feature-level NMS，retained 数即为 `K_pred`。

4. “这和 P0-4 有什么区别？”

   P0-4 是 mask-level rule；P0-4b 加入了 DINO feature-level similarity。P0-4b 更接近 Matcher 的 instance-level matching，也更适合作为论文方法的一部分。

## 7. 输出文件

| 文件 | 说明 |
| --- | --- |
| `feature_scores.csv` | 每个 proposal 的 instance similarity 和过滤状态 |
| `feature_scores.json` | 完整结果、阈值、retained masks |
| `feature_retained_overlay.png` | feature-aware retained mask overlay |
| `feature_retained_grid.png` | retained proposal 网格 |
| `summary.md` | 本次运行摘要 |

## 8. 下一步

现在 Stage 1/2/3 的 elephant 最小闭环已经比较完整。下一步建议两条线并行：

1. 多实例自然场景验证：
   - 换一张有多个同类物体的自然图；
   - 验证 DBSCAN 是否得到 `K_pred > 1`；
   - 验证 feature-level instance matching 是否能分别保留多个目标实例。

2. 医学图像迁移：
   - 用一个 reference mask/box 定义医学目标；
   - 在 query image 上跑同样流程；
   - 比较 no feedback、feedback、feature-level filtering 的 false positive 变化。
