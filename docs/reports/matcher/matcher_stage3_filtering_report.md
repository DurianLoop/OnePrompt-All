# Matcher Stage 3 Instance-level Filtering 最小验证报告

日期：2026-06-24

## 1. 实验目的

本实验回应老师在 Stage 3 中提出的问题：

> instance-level 匹配，threshold 高于 threshold 留下，低于就丢弃；如果分割出 K 个物体，代码应该能明白有 K 个物体存在，然后一个个和 target match。

前两步已经完成：

```text
P0-2: feedback 改善 Stage 1 matched points
P0-3: feedback 后 points 聚类采样，得到 12 组 point prompts
P0-3b: 12 组 prompts 送入 SAM，得到 12 个 mask proposals
```

本阶段做最小 Stage 3：不直接上 Matcher 论文里的完整 EMD / purity / coverage，而是先用可解释的 mask-level filtering 验证 threshold 和 K 估计是否可行。

## 2. 实验输入

输入目录：

```text
runs/research_direction/stage2_proposals/elephant_pair/
```

关键输入文件：

| 文件 | 说明 |
| --- | --- |
| `proposal_scores.json` | 12 个 SAM proposals 的 score、面积、bbox、mask 路径 |
| `proposal_masks/` | 每个 proposal 的二值 mask |
| `proposal_grid.png` | 12 个 proposal 可视化总览 |

同时读取 Stage 2 采样点：

```text
runs/research_direction/stage2_sampling/elephant_pair/prompt_groups.json
```

用于计算 proposal 对 cluster points 的覆盖率。

## 3. 可复现实验命令

检查脚本：

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\check_stage3_filtering.py
```

Stage 3 过滤命令：

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\filter_stage3_proposals.py --proposals-dir runs\research_direction\stage2_proposals\elephant_pair --min-score 0.80 --min-area-pct 5 --max-area-pct 45 --max-bbox-area-pct 45 --min-cluster-coverage 0.30 --nms-iou 0.75 --out-dir runs\research_direction\stage3_filtering\elephant_pair
```

## 4. 过滤指标

当前最小过滤器使用以下指标：

| 指标 | 含义 | 当前阈值 |
| --- | --- | ---: |
| `sam_score` | SAM 自身 mask 质量分数 | >= 0.80 |
| `mask_area_pct` | mask 面积占整图比例 | 5% 到 45% |
| `bbox_area_pct` | mask bbox 面积占整图比例 | <= 45% |
| `cluster_coverage` | Stage 2 cluster points 落入 mask 的比例 | >= 0.30 |
| `prompt_coverage` | 当前 prompt group 的点落入 mask 的比例 | >= 1.00 |
| `nms_iou` | retained masks 间的最大 IoU | < 0.75 |

综合排序分数：

```text
filter_score = sam_score
             + 0.2 * cluster_coverage
             + 0.1 * prompt_coverage
             - area_penalty
             - bbox_penalty
```

这个版本的重点是解释性：它能说明 proposal 被丢弃是因为过小、过大、铺满背景、没有覆盖 prompt 点，还是与已保留 mask 重复。

## 5. 实验结果

输出目录：

```text
runs/research_direction/stage3_filtering/elephant_pair/
```

总体结果：

| 指标 | 结果 |
| --- | ---: |
| input proposals | 12 |
| threshold 后 candidates | 1 |
| NMS 后 retained masks | 1 |
| `K_pred` | 1 |
| retained group id | 5 |

保留的 proposal：

| group id | sample size | SAM score | mask area | bbox area | cluster coverage | prompt coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 3 | 0.936279 | 12.7992% | 19.0690% | 0.5750 | 1.0000 |

主要拒绝原因统计：

| reject reason | count |
| --- | ---: |
| `too_large_bbox` | 8 |
| `low_prompt_coverage` | 5 |
| `too_large_area` | 2 |
| `low_score` | 1 |
| `too_small_area` | 1 |
| `low_cluster_coverage` | 1 |

说明：

- 5 点 prompts 虽然 SAM score 高，但很多 proposal bbox 过大或面积过大，被过滤掉；
- 单点 proposal 中有的太小，有的铺到水面/背景，不能直接保留；
- 3 点 group 5 在面积、bbox、cluster coverage 和 prompt coverage 上最均衡，因此被保留；
- 在单目标 elephant case 中，`K_pred=1` 与预期一致。

## 6. 可视化产物

| 文件 | 说明 |
| --- | --- |
| `filtered_scores.csv` | 每个 proposal 的过滤指标、状态、拒绝原因 |
| `filtered_scores.json` | 完整结果和阈值配置 |
| `retained_overlay.png` | retained mask 叠加图 |
| `retained_grid.png` | retained masks 网格图 |
| `summary.md` | 本次运行摘要 |

视觉观察：

- retained mask 主要覆盖 elephant 主体；
- 过大的水面/背景 masks 被过滤掉；
- retained mask 仍有少量底部水面/边界误差，说明当前 mask-level rule 还不是最终 instance-level matching。

## 7. 对老师问题的正面回答

1. “instance-level threshold 怎么做？”

   当前最小版本已经可以做：把每个 SAM proposal 转成一行 instance-level 指标，按 score、面积、bbox、cluster coverage、prompt coverage 设 threshold，低于阈值或不满足约束就丢弃。

2. “代码怎么知道有 K 个物体？”

   当前版本用 retained mask 数作为 `K_pred`。在这个单目标 case 中，12 个 proposals 经过 threshold 和 NMS 后保留 1 个，因此 `K_pred=1`。

3. “为什么不能只看 SAM score？”

   P0-3b 已经证明 5 点 prompts 的 SAM score 很高，但 mask 经常扩到背景。本阶段也验证了这一点：很多高 score proposal 因为 `too_large_area` 或 `too_large_bbox` 被过滤。

4. “当前是否完成 Matcher Stage 3？”

   完成了 Stage 3 的最小可解释版本，但还不是论文级最终版本。下一步应加入 DINO / Matcher feature-level instance similarity，让 threshold 更接近 Matcher 论文中的 instance-level matching。

## 8. 下一步

建议做 P0-4b：feature-level instance matching。

最小升级：

1. 对 retained/candidate masks 内部提取 DINO feature；
2. 和 reference object feature 或 Stage 1 target prototype 做 cosine similarity；
3. 用 `instance_similarity` 替代或补充当前 `filter_score`；
4. 在多实例自然图上验证 `K_pred` 是否能从 1 扩展到多个实例；
5. 再把这一套迁移到医学图像或小数据集。

如果短期只需要给老师展示，可以先用当前 P0-1 到 P0-4 的 elephant case 做一页流程图：

```text
Stage 1 matched points
-> positive/negative feedback reranking
-> cluster-aware prompt sampling
-> repeated SAM proposals
-> threshold + NMS filtering
-> K_pred = 1
```
