# 多实例自然场景验证报告：Horse Head Case

日期：2026-06-24

## 1. 实验目的

本实验把前面 elephant 单目标最小闭环扩展到多实例自然图，正面回应老师的问题：

> 如果有很多物体，代码怎么知道有 K 个物体存在，然后一个个和 target match？

本次使用本地自然图：

```text
reference: code/gradio_demo/images/horse2.png
target:    code/gradio_demo/images/horse3.png
```

`horse3.png` 中有两匹马，适合验证 `K_pred > 1`。为了减少背景干扰，本实验使用 `horse2.png` 的马头/颈部作为 reference object，而不是整匹马。

## 2. 实验流程

完整流程：

```text
Stage 1: DINO/Matcher matched points
-> no-feedback reranking baseline
-> Stage 2 DBSCAN clustering + prompt sampling
-> SAM proposals
-> Stage 3 feature-level instance matching
-> K_pred
```

这次不使用人工正负 feedback，目的是先验证多实例聚类和 K 估计本身。

## 3. 可复现实验命令

### 3.1 Stage 1 matched points

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\export_matcher_stage1_points.py --reference-image code\gradio_demo\images\horse2.png --target-image code\gradio_demo\images\horse3.png --reference-box 120,35,390,330 --top-k 160 --threshold 0.55 --out-dir runs\research_direction\multi_instance_horse_head\stage1_matching
```

结果：

```text
top-k = 160
selected_count = 26
best point = (321.351, 155.973)
best similarity = 0.622770
```

### 3.2 No-feedback reranking

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\apply_stage1_feedback.py --stage1-dir runs\research_direction\multi_instance_horse_head\stage1_matching --positive-points " " --negative-points " " --sigma 42 --positive-weight 0 --negative-weight 0 --top-k 160 --threshold 0.55 --metric-radius 70 --out-dir runs\research_direction\multi_instance_horse_head\stage1_no_feedback
```

结果：

```text
before selected count = 26
after selected count = 26
```

### 3.3 Stage 2 clustering and prompt sampling

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\sample_stage2_prompts.py --feedback-dir runs\research_direction\multi_instance_horse_head\stage1_no_feedback --selected-only --cluster-method dbscan --eps 65 --min-samples 3 --sample-sizes 1,3 --samples-per-cluster 4 --max-anchor-distance 110 --out-dir runs\research_direction\multi_instance_horse_head\stage2_sampling
```

结果：

| 指标 | 结果 |
| --- | ---: |
| candidate points | 26 |
| DBSCAN clusters / estimated K | 2 |
| noise points | 5 |
| cluster sizes | 12, 9 |
| prompt groups | 16 |

可视化上，两个 cluster 分别落在白马头部和棕马头/颈部区域。

### 3.4 SAM proposal generation

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\generate_stage2_sam_proposals.py --stage2-dir runs\research_direction\multi_instance_horse_head\stage2_sampling --checkpoint assets\checkpoints\sam_vit_b.pth --model-type vit_b --device auto --max-groups 16 --out-dir runs\research_direction\multi_instance_horse_head\stage2_proposals
```

结果：

| 指标 | 结果 |
| --- | ---: |
| proposal count | 16 |
| SAM score mean | 0.923479 |
| SAM score min | 0.826171 |
| SAM score max | 0.966604 |

### 3.5 Feature-level instance matching

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\feature_match_stage3_proposals.py --proposals-dir runs\research_direction\multi_instance_horse_head\stage2_proposals --stage1-dir runs\research_direction\multi_instance_horse_head\stage1_matching --dinov2-weights assets\checkpoints\dinov2_vitl14_pretrain.pth --min-instance-similarity 0.45 --min-score 0.80 --min-area-pct 0.5 --max-area-pct 45 --max-bbox-area-pct 50 --min-cluster-coverage 0.30 --nms-iou 0.75 --max-retained-per-cluster 1 --out-dir runs\research_direction\multi_instance_horse_head\stage3_feature_matching
```

新增参数：

```text
--max-retained-per-cluster 1
```

含义是每个 cluster 最多保留一个 feature score 最高的 proposal。这个参数对 K 估计很重要，否则同一个实例的多个局部 proposal 可能都被保留，导致 K 被高估。

## 4. 最终结果

Stage 3 feature-level filtering 结果：

| 指标 | 结果 |
| --- | ---: |
| proposals | 16 |
| feature candidates | 6 |
| retained masks | 2 |
| `K_pred` | 2 |
| retained group ids | 7, 12 |

保留 proposal 详情：

| group id | cluster id | sample size | SAM score | instance similarity | mask area | cluster coverage | prompt coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0 | 3 | 0.959684 | 0.682757 | 13.1212% | 0.7500 | 1.0000 |
| 12 | 1 | 1 | 0.926443 | 0.610000 | 7.4017% | 0.6667 | 1.0000 |

解释：

- cluster 0 对应棕马区域，最终保留 group 7；
- cluster 1 对应白马区域，最终保留 group 12；
- retained masks 数为 2，因此 `K_pred=2`；
- 这与 target 图中两匹马的视觉事实一致。

## 5. 对老师问题的正面回答

1. “多个物体很多时，代码怎么知道有 K 个物体？”

   当前最小方案可行：Stage 2 先用 DBSCAN 从 matched points 中估计 cluster 数，本例得到 2 个 cluster；Stage 3 再对每个 cluster 最多保留一个 feature-level matched proposal，最终 `K_pred=2`。

2. “一个个和 target match 怎么做？”

   当前用 DINO proposal prototype 与 reference object prototype 做 cosine similarity。每个 proposal 都有 `instance_similarity`，再结合 SAM score、cluster coverage、area/bbox 约束筛选。

3. “为什么需要每 cluster 最多保留 1 个？”

   初次运行时 Stage 3 保留了 4 个 proposal，因为同一个 cluster 内可能有多个局部 mask 不完全重叠。加入 `--max-retained-per-cluster 1` 后，每个 cluster 只保留最优 proposal，`K_pred` 从 4 收敛到 2。

4. “这是否说明完整方法可行？”

   说明最小链路可行，但还不是 benchmark 结论。当前只验证了一张自然图，下一步需要更多多实例图和医学图，并进一步优化 reference mask、prompt sampling 和 feature score。

## 6. 输出目录

| 阶段 | 输出目录 |
| --- | --- |
| Stage 1 | `runs/research_direction/multi_instance_horse_head/stage1_matching/` |
| No-feedback | `runs/research_direction/multi_instance_horse_head/stage1_no_feedback/` |
| Stage 2 sampling | `runs/research_direction/multi_instance_horse_head/stage2_sampling/` |
| SAM proposals | `runs/research_direction/multi_instance_horse_head/stage2_proposals/` |
| Stage 3 feature matching | `runs/research_direction/multi_instance_horse_head/stage3_feature_matching/` |

关键可视化：

```text
runs/research_direction/multi_instance_horse_head/stage2_sampling/cluster_overlay.png
runs/research_direction/multi_instance_horse_head/stage3_feature_matching/feature_retained_overlay.png
```

## 7. 下一步

建议下一步做两个小扩展：

1. 再找 2-3 张多实例自然图，验证 `K_pred` 是否稳定；
2. 把同样流程迁移到医学图，尤其是多病灶或多细胞核场景，验证 false positives 是否减少。
