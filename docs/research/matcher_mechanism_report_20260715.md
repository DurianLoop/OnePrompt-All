# MoNuSeg Matcher 机制分析报告

更新：2026-07-15  
组合：DINOv2-L matcher + SAM ViT-B  
协议：固定 train-only reference，14 个 official test query；所有指标逐 query 配对

## 1. 本报告回答什么

本报告只回答 matcher 本身是否提供了有效的 prompt 证据，以及为什么 K 和背景抑制强度值得自适应学习。它不把测试 GT 输入 policy，也不把这里的 oracle 当成可部署结果。

可复现入口：

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python -B code\benchmarks\analyze_matcher_mechanisms.py
```

机器可读结果位于：

- `runs/research_direction/mechanisms/dinov2l_samvitb_strict_test/candidate_mechanisms.csv`
- `runs/research_direction/mechanisms/dinov2l_samvitb_strict_test/method_mechanism_summary.csv`
- `runs/research_direction/mechanisms/dinov2l_samvitb_strict_test/background_suppression_paired.json`
- `runs/research_direction/mechanisms/dinov2l_samvitb_strict_test/oracle_summary.json`

## 2. 指标如何计算

### 2.1 Point precision@K

把某个 prompt action 选出的 K 个点映射回 query 原图。点落在任意人工标注 nucleus mask 内记为命中：

```text
point precision@K = 命中 nucleus 的点数 / K
```

它直接检查 matcher 选出的点是不是目标点，不经过 SAM，因此不会被 SAM 的分割行为混淆。

### 2.2 Component coverage@K

把二值 GT mask 分解为独立连通域。只要某个 nucleus 连通域内至少有一个 matched point，就认为该实例被点覆盖：

```text
component coverage@K = 被至少一个点命中的 GT nucleus 数 / GT nucleus 总数
```

MoNuSeg 每张图有数百个核，而当前最多只给 75 个点，因此 coverage 的绝对值不会接近 1；此处重点是同样点预算下的相对提升。

### 2.3 Patch AP/AUC

把 query GT 映射到 DINO 37 x 37 patch 网格。含任意 nucleus 像素的 patch 为正，其余为负，再评价 matcher score map 对正负 patch 的排序。该指标检查整张热力图，而 point precision@K 只检查最靠前的 K 个位置。

## 3. 背景抑制的配对机制证据

控制 K=40 和 top-k 布局不变，只比较：

- 前景匹配：`fg_topk_K40`
- 前景减背景：`contrast_topk_K40_lam0.75`

| 指标 | fg-only | fg-bg | 配对变化 | 95% bootstrap CI | case 方向 |
| --- | ---: | ---: | ---: | ---: | --- |
| point precision | 0.5464 | 0.8054 | +0.2589 | [0.2000, 0.3196] | 14 胜 / 0 平 / 0 负 |
| component coverage | 0.0518 | 0.0778 | +0.0260 | [0.0203, 0.0324] | 14 胜 / 0 平 / 0 负 |
| Dice | 0.4404 | 0.5151 | +0.0746 | [0.0500, 0.1038] | 14 胜 / 0 平 / 0 负 |
| predicted area (%) | 72.40 | 59.92 | -12.48 | [-18.64, -6.53] | 13 个 case 面积下降 |

直白解释：只问“哪里像 nucleus”时，很多与 nucleus 外观相似的组织区域也会排到前面；再问“这个位置是否也像 reference 背景”，并用适度权重扣除背景相似度后，前 40 个点更集中在真实 nucleus 内。点本身更准，SAM 随后得到的错误扩张更少，所以 Dice 同步提高。

这组证据支持 H3：reference mask 的背景不是无用区域，而是能排除相似干扰的负证据。

该方向在三个 split 上保持一致，而不是只在 test 偶然出现：

| Split | Episodes | Point precision delta | Dice delta | Dice 95% CI |
| --- | ---: | ---: | ---: | ---: |
| train | 28 | +0.1902 | +0.0651 | [0.0400, 0.0940] |
| validation | 8 | +0.0688 | +0.0643 | [0.0193, 0.1296] |
| test | 14 | +0.2589 | +0.0746 | [0.0500, 0.1038] |

因此“适度使用 reference background 能改善 matcher”可以作为稳定机制结论；具体最优 lambda 仍只能由 train/validation 决定。

## 4. 为什么 lambda 不能无限增大

K=40 时：

| lambda | point precision | component coverage | Dice | predicted area (%) |
| ---: | ---: | ---: | ---: | ---: |
| 0（fg-only） | 0.5464 | 0.0518 | 0.4404 | 72.40 |
| 0.25 | 0.7018 | 0.0666 | 0.5039 | 60.07 |
| 0.75 | **0.8054** | **0.0778** | **0.5151** | 59.92 |
| 1.50 | 0.4125 | 0.0378 | 0.4119 | 79.52 |

`lambda=1.5` 明显过强。原因不是“背景没用”，而是 nucleus 与周围组织并非完全可分；过度扣除背景相似度会把一部分真实 nucleus 一并压低，并把异常位置推到 top-k。于是 lambda 本身就是需要由验证集选择、或由 prompt policy 根据 query 动态决定的变量。

## 5. 固定 K 不是逐 case 最优

14 个 test query 的离线 oracle action 使用了四种 K：`10, 20, 40, 75`，以及两种 lambda：`0.25, 0.75`。固定 action `K=40, lambda=0.75` 的平均 Dice 为 0.5151；逐 case oracle 为 0.5822，保留了 0.0671 的平均可学习空间。

只看 test 时，oracle K 与 GT nucleus 连通域数量的 Spearman 相关系数为 0.667；但同一分析在 train 为 -0.484，在 validation 为 0.000，方向并不稳定。因此不能解释成“核越多就应选更大的 K”。可靠结论只是不同 query 的 oracle K 确实不同；决定 K 的机制不是单一目标密度，仍可能同时依赖 reference-query 匹配质量、score-map 模糊度、点的空间冗余和 SAM 的合并行为。

| Split | Episodes | Oracle K | Oracle gap | Spearman(K, GT components) |
| --- | ---: | --- | ---: | ---: |
| train | 28 | {5,20,40,75} | 0.0471 | -0.484 |
| validation | 8 | {20,40,75} | 0.0271 | 0.000 |
| test | 14 | {10,20,40,75} | 0.0671 | +0.667 |

图：

- `runs/research_direction/mechanisms/dinov2l_samvitb_strict_test/point_precision_component_coverage.png`
- `runs/research_direction/mechanisms/dinov2l_samvitb_strict_test/oracle_action_distribution.png`
- `runs/research_direction/mechanisms/dinov2l_samvitb_strict_test/oracle_k_vs_gt_components.png`

## 6. 结论边界

已经成立：

1. fg-bg matching 在当前 14-case 严格测试中稳定优于 fg-only；
2. 背景抑制强度存在明显的过弱和过强区间；
3. 不同 query 的最优 K 不同，但 K 与场景核密度的简单相关解释不跨 split 稳定；
4. 固定 action 与离线 oracle 之间仍有 0.0671 Dice 的空间。

尚未成立：

1. learned policy 显著优于 fixed action；v6 只有 +0.00145，置信区间包含 0；
2. 以上规律可迁移到所有 matcher 和所有 SAM；
3. 只靠 score entropy 就能可靠预测 K；当前 oracle K 与 entropy 的秩相关仅为 -0.282。

因此下一步的关键不是继续证明 matcher“有一点用”，而是利用多 reference 训练 episode，把已经存在的 oracle gap 转化为严格测试上的可重复 policy 增益。
