# MonuSeg 单域突破：控制变量版研究方案

日期：2026-07-08

## 1. 为什么要收缩研究范围

前一阶段的方向覆盖了 SAM3 text/point、Matcher Stage1/2/3、feedback、learning、IDRID、MonuSeg、跨域训练和 50 例 5-fold。这个路线能证明“整条 pipeline 可行”，但现在已经开始暴露一个问题：变量太多，任何一个结果好或不好，都很难判断到底是哪一步造成的。

当前更合理的策略是先收缩到一个领域，把一个问题做透。推荐先选择 MonuSeg，原因有四点：

1. MonuSeg 是密集小目标场景，和“多实例、点聚类、候选生成”高度相关。
2. 现有结果显示 MonuSeg 的 Stage3 几乎失效，改善空间最大。
3. MonuSeg 图像尺寸较小，实验成本低，适合快速做变量控制。
4. 如果 MonuSeg 能跑通，再迁移到 IDRID 或其它医学领域时，论文逻辑会更扎实。

新的目标不再是：

```text
每个 domain 只训练一次，然后适配任何领域。
```

而是先改成：

```text
在 MonuSeg 一个领域内，把 matched points 如何进入 SAM 这件事做清楚，证明可控点选择和聚类策略能稳定提高候选 mask 质量。
```

## 2. 当前 MonuSeg 事实基础

当前 50 例扩展实验中，MonuSeg 部分有：

| 项目 | 数值 |
| --- | ---: |
| MonuSeg cases | 20 |
| 每个 case proposals | 12 |
| MonuSeg proposals | 240 |
| Dice 阈值正例 | 40 |
| 当前 Stage3 mean Dice | 0.051678 |
| 当前 oracle mean Dice | 0.342465 |

这说明两个事实：

1. Stage3 手写规则在 MonuSeg 上基本不可靠，很多 case 一个 proposal 都不保留。
2. 候选池里已经有一定质量的 mask，但 oracle mean Dice 也只有约 0.34，说明候选生成本身仍然不够强。

因此 MonuSeg 的第一突破点不应该是复杂学习器，而应该是更前面的候选生成：

```text
matched points 如何保留、如何选点、如何聚类、如何送进 SAM。
```

如果候选池质量没有提高，后面的 scoring / learning 再复杂也只是从弱候选里挑一个相对好点的 mask。

## 3. 这阶段先不做什么

为了控制变量，下面这些暂时不作为主实验：

| 暂不做 | 原因 |
| --- | --- |
| 跨 domain 一次训练通用 | 目标太大，容易把问题讲散。 |
| SAM3 text prompt | 当前主问题是 MonuSeg mask proposal，不是文本语义失败。 |
| IDRID + MonuSeg 混合训练 | 两个领域形态差异太大，会干扰结论。 |
| 复杂 proposal score | 分数来源不够直观，容易变成玄学调参。 |
| 随机组合大量 points | 组合数爆炸，不利于分析因果。 |
| 多轮 feedback loop | 先把无 feedback 的单轮候选生成做稳。 |
| adapter / LoRA / fine-tuning | 会引入训练变量，偏离“点如何进入 SAM”的核心问题。 |

这阶段只回答一个更窄的问题：

```text
在 MonuSeg 中，什么样的 matched point selection + clustering + SAM prompt 组合，能稳定生成更好的 nuclei masks？
```

## 4. 固定变量与保留变量

### 4.1 固定变量

为了让实验可解释，以下内容固定：

| 固定项 | 固定方式 |
| --- | --- |
| Domain | 只用 MonuSeg。 |
| Support / reference | 固定一张或一组 MonuSeg support image + manual mask。第一版建议沿用现有 support。 |
| Query set | 使用当前 20 个 MonuSeg query cases。 |
| Feature backbone | 固定当前 DINO / Matcher Stage1 特征。 |
| SAM 模型 | 固定当前 SAM checkpoint 和 model type。 |
| 输入 prompt 类型 | 只用 positive point prompt，暂不加入 box / text / negative point。 |
| SAM multimask 选择 | 每组 prompt 仍取 SAM 返回分数最高的 mask。 |
| 评价指标 | 用 GT 计算 Dice / IoU / oracle Dice。 |
| 每个 case 最大候选数 | 固定预算，比如 12 或 16，避免无限生成 candidates。 |

### 4.2 保留的三个核心变量

只保留三个最关键变量：

| 变量 | 要回答的问题 | 候选取值 |
| --- | --- | --- |
| 保留几个点 | 一个 SAM prompt 里放几个 positive points 最稳？ | 1, 2, 3, 5 |
| 点在哪里 | 从 matched points 中选哪些点进入 SAM？ | top similarity、cluster medoid、diverse top-k、local maxima |
| 聚类算法 | matched points 是否需要先按实例分组？ | none、DBSCAN、heatmap connected components、KMeans 作为对照 |

核心原则：

```text
先控制候选生成，再谈候选筛选。
```

## 5. 三个核心变量怎么定义

### 5.1 变量一：保留几个点

这里的“保留几个点”不是 Stage1 保留 top-k 点，而是指：

```text
每一次送入 SAM 的 point prompt 包含几个正点。
```

比如：

| prompt size | 含义 |
| --- | --- |
| 1 point | 一个点生成一个 mask，最简单、最可控。 |
| 2 points | 两个点共同提示一个 nuclei 区域，可能提高完整性。 |
| 3 points | 提供更多目标区域信息，但可能让 SAM 合并多个实例。 |
| 5 points | 信息更多，但风险是把多个 nuclei 或背景粘在一起。 |

初始假设：

```text
MonuSeg 的目标是密集细胞核，单点可能切得碎，多点可能切得完整，但点太多会导致实例粘连。
```

所以实验要找的是 MonuSeg 中最合适的 prompt size，而不是默认越多越好。

### 5.2 变量二：点在哪里

Stage1 会产生一批 matched points，每个点有 similarity。现在不要再随机组合，而是定义几种确定性的点选择策略。

#### 策略 A：top similarity

直接选相似度最高的点：

```text
prompt = top-n similarity points
```

优点：最简单，和 Stage1 相似度直接相关。  
风险：top points 可能集中在同一个局部，缺少空间覆盖。

#### 策略 B：cluster medoid

先聚类，每个 cluster 选一个最代表性的点。  
medoid 可以理解为：

```text
cluster 内离其它点平均距离最近，且 similarity 较高的点。
```

优点：点更代表一个局部实例区域。  
风险：如果聚类错了，medoid 也会错。

#### 策略 C：diverse top-k

在高相似度点里做空间去重，尽量选分散的点：

```text
先取 top-M points，再用最小距离约束或 farthest-point sampling 选 n 个点。
```

优点：避免多个点挤在同一小块。  
风险：点太分散时，SAM 可能把多个实例合并。

#### 策略 D：local maxima

在 similarity heatmap 上找局部极大值，而不是直接取 patch-level top-k。

优点：更像“每个疑似 nuclei 一个峰”。  
风险：需要设置局部窗口大小，但这个窗口比复杂 score 更容易解释。

### 5.3 变量三：聚类算法

聚类的目的是把 matched points 从“一堆点”变成“疑似实例组”。

#### none

不聚类，直接在所有 top points 里选点。  
这是最基础对照。

#### DBSCAN

按空间距离聚类，不需要提前指定 K。

适合回答老师之前的问题：

```text
代码怎么知道有 K 个物体？
```

但 DBSCAN 有两个关键参数：

```text
eps
min_samples
```

为了控制变量，第一版只跑少量固定值，比如：

```text
eps = 16, 24, 32
min_samples = 2
```

MonuSeg 图像通常是 256×256，这些 eps 比自然图小很多。

#### heatmap connected components

把 similarity heatmap threshold 成二值区域，再做连通域。

优点：和视觉热力图更一致，解释性强。  
风险：需要 threshold，但可以固定为 top percentile，比如 top 5% 或 top 10%。

#### KMeans

KMeans 作为对照，不建议作为第一主线。  
原因是 KMeans 需要提前给 K，而 MonuSeg 的 nuclei 数量本身很多且不固定。

如果要用 KMeans，建议只作为附录对照：

```text
K = connected components 数量，或者 K = DBSCAN cluster count。
```

## 6. 新实验分阶段设计

### Stage A：候选生成上限实验，不训练 selector

目的：

```text
先看不同点策略能不能生成更好的候选池。
```

每个 MonuSeg case 固定最多生成 B 个 candidates，例如：

```text
B = 12 或 B = 16
```

对每个策略组合记录：

| 指标 | 含义 |
| --- | --- |
| oracle Dice@B | B 个候选里最好的 Dice。 |
| oracle IoU@B | B 个候选里最好的 IoU。 |
| positive candidate count | Dice 超过阈值的候选数量。 |
| empty / huge mask rate | SAM 生成明显异常 mask 的比例。 |
| candidate diversity | 候选之间是否高度重复。 |

这一步不看 learned score。  
只回答：

```text
哪个点策略能让候选池里出现更好的 mask？
```

如果这一步没有提升，就不要进入复杂 selector。

### Stage B：点数量 ablation

固定：

```text
domain = MonuSeg
cluster = DBSCAN 或当前最稳聚类
point selection = cluster medoid 或 top similarity
candidate budget = 12
```

只改变：

```text
prompt size = 1, 2, 3, 5
```

输出表：

| prompt size | mean oracle Dice | mean oracle IoU | huge mask rate | positive candidates / case |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 待跑 | 待跑 | 待跑 | 待跑 |
| 2 | 待跑 | 待跑 | 待跑 | 待跑 |
| 3 | 待跑 | 待跑 | 待跑 | 待跑 |
| 5 | 待跑 | 待跑 | 待跑 | 待跑 |

预期：

```text
1 或 2 点可能更适合单个 nuclei；3 或 5 点可能提升区域完整性，但也更容易合并实例。
```

### Stage C：点位置 ablation

固定：

```text
prompt size = Stage B 中最优值
cluster = DBSCAN 或 none
candidate budget = 12
```

只改变：

```text
point location strategy
```

比较：

| 点位置策略 | mean oracle Dice | 优点 | 风险 |
| --- | ---: | --- | --- |
| top similarity | 待跑 | 简单直接 | 点可能扎堆 |
| cluster medoid | 待跑 | 代表 cluster | 依赖聚类质量 |
| diverse top-k | 待跑 | 空间覆盖更好 | 容易合并实例 |
| local maxima | 待跑 | 对应 heatmap 峰值 | 需要窗口参数 |

### Stage D：聚类算法 ablation

固定：

```text
prompt size = Stage B 最优
point location = Stage C 最优
candidate budget = 12
```

只改变：

```text
cluster method
```

比较：

| 聚类方法 | mean oracle Dice | estimated K 稳定性 | 解释性 |
| --- | ---: | ---: | --- |
| none | 待跑 | 无 K | 基线 |
| DBSCAN | 待跑 | 中 | 最贴近当前 Matcher 设想 |
| heatmap CC | 待跑 | 高 | 最容易讲图 |
| KMeans | 待跑 | 低/中 | 只做对照 |

### Stage E：固定候选生成后，再训练轻量 selector

只有当 Stage A-D 找到稳定候选生成策略后，才进入 selector。

这时不要再用一堆玄学分数。先只用少量直观特征：

| 特征 | 是否保留 | 原因 |
| --- | --- | --- |
| sam_score | 保留 | SAM 自身置信度，基础信号。 |
| prompt_coverage | 保留 | prompt 点是否落在 mask 内。 |
| cluster_coverage | 保留 | cluster 点是否被 mask 覆盖。 |
| mask_area_pct | 保留 | 控制过大/过小 mask。 |
| bbox_area_pct | 可选 | 和 mask_area 可能冗余，先作为 ablation。 |
| feedback_learning_score | 暂不作为主特征 | 它是混合分数，解释性较弱。 |
| filter_score | 暂不作为主特征 | 它本身就是手写规则，容易把问题绕回 Stage3。 |

轻量 selector 的第一版只回答：

```text
在候选生成固定后，能否用 3-5 个可解释特征选中接近 oracle 的 proposal？
```

## 7. 推荐实验矩阵

不要一开始全组合。建议分三轮跑。

### Round 1：最小定位实验

目的：快速判断 prompt size 是否是主要瓶颈。

| 固定项 | 设置 |
| --- | --- |
| clustering | none |
| point strategy | top similarity |
| prompt size | 1, 2, 3, 5 |
| candidate budget | 12 |

如果 1 点最好，说明多点容易合并实例。  
如果 2 或 3 点最好，说明多点能帮助 SAM 切得更完整。  
如果 5 点明显变差，就可以后续直接砍掉 5 点。

### Round 2：点位置实验

目的：判断 top similarity 是否足够，还是需要空间多样性/cluster medoid。

| 固定项 | 设置 |
| --- | --- |
| prompt size | Round 1 最优 |
| clustering | none 或 DBSCAN |
| point strategy | top similarity / diverse top-k / local maxima / cluster medoid |
| candidate budget | 12 |

### Round 3：聚类实验

目的：判断 MonuSeg 是否需要显式聚类。

| 固定项 | 设置 |
| --- | --- |
| prompt size | Round 1 最优 |
| point strategy | Round 2 最优 |
| clustering | none / DBSCAN / heatmap CC |
| candidate budget | 12 |

如果聚类提升 oracle Dice，说明 “matched points -> instance groups -> SAM prompts” 是核心贡献。  
如果不聚类反而最好，说明 MonuSeg 的问题可能不是 K 估计，而是点本身或 SAM 对 dense nuclei 的响应方式。

## 8. 最重要的汇报图应该是什么

为了避免再次变成表格堆砌，MonuSeg 单域突破需要三类图。

### 图 1：点进入 SAM 的过程图

一行四列：

```text
query image + GT
Stage1 similarity heatmap
selected points for SAM
SAM candidate mask
```

这张图回答：

```text
点在哪里，为什么这些点被选中，这些点进入 SAM 后发生了什么。
```

### 图 2：点数量 ablation 图

同一个 case 展示：

```text
1 point -> mask
2 points -> mask
3 points -> mask
5 points -> mask
```

重点看：

```text
点越多是否导致 nuclei 粘连？
```

### 图 3：聚类策略对比图

同一个 case 展示：

```text
none
DBSCAN
heatmap connected components
```

重点看：

```text
聚类是否把 dense nuclei 分成更合理的 prompt groups。
```

## 9. 成功标准

短期不要用“能适配任何领域”作为标准，改成 MonuSeg 内部的明确标准。

### 9.1 候选生成成功标准

当前 MonuSeg oracle mean Dice 约为：

```text
0.342465
```

第一阶段目标：

```text
controlled candidate generation 后，MonuSeg oracle mean Dice 提升到 0.45 以上。
```

更理想目标：

```text
oracle mean Dice 接近或超过 0.55。
```

### 9.2 筛选成功标准

当前 Stage3 mean Dice 约为：

```text
0.051678
```

如果候选池提高后，轻量 selector 能达到：

```text
selected mean Dice >= oracle mean Dice 的 70%
```

就说明 selector 有实际意义。

例如：

```text
oracle mean Dice = 0.50
selector mean Dice >= 0.35
```

这比单纯 F1 更直接，因为最终目标是 mask 质量。

### 9.3 论文叙事成功标准

能讲出下面这条链：

```text
MonuSeg 中，原 Stage3 失败不是因为完全没有候选，而是点选择和候选生成不稳定。
通过控制 prompt size、point location 和 clustering，我们显著提高候选池 oracle quality。
在候选生成固定后，再用少量可解释特征做 proposal selection。
```

## 10. 重新定义论文主线

旧主线太宽：

```text
Feedback-Learned Matcher for arbitrary-domain promptable segmentation.
```

建议当前主线改成：

```text
Controlled Point-to-Mask Prompting for MonuSeg with Matcher and SAM.
```

中文可以叫：

```text
面向 MonuSeg 细胞核分割的可控点提示生成与聚类驱动 SAM 候选分割。
```

更论文式一点：

```text
Cluster-Aware Point Prompt Selection for Reference-Guided Nuclei Segmentation.
```

这个标题更窄，但更容易做扎实。

## 11. 下一步具体任务

### T1：写一个 MonuSeg controlled runner

输入：

```text
support image / mask
query image / mask
point strategy
prompt size
cluster method
candidate budget
```

输出：

```text
selected_points.csv
prompt_groups.json
proposal_masks/
proposal_overlays/
metrics.csv
summary.json
```

### T2：先跑 Round 1

固定：

```text
cluster = none
point_strategy = top_similarity
prompt_size = 1,2,3,5
candidate_budget = 12
```

输出：

```text
round1_prompt_size_ablation.csv
round1_prompt_size_ablation.html
```

### T3：根据 Round 1 结果砍变量

如果 `prompt_size=5` 明显粘连，就从后续实验移除。  
如果 `prompt_size=1` 已经最好，就重点研究点位置，而不是多点组合。

### T4：再跑 Round 2 和 Round 3

Round 2 只比较点位置。  
Round 3 只比较聚类算法。  
不要交叉全组合。

### T5：最后再做 selector

只有当候选池 oracle Dice 明显提升后，再做 selector。  
第一版 selector 只用：

```text
sam_score
prompt_coverage
cluster_coverage
mask_area_pct
```

暂时不要用 `feedback_learning_score` 作为主结论。

## 12. 给老师的简短说法

可以这样汇报：

```text
老师，我觉得之前的方向有点铺太宽了。现在我想先把问题收缩到 MonuSeg 一个领域，不再同时追求跨域泛化、复杂 learning 和多轮 feedback。因为目前变量太多：点保留多少、点在哪里、怎么聚类、生成多少 candidates、怎么打分，都会影响结果。

我计划先控制变量，只研究三个最关键因素：每次送入 SAM 的点数、这些点在 matched points 中如何选择、以及是否需要聚类。第一阶段不训练复杂 selector，也不强调 score，而是看不同点策略能否提高 MonuSeg 的候选池 oracle Dice。等候选池质量提高后，再用少量可解释特征训练一个简单 selector。

这样做的好处是，论文贡献会更清楚：不是泛泛说适配任意 domain，而是先证明在 MonuSeg 这种密集细胞核场景里，cluster-aware point prompt selection 能显著改善 SAM 候选分割。之后再把这个机制迁移到 IDRID 或其它领域。
```

