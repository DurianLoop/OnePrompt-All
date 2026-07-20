# MonuSeg 单域突破 V2：多点提示与前景-背景对比匹配方案

日期：2026-07-08

## 1. 重新定位：先做 MonuSeg，不再泛化到所有领域

之前的路线覆盖了 SAM3 text/point、Matcher Stage1/2/3、feedback、learning、IDRID、MonuSeg、跨域训练和 50 例 5-fold。它能证明 pipeline 可行，但作为论文主线会显得太宽：变量多，结论容易散。

现在建议先收缩到一个领域：

```text
MonuSeg 细胞核分割
```

原因是 MonuSeg 的任务形态非常适合研究“点如何进入 SAM”：

1. 细胞核是密集、散落的小目标。
2. 目标不是只分一个实例，而是尽量分出整张图中的 all-nuclei foreground。
3. 当前 match 得到的 top-100 points 已经有一定质量，值得研究如何把这些点更好地转成 SAM prompt。
4. 原 Stage3 在 MonuSeg 上效果很弱，说明问题不只是后处理，而是候选生成和 prompt 策略本身需要重新设计。

新的目标不是：

```text
每个 domain 训练一次，然后适配任何领域。
```

而是：

```text
先在 MonuSeg 一个领域里证明：reference-guided matched points 经过合理选择后，多点输入 SAM 能更稳定地得到全图细胞核 mask。
```

## 2. 你的三个核心判断

### 2.1 MonuSeg 更可能需要多点 dense prompt

细胞核散落在图像各处。如果只给 SAM 一个点，它很可能只切出局部一小块；你之前试 SAM3 时也观察到，单点 prompt 倾向于只分割点附近区域，类似的其它细胞核容易被忽略。

所以对 MonuSeg 来说，主假设应该是：

```text
top-100 matched points 如果已经覆盖了多个细胞核区域，那么多个点一起输入 SAM 可能比少量点更适合 all-nuclei segmentation。
```

也就是说，研究重点不应该只是 `1/2/3 points`，而应该把：

```text
K = 5, 10, 20, 50, 100
```

这些 dense point budgets 放到主实验里。

### 2.2 聚类不一定好

聚类的直觉是估计 K 个实例，但 MonuSeg 不一定需要先把每个细胞核分成独立 group。很多有用点可能落在细胞核边缘，或者落在前景纹理强的位置；这些点不一定是 cluster 中心，但可能对 SAM 判断边界更有帮助。

强行聚类可能带来问题：

1. 有用边缘点被当成噪声点丢掉。
2. 相邻细胞核被合成一个 cluster。
3. 全图 foreground 任务被错误地改成“找若干独立实例”的任务。

所以 V2 里聚类不是默认主线，而是对照模块：

```text
no clustering 必须作为强基线。
```

如果 no clustering + dense points 最好，这不是失败，而是一个有价值结论：

```text
Dense nuclei segmentation may prefer distributed matched points over forced instance clustering.
```

### 2.3 reference mask 应同时提供 foreground 和 background

reference 图带 mask，就不应该只用 mask 内前景。mask 外区域同样有价值，因为它告诉我们哪些特征是背景。

现有 matching 主要是：

```text
score(x) = sim(query_x, reference_foreground)
```

更合理的是：

```text
score(x) = sim(query_x, foreground_proto) - lambda * sim(query_x, background_proto)
```

直观说：

```text
我们要找的是“最像细胞核前景、最不像参考背景”的点。
```

这一步可以成为 MonuSeg 单域突破的第一个方法创新点。

## 3. 当前 MonuSeg 基线

当前 50 例扩展实验中的 MonuSeg 部分：

| 项目 | 数值 |
| --- | ---: |
| MonuSeg cases | 20 |
| 每个 case 当前 proposals | 12 |
| MonuSeg proposals | 240 |
| Dice 阈值正例 | 40 |
| 当前 Stage3 mean Dice | 0.051678 |
| 当前 oracle mean Dice | 0.342465 |

这说明：

1. Stage3 手写规则几乎不能解决 MonuSeg。
2. 当前候选池也不够强，oracle mean Dice 只有约 0.34。
3. 因此应该先提高候选生成质量，而不是先堆复杂 score。

新的第一目标：

```text
把 MonuSeg candidate oracle Dice 从 0.34 提高到 0.45+。
```

理想目标：

```text
oracle Dice 接近或超过 0.55，并且 object recall 明显提高。
```

## 4. 这阶段先不做什么

为了控制变量，暂时不做：

| 暂不做 | 原因 |
| --- | --- |
| 跨 domain 一次训练通用 | 目标太大，当前不利于形成清晰结论。 |
| IDRID + MonuSeg 混合训练 | 两个领域形态不同，会干扰 MonuSeg 机制分析。 |
| SAM3 text prompt | 当前问题是点如何进入 SAM，不是文本 prompt。 |
| 大量随机 point combinations | candidate 数爆炸，难以解释。 |
| 复杂 proposal score | 分数太玄，容易掩盖候选生成问题。 |
| adapter / LoRA / fine-tuning | 会引入训练变量，不利于先看点策略。 |

核心原则：

```text
先研究点，再研究分数。
先提高候选池，再训练 selector。
```

## 5. 固定项与核心变量

### 5.1 固定项

| 固定项 | 设置 |
| --- | --- |
| Domain | MonuSeg only |
| 任务 | all-nuclei foreground segmentation |
| Support | 固定 MonuSeg reference image + manual mask |
| Query | 当前 20 个 MonuSeg query cases |
| Feature backbone | 当前 DINO / Matcher feature |
| SAM | 当前 SAM checkpoint 和 model type |
| Prompt 类型 | positive point only |
| 评价 | Dice / IoU / object recall / merge error |

### 5.2 核心变量

| 变量 | 问题 | 候选设置 |
| --- | --- | --- |
| matching score | 点是否像前景且不像背景？ | foreground-only / foreground-background contrast |
| point budget | 这个任务需要多少点？ | 1, 3, 5, 10, 20, 50, 100 |
| point layout | 点如何分布？ | top-k / diverse top-k / local maxima / boundary-aware mix |
| clustering | 是否需要聚类？ | none / DBSCAN / heatmap CC / KMeans 对照 |

这里最重要的变化是：

```text
point budget 不再只看少量点，而是把 dense multi-point prompt 作为主实验。
```

## 6. Stage1：前景-背景对比匹配

### 6.1 foreground prototype

reference mask 内提取 DINO features：

```text
foreground_proto = mean(features where reference_mask == 1)
```

计算 query 每个位置与 foreground prototype 的相似度：

```text
s_fg(x) = cosine(feature_q(x), foreground_proto)
```

这是当前 foreground-only baseline。

### 6.2 background prototype

reference mask 外提取背景 features：

```text
background_proto = mean(features where reference_mask == 0)
```

可以分三种背景：

| 背景类型 | 说明 | 优先级 |
| --- | --- | --- |
| global background | mask 外所有区域 | P0 |
| near-boundary background | mask 边界外扩一圈 | P0 |
| hard background | 与前景相似但不在 mask 内的背景 | P1 |

### 6.3 contrastive score

基础公式：

```text
s_fg(x) = cosine(feature_q(x), foreground_proto)
s_bg(x) = cosine(feature_q(x), background_proto)
score(x) = z(s_fg(x)) - lambda * z(s_bg(x))
```

其中：

```text
z(.) = query 图内标准化
lambda = 背景抑制强度
```

第一轮只扫：

```text
lambda = 0, 0.25, 0.5, 1.0
```

`lambda=0` 就是当前 foreground-only baseline。

### 6.4 为什么这比普通相似度更有设计空间

它利用了 reference mask 的完整信息：

```text
mask 内 = 想要的细胞核前景
mask 外 = 不想要的组织背景
```

因此 Stage1 不只是找“像前景”的点，而是找：

```text
像前景，并且不像背景的点。
```

后续也可以把 `lambda` 学出来，而不是手调。

## 7. Stage2：多点如何进入 SAM

### 7.1 Dense top-k prompt

最直接实验：

```text
取 contrastive score 最高的 K 个点，一次性作为 positive points 输入 SAM。
```

K 取：

```text
1, 3, 5, 10, 20, 50, 100
```

这直接验证你的第一个判断：

```text
MonuSeg 中，多点一起进 SAM 是否比单点/少点更能覆盖全图细胞核？
```

### 7.2 Spatially-diverse top-k

如果 top-k 点扎堆，就加空间去重：

```text
先取 top-M 高分点，再做 distance NMS 或 farthest point sampling，保留 K 个分散点。
```

它的目的不是聚类，而是避免所有点落在同一小块。

### 7.3 Local maxima

在 score heatmap 上找局部峰值：

```text
contrastive score map -> local maxima -> top K peaks
```

它比普通 top-k 更像“每个高响应区域选一个代表点”。

### 7.4 Boundary-aware mix

你提到某些点可能瞄到了细胞核边缘，这些点可能比中心点更能帮助 SAM 确定边界。  
因此可以把点分成：

```text
interior points: score 高、局部稳定
boundary points: score 高，且位于 score gradient 较大处
```

组合：

| K | interior | boundary |
| ---: | ---: | ---: |
| 20 | 15 | 5 |
| 50 | 35 | 15 |
| 100 | 70 | 30 |

这一步把“边缘点可能有用”变成可验证设计。

## 8. 聚类：从默认模块降级为对照模块

聚类只回答一个问题：

```text
在 MonuSeg 中，先把 matched points 分组是否真的比 dense points 更好？
```

对照：

| 方法 | 角色 | 可能结果 |
| --- | --- | --- |
| no clustering | 强基线 | dense points 可能最好 |
| DBSCAN | 空间聚类 | 可能去噪，也可能丢边缘点 |
| heatmap connected components | 热力图区域 | 可能更稳定 |
| KMeans | 指定 K 对照 | 不一定适合 |

评价不看“cluster 看起来是否合理”，而看：

```text
最终 Dice、object recall、merge error 是否改善。
```

如果聚类不好，就明确写：

```text
MonuSeg 的 all-nuclei task 不应强制 instance clustering。
```

## 9. Candidate 数量控制

为了避免 x 个 candidates 变量爆炸，第一版只生成确定性候选。

### Round 1 candidate set

```text
foreground-only top-k
K = 1, 3, 5, 10, 20, 50, 100
```

每个 case 7 个 candidates。

### Round 2 candidate set

```text
contrastive top-k
lambda = 0.25, 0.5, 1.0
K = Round 1 最优的 1-2 个 K
```

每个 case 3-6 个 candidates。

### Round 3 candidate set

```text
top-k
diverse top-k
local maxima
boundary-aware mix
DBSCAN medoid
heatmap CC
```

固定 K 和 lambda 后，每个 case 约 6 个 candidates。

这样每一轮都能解释 candidate 从哪里来，而不是随机组合。

## 10. 指标：不能只看 proposal F1

MonuSeg 是 all-nuclei mask 任务，所以要看：

### 10.1 mask-level

| 指标 | 作用 |
| --- | --- |
| Dice | 全图前景重叠 |
| IoU | 更严格的重叠 |
| foreground precision | 是否过分扩张 |
| foreground recall | 是否漏掉大量细胞核 |

### 10.2 object-level

| 指标 | 作用 |
| --- | --- |
| object recall | GT nuclei 有多少被覆盖 |
| object precision | 预测区域有多少是真 nuclei |
| connected component count error | 预测实例数量偏差 |
| merge error | 多个 nuclei 是否粘成一片 |

重点看：

```text
dense points 是否提升 recall，同时是否造成严重 merge。
```

## 11. 三轮实验设计

### Round 1：多点数量实验

目的：

```text
验证多点 dense prompt 是否优于少点。
```

固定：

```text
matching = foreground-only
layout = top-k
clustering = none
```

变化：

```text
K = 1, 3, 5, 10, 20, 50, 100
```

输出：

| K | mean Dice | mean IoU | object recall | merge error | 结论 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 待跑 | 待跑 | 待跑 | 待跑 | single-point baseline |
| 3 | 待跑 | 待跑 | 待跑 | 待跑 | few points |
| 5 | 待跑 | 待跑 | 待跑 | 待跑 | small dense |
| 10 | 待跑 | 待跑 | 待跑 | 待跑 | medium dense |
| 20 | 待跑 | 待跑 | 待跑 | 待跑 | dense |
| 50 | 待跑 | 待跑 | 待跑 | 待跑 | very dense |
| 100 | 待跑 | 待跑 | 待跑 | 待跑 | use all top-100 |

### Round 2：前景-背景对比匹配

目的：

```text
验证 reference background 是否能提高 matched point 质量。
```

固定：

```text
K = Round 1 最优或前二
layout = top-k
clustering = none
```

变化：

```text
lambda = 0, 0.25, 0.5, 1.0
```

输出：

| lambda | mean Dice | object recall | background false positive | 结论 |
| ---: | ---: | ---: | ---: | --- |
| 0 | 待跑 | 待跑 | 待跑 | foreground-only |
| 0.25 | 待跑 | 待跑 | 待跑 | weak bg suppression |
| 0.5 | 待跑 | 待跑 | 待跑 | medium bg suppression |
| 1.0 | 待跑 | 待跑 | 待跑 | strong bg suppression |

### Round 3：点布局与聚类对照

目的：

```text
验证点如何分布、聚类是否必要。
```

固定：

```text
K = Round 1 最优
lambda = Round 2 最优
```

变化：

| 方法 | clustering | mean Dice | object recall | merge error |
| --- | --- | ---: | ---: | ---: |
| top-k | none | 待跑 | 待跑 | 待跑 |
| diverse top-k | none | 待跑 | 待跑 | 待跑 |
| local maxima | none | 待跑 | 待跑 | 待跑 |
| interior + boundary mix | none | 待跑 | 待跑 | 待跑 |
| cluster medoid | DBSCAN | 待跑 | 待跑 | 待跑 |
| heatmap components | CC | 待跑 | 待跑 | 待跑 |

## 12. 学习、算法和损失函数放在第二阶段

你说“取多少个点和任务本身相关，可以根据训练测试得知”，这个判断很重要。  
但顺序应该是：

```text
先通过 controlled ablation 找到有效变量，再设计学习算法。
```

### 12.1 可学习 matching score

将手工公式：

```text
score = z(s_fg) - lambda * z(s_bg)
```

扩展为：

```text
score(x) =
  w_fg * z(s_fg(x))
  - w_bg * z(s_bg(x))
  + w_peak * local_peak(x)
  + w_edge * edge_score(x)
```

可学习参数：

| 参数 | 含义 |
| --- | --- |
| w_fg | 前景相似度权重 |
| w_bg | 背景抑制权重 |
| w_peak | 局部峰值权重 |
| w_edge | 边缘点权重 |

### 12.2 可学习 point budget

不同任务需要不同点数：

```text
single object -> few points
all nuclei -> dense points
large organ -> sparse + boundary points
```

MonuSeg 中可以先通过验证集选择：

```text
K*
```

后续再让模型根据 query 的 heatmap 分布预测：

```text
K = f(score_map statistics)
```

例如输入：

```text
heatmap entropy
number of local maxima
foreground area estimate
score concentration
```

输出：

```text
K = 10 / 20 / 50 / 100
```

### 12.3 可能的 loss

#### point-level ranking loss

用 query GT 给点打标签：

```text
foreground point > background point + margin
```

loss：

```text
max(0, margin - score_fg + score_bg)
```

#### score map BCE / Dice loss

把 query GT 下采样到 feature map 尺度：

```text
BCE(score_map, downsampled_query_mask)
DiceLoss(score_map, downsampled_query_mask)
```

#### policy-level validation objective

如果 SAM 不可微，就用离线评估：

```text
选择 K、lambda、layout，使验证集 mean Dice + object recall 最大，同时 merge error 不恶化。
```

这可以先用 grid search / Bayesian optimization / bandit，不必一开始端到端训练。

## 13. 论文贡献如何重新表述

更窄但更清楚的论文方向：

```text
Reference-Guided Dense Point Prompting for Nuclei Segmentation
```

或：

```text
Foreground-Background Contrastive Point Selection for SAM-based Nuclei Segmentation
```

贡献点：

1. 从 reference mask 同时提取 foreground 和 background prototypes。
2. 提出适合 MonuSeg 的 dense point prompting，而不是默认少点 prompt。
3. 系统分析 point budget、point layout 和 clustering 对 all-nuclei segmentation 的影响。
4. 证明候选生成质量比复杂 proposal score 更关键。

## 14. 下一步实现任务

### T1：MonuSeg contrastive matching 脚本

建议新增：

```text
code/benchmarks/run_monuseg_contrastive_matching.py
```

输出：

```text
s_fg.npy
s_bg.npy
contrastive_score.npy
topk_points.csv
score_heatmap.png
topk_overlay.png
```

### T2：MonuSeg dense point prompt runner

建议新增：

```text
code/benchmarks/run_monuseg_dense_point_prompts.py
```

输入：

```text
topk_points.csv
K
layout strategy
SAM checkpoint
```

输出：

```text
prompt_points.json
mask.png
overlay.png
metrics.json
```

### T3：Round 1 报告

输出：

```text
runs/research_direction/monuseg_dense_prompt/round1_prompt_budget.csv
docs/reports/monuseg_dense_prompt_round1.html
```

核心图：

```text
K=1 / 5 / 20 / 50 / 100 的同 case 对比图
```

### T4：Round 2 报告

输出：

```text
foreground-only vs foreground-background contrastive matching
```

核心图：

```text
s_fg heatmap
s_bg heatmap
contrastive score heatmap
top-k points overlay
SAM mask result
```

### T5：Round 3 报告

输出：

```text
top-k / diverse top-k / local maxima / boundary-aware / clustering 对比
```

核心图：

```text
同一 query case 中不同点布局进入 SAM 后的 mask 对比
```

## 15. 给老师的简短说法

可以这样讲：

```text
老师，我重新收缩了一下方向。之前想做每个 domain 训练一次再泛化到所有领域，但这个目标太宽，变量也太多。现在我想先在 MonuSeg 一个领域里突破，因为细胞核是密集散落目标，最适合研究 matched points 如何进入 SAM。

我的新假设是：MonuSeg 不一定适合单点或少点 prompt。因为需要分割的细胞核散落在整张图里，如果 top-100 matched points 已经比较准，那么 dense multi-point prompt 可能更适合。聚类也不一定一定好，因为有些点可能落在细胞核边缘，对边界反而有帮助，强行聚类可能会把这些点丢掉。

另外 reference 图带 mask，所以 matching 不应该只看前景相似度，还应该利用背景：也就是找“像 reference foreground 且不像 reference background”的点。下一步我会先做 foreground-background contrastive matching，再比较 K=1、3、5、10、20、50、100 个点一起输入 SAM 的效果，最后再看 spatial diversity、local maxima、boundary-aware 点和聚类是否真的有帮助。

这样这项工作会更聚焦：先证明 MonuSeg 中可控点选择能提高 SAM 候选 mask 质量，再考虑学习点数、学习前景背景权重和迁移到其它领域。
```

