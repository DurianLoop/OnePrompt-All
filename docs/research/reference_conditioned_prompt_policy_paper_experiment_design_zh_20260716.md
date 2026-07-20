# 学习如何从 Reference 生成 Prompt：整体论文思路与公平实验设计

更新日期：2026-07-16  
方法暂名：**RefPrompt Policy**  
英文全称：**Reference-Conditioned Prompt Policy Learning**  
主任务：MoNuSeg dense all-nuclei segmentation  
文档性质：论文方法、实验协议与执行路线统一设计稿  

---

## 0. 核心结论

这篇论文不应被描述为“学习 K、负点数量和 DBSCAN 参数”，而应被统一表述为：

> **给定一张 reference image-mask pair，从 reference-query matching 关系中学习如何为冻结的 SAM 生成与任务意图一致的 Prompt。**

其中：

- `K` 是策略最终选择的正点数量，不是论文问题本身；
- `Y` 是策略最终选择的负点数量，不是单独调出来的超参数；
- 聚类是可选的空间组织专家，不是默认步骤；
- matcher 和 SAM 均冻结；
- 唯一训练对象是轻量 Prompt Policy；
- 训练目标是逼近 train-only prompt oracle，并在质量基本不下降时减少 prompt 和 candidate 成本。

但是，**完整方法不等于所有变量同时实验**。论文采用统一框架、分阶段验证：

```text
完整方法：Positive + Negative + Budget + Layout + Response

实验顺序：
先只开放 Positive Budget
再开放 Negative Selection
再开放 Layout Router
最后测试跨 matcher / segmenter / domain
```

这样既保留完整的方法论文高度，也能保证每个结论都有可解释的对照。

---

## 1. 从外部建议中吸收什么、修正什么

### 1.1 应当完整吸收的部分

1. **学习 Prompt Policy，而不是手工设计 Prompt。**
2. 策略输出应覆盖 positive、negative、prompt 数量和空间组织，而不仅是预测一个 K。
3. matcher 与 SAM 冻结，只训练轻量策略模块。
4. 使用 episodic reference-query training，避免依赖某一张固定 reference。
5. 必须加入 oracle，量化固定策略还剩多少可学习空间。
6. 必须加入不同 matcher 和 SAM-family 模型，检验可插拔性和失败边界。
7. Prompt 存在 domain-level style：密集核场景和单器官场景需要不同的预算与正负点比例。

### 1.2 不能原样写进论文的部分

#### “现有工作基本没有 Negative Prompt”不准确

SAT/GBMSeg、FoB、AlignSAM、Plug-and-Play PPO 等工作已经涉及正负点、背景提示或动态点优化。因此本文不能把“使用负点”本身写成创新。

本文更准确的差异是：

> 在 reference-defined dense all-instance task 中，从显式 foreground/background matching evidence 构造正负候选池，并用 train-only policy 学习预算、选择和冻结 SAM 响应接受规则。

#### “直接用 Dice Loss 穿过 SAM 训练”在离散选点下并不成立

Top-K、正负标签和聚类路由都是离散操作，冻结 SAM 也不提供一个天然可微的点选择路径。主方案不能只写：

```text
Prediction -> Dice Loss -> Prompt Policy
```

而不解释梯度如何到达策略。

本文推荐采用**离线 action utility learning + pairwise ranking**：先在 train-only case 上运行候选 prompt 得到真实分割质量，再训练 policy 预测不同动作的相对收益。SAM 作为冻结黑盒 evaluator，不需要对其反向传播。

强化学习、Gumbel-Softmax 或 straight-through estimator 可以作为后续扩展，但不是第一版主方法的必要条件。

#### “Dataset Prior”需要多域证据

只在 MoNuSeg 一个领域训练时，所谓 dataset prior 只是该领域 policy 的偏置，不能证明跨数据集学习到了可迁移先验。

因此分两层表述：

- 第一阶段：**domain-specific prompt prior**，在 MoNuSeg 内验证；
- 第二阶段：加入 MoNuSAC 和至少一个新的病理核数据集，验证 shared/domain-conditioned policy 后，才讨论 dataset-level prior。

#### “Plug-and-Play / Model-Agnostic”必须通过实验获得

同一代码能接不同 backbone，只能称为 plug-compatible；同一冻结 policy 在不同 matcher/SAM 上仍有效，才能称为 model-agnostic。

---

## 2. 论文要解决的核心问题

### 2.1 传统 Matching + SAM 的固定流程

```text
Reference image-mask
        |
        v
Dense matching
        |
        v
Similarity map
        |
        v
Hand-crafted Top-K / clustering / negatives
        |
        v
Frozen SAM
```

主要局限不是 SAM 分割能力不足，而是 Prompt 生成仍由固定启发式规则决定。

### 2.2 四类决策缺口

#### 缺口一：固定正点预算无法表达任务规模

单个正点通常表达“分割这里的目标”，但不能区分：

- 只分一个实例；
- 分若干实例；
- 分割全图所有同类实例。

MoNuSeg 每张图有数百个核。`K=1` 很可能只提供局部意图，而固定 `K=40` 又可能在简单图上冗余、在复杂图上不足。

#### 缺口二：背景证据没有被系统利用

reference mask 外的区域说明“什么不是目标”。只寻找像前景的位置，会把染色相似的间质、空洞或边缘结构当成正点。

#### 缺口三：空间组织策略固定

Top-K、NMS、local maxima、DBSCAN、KMeans 各自有适用场景。聚类可能提高空间覆盖，也可能删除有价值的边缘点或把相邻核合并。

#### 缺口四：Prompt 风格具有 domain 和 model dependence

- dense nuclei 可能需要较多分散正点；
- 单器官分割可能只需要少量正点和更多边界负点；
- 同一组 joint points 在 SAM ViT-B、SAM2、SAM3 或 medical SAM 上的响应并不相同。

固定规则很难同时处理这些差异。

### 2.3 论文研究问题

> 能否从 reference-query matching evidence 中学习一个轻量 Prompt Policy，自动决定选择哪些正点、哪些负点、使用多少点以及如何组织这些点，并在冻结 matcher 与 SAM 的条件下，以有限调用成本获得更符合 reference-defined task intent 的分割？

---

## 3. 方法总览

### 3.1 输入与输出

输入：

- reference image `R`；
- reference mask `M_R`；
- query image `Q`；
- 冻结 feature matcher `F`；
- 冻结 segmenter `S`；
- 可选 domain token `d` 和 segmenter token `m`。

输出：

- positive point set `P+`；
- negative point set `P-`；
- layout expert `c`；
- candidate/SAM response acceptance decision `g`。

最终预测：

```text
Y_hat = S(Q, P+, P-, c)
```

### 3.2 完整 Pipeline

```text
Reference image + reference mask
                |
                v
     Frozen dense feature matcher
                |
        +-------+-------+
        |               |
        v               v
  FG similarity     BG similarity
        |               |
        +-------+-------+
                |
                v
      Positive / negative pools
                |
                v
       Prompt Policy Network
        +-------+-------+
        |       |       |
        v       v       v
     Budget   Point    Layout
      heads   utility  router
        |       |       |
        +-------+-------+
                |
                v
         Structured Prompt
                |
                v
            Frozen SAM
                |
                v
       Optional response gate
                |
                v
           Final prediction
```

### 3.3 训练和冻结边界

| 模块 | 是否训练 | 作用 |
| --- | --- | --- |
| feature matcher | 冻结 | 产生 reference-query matching evidence |
| SAM-family segmenter | 冻结 | 根据 Prompt 生成 mask |
| candidate pool builder | 固定规则 | 提供可审计、有限的动作空间 |
| Prompt Policy Network | **训练** | 预测预算、点效用和布局 |
| response gate | **训练** | 判断是否接受额外负点/候选响应 |

论文不是微调 SAM，也不是训练一个完整分割网络，而是在两个冻结 foundation components 之间学习结构化 Prompt 决策。

---

## 4. Foreground-Background Candidate Pool

### 4.1 reference foreground/background prototypes

将 reference mask 映射到 feature grid，分别计算：

```text
p_fg = normalize(sum_i w_fg(i) F(R)_i)
p_bg = normalize(sum_i w_bg(i) F(R)_i)
```

query 位置 `x`：

```text
s_fg(x) = cosine(F(Q)_x, p_fg)
s_bg(x) = cosine(F(Q)_x, p_bg)
s_contrast(x) = z(s_fg(x)) - lambda * z(s_bg(x))
```

已有严格证据表明，在固定 `K=40`、DINOv2-L、SAM ViT-B 下：

```text
FG-only Dice = 0.473325
FG-BG Dice   = 0.538050
Delta        = +0.064725
95% CI       = [0.046994, 0.084141]
```

因此 FG/BG evidence 是当前最可靠的机制基础。

### 4.2 positive candidate pool

从 `s_contrast` 中构造最多 `M+=100` 个候选正点。每个点保存：

- 坐标；
- `s_fg`、`s_bg` 和 contrast margin；
- 局部峰值强度；
- 邻域熵；
- 到已选高分点的距离；
- 所属 score-map component；
- reference-query feature gap。

候选池不是最终 Prompt。它只规定 policy 可以从哪些可信位置中选择。

### 4.3 negative candidate pool

负点不简单等于 bottom-Y。候选来源包括：

1. 高背景相似度位置；
2. 前景相似度不低但 contrast margin 为负的 hard background；
3. 正点附近的背景边界；
4. SAM anchor mask 外但 matcher 易混淆的位置。

每个候选负点同样保存背景置信度、与正点距离和局部结构特征。主候选池上限固定为 `M-=100`，避免动作空间无限扩张。

### 4.4 为什么先构造 pool，再学习选择

直接在百万像素空间学习任意坐标对当前 37 个独立训练病例过于困难。有限候选池具有三个优势：

- 保留 matcher 的可解释性；
- 降低训练样本需求；
- 可以计算 train-only oracle，明确策略上限和 selector regret。

---

## 5. Prompt Policy Network

### 5.1 核心不是预测一个 K

Policy 学习的是一个结构化动作：

```text
a = (selected positives,
     selected negatives,
     positive budget,
     negative budget,
     layout expert,
     response acceptance)
```

最后出现 `17 positive + 4 negative + no clustering`，只是策略对某个 query 的输出结果。

### 5.2 Context Encoder

Context Encoder 读取不含 query GT 的部署可用信息：

- FG/BG/contrast maps 的均值、方差、熵和峰值数量；
- top-score decay curve；
- candidate spatial coverage、密度和最近邻距离；
- reference mask 面积与 reference instance count；
- query 图像颜色和纹理统计；
- optional domain/matcher/segmenter token。

得到上下文表示：

```text
h = Encoder(context)
```

为支持不同分辨率和候选数量，地图统计使用归一化数值，point candidates 使用 set encoder 或共享 MLP + pooling，而不是依赖固定像素索引。

### 5.3 Positive Policy Head

对每个正点候选预测保留效用：

```text
u_i+ = PositiveHead(h, point_i+)
```

同时预测正点预算分布：

```text
pi_K+ = BudgetHead+(h)
K+ in {1, 2, 4, 8, 16, 32, 40, 64, 100}
```

最终从候选池中选择预算允许的高效用点。第一阶段实验暂时固定候选排序，只学习 `K+`；成功后再开放 point-wise reranking，以免同时改变点数和位置。

### 5.4 Negative Policy Head

```text
u_j- = NegativeHead(h, point_j-)
pi_K- = BudgetHead-(h)
K- in {0, 5, 10, 20}
```

`K-=0` 是合法且重要的动作。策略不是被迫使用负点，而是学习当前 query 是否需要背景约束。

### 5.5 Layout Router

布局不直接预测连续 DBSCAN 参数，而是先在少量固定、validation-calibrated 专家之间路由：

```text
C = {
  top-k / no-clustering,
  diverse top-k,
  local maxima,
  heatmap components,
  DBSCAN medoid
}
```

```text
pi_c = Router(h)
```

这种 mixture-of-prompt-experts 设计比同时学习任意聚类算法、`eps` 和 `min_samples` 更稳定，也更容易公平消融。

### 5.6 Domain Prior 与 query residual

完整版本采用分层策略：

```text
policy logits = global prior
              + domain prior(d)
              + query residual(h)
              + segmenter calibration(m)
```

直观上：

- domain prior 学习 MoNuSeg 通常偏好多正点、少负点；
- query residual 处理当前图更密、更稀或更易混淆的差异；
- segmenter calibration 处理同一 Prompt 在不同 SAM 上响应不同的问题。

在只有 MoNuSeg 时，只验证 domain-specific policy；加入至少三个具有不同目标结构的数据集后，才验证 shared/domain-conditioned prior。

### 5.7 Response Gate

pre-SAM policy 无法完全预测冻结 SAM 如何组合多个点。因此允许在固定调用预算下运行 anchor 和 challenger：

```text
anchor     = validation-safe prompt
challenger = policy proposal
```

post-SAM gate 只读取无 GT 响应：

- SAM internal quality score；
- predicted area change；
- component count change；
- matcher-mask consistency；
- prompt coverage change。

当预测相对收益超过 validation threshold 时接受 challenger，否则回退 anchor。

当前 `+0.010097 Dice` 是该 response gate 在固定 K40 上的边际收益，不代表完整 Prompt Policy 的总收益。

---

## 6. 训练方式

### 6.1 Episodic training

每个 episode 包括：

```text
reference image-mask + query image-mask + frozen matcher + frozen SAM
```

reference 只能来自当前训练折，并且不能与 query 相同。训练期间轮换 reference，使 policy 学到 reference-query 关系，而不是记住某一张 support。

实际本地数据不是假设中的 100 张，而是：

```text
37 source-train + 14 source-test = 51 case
```

因此不能直接使用假设的 `70/10/20`。当前主协议使用 37 source-train 的 5-fold nested-CV，并按 query 聚合多 reference episodes。

### 6.2 Offline action bank

对每个 train/validation episode，运行预注册动作集合：

```text
K+ x K- x layout
```

保存：

- Prompt 坐标和标签；
- SAM mask；
- Dice、AJI+、PQ 等 train-only outcome；
- SAM response features；
- 运行时间和调用数。

test GT 不参与 action label、模型训练、reference 选择或阈值选择。

### 6.3 Train-only oracle

每个训练 episode 的 oracle action：

```text
a* = argmax_a quality(Y_hat_a, Y_gt)
```

oracle 不是可部署方法，而是回答：

1. 当前 candidate pool 的上限有多高；
2. 固定策略还有多少可学习空间；
3. policy 的误差来自 candidate generation 还是 action selection。

### 6.4 推荐主训练目标

对每个动作预测相对 validation anchor 的收益：

```text
Delta(a) = Dice(a) - Dice(anchor)
```

主要损失：

```text
L_reg  = Huber(predicted_gain(a), true_gain(a))
L_rank = pairwise logistic loss(a_i, a_j)
L      = L_reg + beta * L_rank
```

其中 pairwise target 由同一 episode 内两个动作的真实 train-only Dice 顺序产生。这样不会把不同病例的绝对难度误当成动作优劣。

### 6.5 Prompt efficiency 不使用玄学加权分数

不建议直接写一个难解释的：

```text
Dice + alpha * recall - beta * merge - gamma * K
```

主决策采用字典序约束：

1. 先预测质量；
2. 选择预测质量距离最佳不超过 `epsilon` 的动作；
3. 在这些动作中选择 prompt 数更少、SAM calls 更少的动作。

```text
a_hat = minimum-cost action
        subject to predicted_quality(a)
        >= max predicted_quality - epsilon
```

`epsilon` 只能在 inner-validation 冻结。

### 6.6 分阶段训练

| 阶段 | 开放变量 | 固定变量 | 目的 |
| --- | --- | --- | --- |
| T1 | K+ | matcher、lambda、point order、K-=0、top-k、SAM | 证明 adaptive positive budget |
| T2 | K+、point reranking | K-=0、top-k | 证明学习点位置超过 matcher 原排序 |
| T3 | K+、K-、正负点效用 | layout=top-k | 证明 adaptive negative selection |
| T4 | positive、negative、layout router | matcher/SAM | 证明完整 policy |
| T5 | domain/model tokens | policy architecture | 证明 shared/domain/model-aware 泛化 |

每阶段使用上一阶段的冻结结果作为强 baseline，而不是改变多个变量后只看最终均值。

---

## 7. 数据与防泄漏协议

### 7.1 37/14 与 30/14 必须分开命名

本地官方页面下载包观察到：

```text
37 source-train + 14 source-test
```

经典 challenge 常见口径是：

```text
30 train + 14 test
```

正式写作前必须审计额外 7 例并冻结：

```text
runs/research_direction/dataset_audit/monuseg_challenge30_ids.csv
runs/research_direction/dataset_audit/monuseg_extra7_audit.csv
```

在审计完成前：

- `37-train nested-CV` 称为 source-package protocol；
- `30/14` 才称为 classic challenge protocol；
- 不能把二者混写成同一个“官方完整协议”。

### 7.2 主统计：37-query 5-fold nested-CV

每折：

- outer-test：7 或 8 query cases；
- inner-validation：6 cases；
- inner-train：23 或 24 cases；
- 每个 query 使用 5 个当前折 train-only references；
- 五折后 37 个 case 各 outer-test 一次；
- 共 185 outer-test reference-query episodes。

统计时先对同一 query 的 5 个 references 聚合，再以 37 个 query 为独立统计单位。不能把 185 episodes 当作 185 个独立病例。

### 7.3 官方 14-case test 的诚实定位

过去已经在 14-case source-test 上进行过多轮 K/lambda/backbone screening，因此旧结果属于 exploratory evidence。

新方法必须先在 nested-CV 冻结，再运行新的 confirmatory challenge-set evaluation。论文主显著性结论仍以 nested-CV 为主，并增加真正未触碰的外部测试集。

### 7.4 reference 规则

- support 必须来自当前 train role；
- support 和 query 不得相同；
- 不按 query GT 选择 reference；
- main setting 是 one reference per inference；
- 5-reference 用于鲁棒性评估，先按 query 聚合；
- reference oracle 仅为 upper bound；
- multi-reference mask ensemble 必须单独计算成本。

---

## 8. Baseline 设计

单点是必要下限，但不能成为唯一 baseline。完整 baseline 分五组。

### 8.1 Group A：无 reference 的 SAM 下限

| ID | 方法 | Prompt | 作用 |
| --- | --- | --- | --- |
| A0 | center single point | 1 正点 | 确定性弱下限 |
| A1 | random single point | 1 正点，5 seeds | 随机下限 |
| A2 | fixed four-point grid | 4 正点 | 多点但无 reference intent |
| A3 | dense grid | 固定网格 | 验证“点多本身”是否足够 |
| A4 | box/full-image box | 1 box | SAM/MedSAM 接口参考，单独分栏 |

这些方法回答“普通 Prompt 能否表达任务”，不能单独支撑 learned policy 优越性。

### 8.2 Group B：固定 Matching + SAM 强 baseline

| ID | 方法 | 说明 |
| --- | --- | --- |
| B0 | FG top-1 | reference matcher 单点 |
| B1 | FG-BG top-1 | reference foreground/background 单点 |
| B2 | fixed FG top-K | validation 选择全局 K |
| B3 | fixed FG-BG top-K | validation 选择全局 K/lambda，最重要固定强基线 |
| B4 | fixed FG-BG + fixed negatives | 固定 K+/K- |
| B5 | fixed FG-BG + fixed layout | 固定 top-k/local-max/cluster |

### 8.3 Group C：学习型对照

| ID | 方法 | 学习内容 |
| --- | --- | --- |
| C0 | majority action | 所有 query 使用训练集多数/均值最优动作 |
| C1 | heuristic adaptive K | 峰数量/覆盖率手写规则 |
| C2 | ExtraTrees/GBDT utility | 传统轻量 action regressor |
| C3 | positive budget policy | 只学习 K+ |
| C4 | point utility policy | 学习 K+ 和正点 reranking |
| C5 | positive-negative policy | 学习正负预算与选择 |
| C6 | full RefPrompt Policy | 正负点、预算、layout、response gate |

### 8.4 Group D：公开 reference/prompt 方法

优先加入：

1. Matcher；
2. GBMSeg-style adapted；
3. FoB-SAM 同折 MoNuSeg 适配；
4. AlignSAM / Plug-and-Play PPO，在代码和调用预算能公平对齐时加入；
5. SAT 只有完整官方代码可运行后才填本地数值；
6. One-Prompt、UniverSeg 等作为不同训练范式参考。

adapted baseline 必须披露 checkpoint、输入接口、支持集、改动和训练成本，不能冒充 official reproduction。

### 8.5 Group E：Oracle 与监督上限

| 方法 | 地位 |
| --- | --- |
| per-query prompt oracle | 使用 query GT 事后选动作，不可部署 |
| shortlist oracle | 检查 policy shortlist 是否包含好动作 |
| nnU-Net / supervised nuclei model | fully supervised upper bound，不是同设置竞争者 |

---

## 9. 公平比较原则

### 9.1 每增加一个组件，是否都要看效果？

是，但必须同时做两种实验。

#### 累积增量

```text
M0: reference top-1
M1: fixed multi-positive
M2: + FG/BG evidence
M3: + adaptive positive policy
M4: + adaptive negative policy
M5: + layout router
M6: + response gate
```

每次只开放一个新自由度，并与直接前驱做 query-level paired comparison。

#### 最终方法移除消融

```text
Full
w/o FG/BG
w/o adaptive positive
w/o adaptive negative
w/o layout router
w/o response gate
w/o domain token
w/o segmenter token
```

累积实验回答“加上它是否改善”，移除消融回答“最终方法是否依赖它”。两者不能互相替代。

### 9.2 强 baseline 不能移动

所有学习组件的核心对手固定为：

```text
validation-best fixed FG-BG prompt policy
```

不能在不同实验中分别挑一个更弱的固定 K，再宣称每个模块都提升。

### 9.3 三类成本必须报告

| 成本 | 定义 |
| --- | --- |
| prompt budget | 正点和负点数量 |
| candidate budget | 为一个 query 生成并评估多少套 Prompt |
| SAM call budget | 实际进行多少次 mask decoding |

`40 points joint prompt` 通常只调用一次 SAM；`40 points independent union` 调用 40 次。两者必须放入质量-成本曲线，不能只比较 Dice。

### 9.4 两张公平主表

#### 主表 A：固定一次 SAM call

比较：

- fixed K；
- heuristic adaptive K；
- learned adaptive K；
- learned positive point reranking。

回答：不增加 SAM 调用，policy 是否能生成更合适的 Prompt。

#### 主表 B：固定候选预算或质量-成本 Pareto

分别报告 1、2、3 次 SAM calls。response gate 与 fixed two-candidate baseline 使用相同调用数。

### 9.5 固定工程口径

以下内容必须写入 metadata：

- image resize/crop；
- matcher layer、patch size 和 checkpoint hash；
- reference mask 下采样；
- coordinate transform；
- joint/per-point execution；
- SAM multimask 选择规则；
- 后处理；
- precision/device；
- policy checkpoint、seed 和 fold。

---

## 10. 论文实验设计

### Experiment 1：单点为何不够，多点收益到底有多大

同一 nested-CV、同一 reference、同一 matcher/SAM 下比较：

```text
K+ in {1, 2, 4, 8, 16, 32, 40, 64, 100}
K- = 0
layout = top-k
```

输出：

- Dice/AJI+/PQ 曲线；
- object coverage 和 merge error；
- point efficiency；
- per-query oracle K；
- validation-best fixed K；
- fixed-to-oracle gap。

这是整篇论文的地基。没有同协议 `K=1 -> multi-point`，不能声称 policy 学会了 all-instance intent。

### Experiment 2：学习 Positive Prompt Policy

比较：

```text
reference top-1
validation-best fixed K
heuristic adaptive K
ExtraTrees/GBDT K policy
proposed positive budget policy
proposed budget + point reranking
oracle positive policy
```

关键判断：

1. learned K 是否超过 fixed K；
2. point reranking 是否超过 matcher 原始排序；
3. 是否在相同一次 SAM call 下减少平均点数；
4. 是否缩小 fixed-to-oracle gap。

### Experiment 3：学习 Negative Prompt Policy

固定 Experiment 2 的 positive policy，比较：

```text
no negative
fixed bottom-5
fixed hard-negative-5
validation-best fixed K-
adaptive K-
adaptive negative point selection
adaptive negative + response gate
negative oracle
```

重点不是证明“负点总是有用”，而是证明 policy 能判断何时 `K-=0`、何时 `K->0`，并减少错误负点造成的性能下降。

### Experiment 4：学习 Layout Router

固定正负点候选池和预算，对比：

```text
no clustering / top-k
diverse top-k
local maxima
heatmap components
DBSCAN medoid
learned layout router
layout oracle
```

同时报告每个 expert 被选频率和适用病例。若 learned router 无法超过 validation-best fixed expert，则不把 clustering 放进最终主方法。

### Experiment 5：Dataset/Domain Prompt Prior

仅在至少三个 domain 可用后执行。建议包含：

- MoNuSeg：密集 all-nuclei；
- MoNuSAC：病理核、多患者；
- 一个新病理核数据集，如 CoNSeP/TNBC/PanNuke 的可比子任务；
- 若要证明跨任务 prompt style，再加入单器官或单病灶数据集。

比较：

```text
per-domain policy
shared policy without domain token
shared policy with domain token
leave-one-domain-out transfer
zero-shot shared policy
```

只有 domain token/shared policy 明确改善多个领域后，才写“学习 dataset-level prompt prior”。

### Experiment 6：Plug-and-Play 与 model dependence

#### Matcher 轴

- DINOv2-B/L；
- BioMedCLIP；
- pathology-specific backbone，如 UNI/CONCH，在权重许可时；
- SAM image encoder feature baseline。

#### Segmenter 轴

- SAM ViT-B/H；
- SAM2.1 Large；
- SAM-Med2D；
- Medical-SAM-Adapter；
- SAM3 point API，在完成严格接口审计后。

需要区分：

1. **model-specific retraining**：同结构 policy 针对每个组合训练，证明 plug-compatible；
2. **shared policy transfer**：同一个冻结 policy 跨模型，才检验 model-agnostic；
3. **model-token policy**：一个策略显式条件于 matcher/segmenter identity。

### Experiment 7：Reference 鲁棒性

比较：

- fixed single reference；
- random train-only reference；
- density-stratified references；
- 1/3/5 references 独立评估；
- reference ensemble，单独计算调用成本；
- worst-reference 与 oracle-reference upper bound。

报告 reference 方差，而不是只报告均值。

### Experiment 8：效率与安全性

报告：

- average positive/negative points；
- candidates and SAM calls；
- matcher/policy/SAM/end-to-end latency；
- peak VRAM 和 trainable parameters；
- predicted-gain calibration；
- safety gate 的 switch precision；
- risk-coverage curve；
- 失败切换和回退比例。

---

## 11. Ablation 矩阵

### 11.1 核心模块消融

| Variant | FG/BG | Positive policy | Negative policy | Layout router | Response gate |
| --- | --- | --- | --- | --- | --- |
| fixed matcher | yes | no | no | no | no |
| + adaptive positive | yes | yes | no | no | no |
| + adaptive negative | yes | yes | yes | no | no |
| + layout router | yes | yes | yes | yes | no |
| Full | yes | yes | yes | yes | yes |
| Full w/o BG | no | yes | yes | yes | yes |
| Full w/o safety anchor | yes | yes | yes | yes | unanchored |

### 11.2 训练目标消融

- regression only；
- ranking only；
- regression + ranking；
- absolute Dice target vs anchor-relative gain；
- without minimum-cost constraint；
- different validation-frozen `epsilon`。

### 11.3 Candidate pool 消融

- `M+={20,50,100}`；
- `M-={20,50,100}`；
- only global background；
- + hard background；
- + near-boundary background；
- matcher original ranking vs learned reranking。

### 11.4 Policy condition 消融

- no reference statistics；
- no score-map context；
- no spatial features；
- no domain token；
- no model token；
- 1/3/5 training references per query。

---

## 12. 指标与统计

### 12.1 主分割指标

- binary Dice；
- AJI/AJI+；
- Panoptic Quality；
- object-level F1。

正式报告 AJI/PQ 前，必须从 MoNuSeg XML 生成 instance-ID masks。相邻核接触时，binary connected components 不能替代真实实例标注。

### 12.2 Prompt 机制指标

- positive precision@K；
- negative precision@K；
- instance coverage@K；
- selected K+/K- distribution；
- merge/split error；
- predicted foreground area；
- fixed-to-oracle gap；
- shortlist oracle 和 selector regret；
- reference sensitivity。

### 12.3 统计规则

- query image 是独立统计单位；
- 同一 query 的多个 references 先聚合；
- paired bootstrap 10,000 次，报告 95% CI；
- wins/ties/losses；
- exact sign test 或 Wilcoxon signed-rank；
- 多个预注册主比较使用 Holm correction；
- 随机 policy 报告 5 seeds 或证明确定性；
- 不把 candidate、point、patch 或 reference episode 当独立病例扩大样本量。

---

## 13. 论文主表设计

### Table 1：与强 baseline 公平比较

| Method | Ref | Learned policy | Avg K+ | Avg K- | SAM calls | Dice | AJI+ | PQ | Runtime |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SAM single point | no | no | 1 | 0 | 1 | E1 | E1 | E1 | E8 |
| Matcher FG top-K | yes | no | val-fixed | 0 | 1 | E1 | E1 | E1 | E8 |
| Fixed FG-BG | yes | no | val-fixed | val-fixed | 1 | E1/E3 | E1/E3 | E1/E3 | E8 |
| GBMSeg-style | yes | no | method | method | method | E5 | E5 | E5 | E8 |
| FoB adapted | yes | yes | method | method | method | E5 | E5 | E5 | E8 |
| Positive Policy | yes | yes | adaptive | 0 | 1 | E2 | E2 | E2 | E8 |
| Full RefPrompt Policy | yes | yes | adaptive | adaptive | 2 | E3/E4 | E3/E4 | E3/E4 | E8 |
| Prompt oracle | yes | test GT | adaptive | adaptive | all | upper | upper | upper | grid cost |

### Table 2：累积组件增益

| Method | Change | Dice | Delta vs predecessor | 95% CI | Avg K+/K- |
| --- | --- | ---: | ---: | --- | --- |
| M0 | reference top-1 | E1 | - | - | 1/0 |
| M1 | fixed multi-positive | E1 | E1 paired | E1 | fixed/0 |
| M2 | + FG/BG evidence | 0.538050* | +0.064725* | [0.046994,0.084141]* | 40/0* |
| M3 | + positive policy | E2 | E2 paired | E2 | adaptive/0 |
| M4 | + negative policy | E3 | E3 paired | E3 | adaptive/adaptive |
| M5 | + layout router | E4 | E4 paired | E4 | adaptive/adaptive |
| M6 | + response gate | E3/E8 | paired | bootstrap | adaptive/adaptive |

`*` 是当前 K40 nested-CV 证据；完整组件表必须在统一新 action bank 上重算。

### Table 3：Policy 与 Oracle gap

| Policy | SAM calls | Dice | Oracle gap | Prompt cost | Selector regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed validation-best | 1 | E1 | E1 | E1 | - |
| heuristic adaptive | 1 | E2 | E2 | E2 | E2 |
| learned positive | 1 | E2 | E2 | E2 | E2 |
| full learned | 2 | E3/E4 | E3/E4 | E8 | E3/E4 |
| oracle | all | upper | 0 | all | 0 |

### Table 4：跨 matcher / segmenter

| Matcher | Segmenter | Fixed policy | Model-specific policy | Shared policy | Delta | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| DINOv2-L | SAM ViT-B | E6 | E6 | E6 | E6 | 主组合 |
| DINOv2-L | SAM2.1 Large | E6 | E6 | E6 | E6 | segmenter transfer |
| BioMedCLIP | SAM ViT-B | E6 | E6 | E6 | E6 | matcher transfer |
| pathology backbone | medical SAM | E6 | E6 | E6 | E6 | domain-specific combo |

---

## 14. 需要展示的图

### Figure 1：方法总览

reference image/mask、FG/BG maps、候选正负点、Policy heads、Structured Prompt、Frozen SAM、Response Gate。

### Figure 2：为什么单点不够

同一 query 分开显示：

- K=1；
- validation-best fixed K；
- learned K；
- GT；
- FP/FN/TP error map。

### Figure 3：Prompt Policy 可解释输出

每例显示：

- positive/negative candidate pools；
- policy 保留/拒绝概率；
- 最终 K+/K-；
- layout expert；
- Dice/AJI+/PQ。

### Figure 4：质量-成本曲线

- K vs Dice/PQ；
- SAM calls vs Dice；
- runtime vs Dice；
- fixed-to-oracle gap。

### Figure 5：Domain/Model Prompt Style

不同数据集和 segmenter 的 K+、K-、layout 选择分布。只有多域实验完成后才进入正文。

所有定性图必须把 prediction、GT 和 error map 分开，不能只用高透明度颜色叠加导致看不清。

---

## 15. 现有研究成果如何放入新框架

### 15.1 已经完成且可以复用

- [x] 本地数据核验：37 source-train + 14 source-test；
- [x] 37-query 5-fold nested-CV；
- [x] 每个 query 使用 5 个 train-only references；
- [x] DINOv2-L FG/BG matching；
- [x] FG-only K40 `0.473325` -> FG-BG K40 `0.538050`；
- [x] fixed `K-=N={0,5,10,20}` action bank；
- [x] K40 negative-response gate `0.537458 -> 0.547555`；
- [x] 当前 gate 关闭 fixed-to-oracle gap 的 66.1%；
- [x] MoNuSAC patient-disjoint 外部验证；
- [x] SAM-Med2D 同协议 gate 负结果；
- [x] 早期 15-action K/lambda policy 的失败审计；
- [x] Matcher、SAT/GBMSeg、FoB、PPO 相关工作与复现边界审计；
- [x] CPU/GPU 同 Prompt 数值一致性审计。

### 15.2 这些结果现在分别证明什么

| 已有结果 | 可以证明 | 不能证明 |
| --- | --- | --- |
| FG-BG +0.064725 | reference background evidence 有效 | policy 学会了多点 |
| negative gate +0.010097 | 固定 K40 后负点响应存在可学习边际收益 | 完整 Prompt Policy 总增益 |
| MoNuSAC zero-shot 正向 | 部分响应 gate 可跨域纠错 | shared policy 跨域最优 |
| SAM-Med2D 非正向 | policy 依赖 segmenter response | model-agnostic |
| V7/V8 失败 | 小样本下联合动作学习不稳定 | Prompt Policy 思路无效 |

### 15.3 目前真正缺失的核心证据

- [ ] strict nested-CV reference top-1；
- [ ] `K+={1,2,4,8,16,32,40,64,100}` 完整同协议 action bank；
- [ ] fixed K 到 per-query positive oracle 的 gap；
- [ ] Positive Budget Policy；
- [ ] point-wise positive reranking；
- [ ] adaptive negative point selection，而不只是 N0/N5 response gate；
- [ ] layout router；
- [ ] instance-ID mask、AJI+、PQ；
- [ ] classic challenge 30-case 和 extra-7 审计；
- [ ] GBMSeg-style/FoB 同折强 baseline；
- [ ] shared/domain/model-conditioned policy；
- [ ] 新的未触碰外部数据集。

---

## 16. 执行路线 P0-P6

### P0：冻结协议与实例标注

1. 审计 challenge30/extra7；
2. 冻结 folds、references、seeds；
3. 从 XML 生成 instance-ID masks；
4. 固定 DINOv2-L、SAM ViT-B、lambda=0.75、top-k、K-=0；
5. 固定图像、坐标、multimask 和后处理口径。

### P1：补齐单点到多点的真实证据

1. reference top-1；
2. full positive K curve；
3. validation-best fixed K；
4. per-query positive oracle；
5. Dice/AJI+/PQ 与成本曲线。

Go 条件：`K=1 -> fixed multi-point` 有明确收益，并且 fixed-to-oracle gap 足够大。

### P2：Positive Prompt Policy

1. 只学习 K+；
2. 比较 heuristic、ExtraTrees/GBDT、ranker；
3. 再开放 point reranking；
4. outer-test paired CI；
5. selected K、oracle K 和 selector regret。

Go 条件：相对 validation-best fixed K 提升，或在质量不降条件下显著减少点数。

### P3：Negative Prompt Policy

1. 生成 global/hard/boundary negative pools；
2. 学习 K- 和 point utility；
3. 接入已有 response gate；
4. 与同调用预算 fixed negative baseline 比较。

### P4：Layout Router

1. 冻结少量 layout experts；
2. 训练 router；
3. 比较 fixed best expert 和 layout oracle；
4. 无增益则不进入最终模型。

### P5：强 baseline 与模型轴

1. Matcher/GBMSeg-style；
2. FoB 同折适配；
3. DINOv2/BioMedCLIP/pathology backbone；
4. SAM/SAM2/medical SAM；
5. model-specific 与 shared policy 分开报告。

### P6：Domain Prior 与论文冻结

1. 加入至少一个新病理核数据集；
2. shared/domain-token/leave-one-domain-out；
3. 官方 14-case confirmatory evaluation；
4. 主表、消融、统计、定性和效率图；
5. 冻结英文标题、摘要和贡献。

---

## 17. 论文贡献应该怎样写

只有对应实验成立后，贡献才可以写成下面三点。

### Contribution 1：Reference-conditioned structured Prompt Policy

提出一个在冻结 matcher 和 SAM 之间学习结构化 Prompt 决策的框架，从 reference foreground/background evidence 中联合建模正点、负点、预算和空间组织，而不是依赖固定 Top-K 启发式规则。

不能写“首个 reference prompt policy”，除非完成最新相关工作检索并能严格证明。

### Contribution 2：可审计的 action utility learning

提出 train-only、anchor-relative 的 Prompt action utility/ranking 学习与有限候选 safety gate，在不反向传播穿过 SAM 的条件下，以固定调用预算逼近 per-query prompt oracle，并分解 candidate regret 和 selector regret。

### Contribution 3：Dense all-instance 机制与泛化边界

在密集细胞核任务中系统分析 foreground/background evidence、positive/negative budget、layout、reference 和 segmenter response，报告 query-level nested-CV、instance metrics、效率和失败边界；只有多域/多模型结果成立后，再加入 domain prior 和 plug-and-play 泛化表述。

---

## 18. 最终论文故事

完整故事不再是：

> 我们找到一个比 K=40 更好的 K，或者给 K40 加 5 个负点。

而是：

> 点提示本身不能表达 dense all-instance task intent。我们利用一张带 mask 的 reference 同时定义目标和背景，从冻结 matcher 产生可解释的正负候选池，再训练一个轻量结构化 Prompt Policy，为每个 query 决定点的数量、符号、位置和空间组织，并根据冻结 SAM 的响应安全地接受或回退。通过严格的 query-level nested-CV、固定调用预算、prompt oracle 和跨模型诊断，我们区分了“reference matching 带来的总收益”与“Prompt Policy 相对强固定策略的额外收益”。

一句话英文定位：

> **We learn how to prompt a frozen segmenter from reference-target matching evidence, rather than hand-crafting a fixed prompting rule.**

---

## 19. 论文能否成立的三道门槛

### 门槛一：candidate pool 有空间

strict `K=1 -> fixed multi-point` 必须明显改善，且 per-query oracle 高于 validation-best fixed policy。

### 门槛二：policy 真正学到动作

Positive Policy 必须在相同一次 SAM call 下超过 fixed K，或者在相近质量下显著减少平均 Prompt；不能只靠运行更多 candidates 获得提升。

### 门槛三：完整框架不是堆模块

Negative、layout 和 response 组件都必须通过同协议增量与移除消融。失败组件不强行进入最终方法；跨模型或跨域失败必须用于限定 plug-and-play/model-agnostic 主张。

如果门槛一成立、门槛二失败，论文应收缩为 foreground-background reference matcher 机制论文。如果门槛二成立，Prompt Policy 方法论文主线成立。如果门槛三和多域泛化进一步成立，才有资格采用更强的通用可插拔框架定位。
