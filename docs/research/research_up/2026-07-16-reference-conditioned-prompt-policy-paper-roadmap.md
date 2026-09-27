# Adaptive Prompt Policy for Matching-Based Foundation Segmentation

> **文档性质：** 从零开始的方法与实验设计稿。它不继承或引用此前实验结果，也不预设任何性能数字。
>
> **论文主张（待实验检验）：** Prompt 不应由固定 Top-K、固定负点数或固定聚类规则手工指定；在 reference-guided matching 场景中，应学习一个与图像、reference 和分割器响应条件相关的 Prompt Policy。

## 1. 研究问题与目标

给定一张带标注的 reference image-mask pair `(R, M_R)` 和一张未标注 query image `Q`，我们希望用冻结的特征 matcher 和冻结的分割基础模型生成 prompt，并分割 query 中由 reference 定义的同类目标。

传统匹配式提示流程通常是：

```text
reference mask → dense matching → score map → fixed Top-K / fixed clustering → SAM
```

其核心限制不是 SAM 的主干能力，而是 prompt 决策由人手规则决定：

- 正点数 `K` 被固定，不能适应目标数量、密度与匹配置信度；
- 负点通常缺失或由固定 bottom-K 规则生成，无法区分真正有用的 hard background；
- Top-K、NMS、KMeans、DBSCAN 与 MeanShift 的空间组织策略被预先指定；
- 同一组 prompt 在不同 matcher、不同 segmenter 与不同数据集上可能具有不同效果。

本文要检验的问题是：

> 能否从 reference-query matching evidence 中学习一个轻量 Prompt Policy，联合决定哪些候选点作为正点、哪些作为负点、何时停止增加点，以及采用何种空间布局，使冻结的分割器得到更符合 reference 定义任务的结果？

## 2. 方法总览

### 2.1 统一流程

```text
reference image R + reference mask M_R          query image Q
                 │                                     │
                 └──── frozen feature matcher F ───────┘
                                       │
                         foreground / background matching maps
                                       │
                         positive and negative candidate pools
                                       │
                         Adaptive Prompt Policy πθ
                      ┌────────────────┼────────────────┐
                      │                │                │
              positive selection  negative selection  layout routing
                      │                │                │
                      └────────────────┴────────────────┘
                                       │
                         structured point prompt P(Q, R)
                                       │
                         frozen segmenter S (SAM family)
                                       │
                                 predicted mask Ŷ
```

`F` 与 `S` 均冻结；唯一训练对象是轻量策略 `πθ`。因此本文不是微调 SAM，也不是训练新的端到端分割网络，而是在两个冻结基础组件之间学习可审计的 prompt 决策。

### 2.2 Foreground-background matching evidence

reference mask 定义前景与背景区域。将其投影到 feature grid 后，得到前景和背景 prototype：

```text
p_fg = normalize(weighted_mean(F(R) inside M_R))
p_bg = normalize(weighted_mean(F(R) outside M_R))
```

对于 query patch `x`，计算：

```text
s_fg(x) = cosine(F(Q, x), p_fg)
s_bg(x) = cosine(F(Q, x), p_bg)
s_ctr(x) = normalize(s_fg(x)) - λ · normalize(s_bg(x))
```

其中 `s_ctr` 是候选构造的证据，不是最终决策。它通过 reference 同时回答“像什么”和“不像什么”。

### 2.3 有限、可审计的候选动作空间

为避免策略在连续图像空间中任意点选取而不可训练、不可复现，先由确定性模块生成有限候选池。

| 候选池 | 构造原则 | 每个候选保存的特征 |
| --- | --- | --- |
| 正点池 `C+` | `s_ctr` 高分点、局部峰值、空间去重后的并集 | 坐标、`s_fg`、`s_bg`、contrast margin、局部峰强度、邻域密度、与已选点距离 |
| 负点池 `C-` | 高背景相似、负 contrast、正点邻域边界、初始预测外的易混淆区域 | 坐标、背景置信度、边界距离、正点距离、局部纹理与匹配差 |
| 布局动作 `C_layout` | `{raw-topk, local-maxima, KMeans, DBSCAN, MeanShift}` | 点集分散度、簇数、簇大小、覆盖面积、运行代价 |

候选池上限 `M+`、`M-` 和每种布局的合法参数范围在训练前冻结，并写入实验配置。任何方法都只能在同一候选空间内比较。

### 2.4 Adaptive Prompt Policy

策略由一个共享 episode encoder 和三个决策头组成：

```text
z = Encoder(global matching statistics, candidate features, reference-query gap,
            optional matcher token, optional segmenter token)

positive head:  u+_i = f+(z, c+_i), i ∈ C+
negative head:  u-_j = f-(z, c-_j), j ∈ C-
layout head:    p(l | z), l ∈ C_layout
stop/budget:    p(stop | z, selected prompts)
```

推理时：

1. 按正点 utility 依次选择候选；当 stop probability 超过冻结阈值或达到预算上限时停止；
2. 同样选择少量负点，允许输出零个负点；
3. 选择一个布局 expert；若 layout 不适用或产生非法点集，则使用同一动作空间中预先规定的回退规则；
4. 将最终正负点和标签输入冻结 segmenter。

**第一版约束：** 不学习 DBSCAN 的连续参数，也不学习任意坐标偏移。布局只作为有限离散动作选择。这能把“是否学习空间组织”与“连续聚类调参”分开验证，避免动作空间无界增长。

### 2.5 训练：离线 action-utility learning

正负选点、Top-K 和聚类均为离散操作，不能将 Dice loss 简单地穿过冻结 SAM 反向传播。因此主方案采用离线 action-utility learning，而不是声称端到端可微训练。

对每个 train-only reference-query episode：

1. 在预先冻结的 action grid 中运行候选 prompt；
2. 用该 episode 的 GT 计算 Dice、IoU、AJI、实例召回、面积偏差与调用成本；
3. 构造相对于该 episode 安全 anchor 的 utility，例如：

```text
U(a) = Dice(a) - α · prompt_count(a) - β · SAM_calls(a) - γ · oversegmentation(a)
```

4. 用 pairwise ranking 或 listwise ranking 训练策略预测动作相对效用；
5. 只在 validation set 中选择策略容量、预算上限、early-stop 阈值、utility 权重与 anchor；
6. 将所有选择冻结后，仅评估一次 test set。

GT 仅用于 train action reward 和最终 test reporting；它绝不进入 query 的策略输入，也不用于 test-time 候选选择。

## 3. 实验协议

### 3.1 数据集设计

至少使用三个具有不同组织与目标形态的数据集：

| 数据集 | 任务角色 | 用途 |
| --- | --- | --- |
| MoNuSeg | 密集病理细胞核 | 主任务、主表和完整消融 |
| GlaS | 腺体/腔体结构 | 与密集核不同的空间组织与负点需求 |
| CPM-17 或另一独立病理核数据集 | 域外病理核 | 评估同类任务的域迁移 |

数据审计在实验前完成：记录官方 split、病例/患者标识、图像层级与 mask 语义。若数据集包含患者信息，train/validation/test 必须患者隔离；reference 也必须来自当前允许的训练分区，不能与 query 泄漏。

### 3.2 Episode-level split

对每个数据集独立建立以下集合：

```text
train cases      → 生成 train-only reference-query episodes 和 action utility
validation cases → 选择所有模型与实验超参数
test cases       → 冻结一次性报告
```

- 训练中随机或分层抽取 reference，但 reference 与 query 不得是同一 case；
- 每个 test query 使用预先规定数量的 train-only reference；多 reference 结果先在同一 query 内聚合，再以 query 或患者为独立推断单位；
- 所有方法共享同一 reference 采样、同一候选池和同一 test episodes；
- 数据集间迁移实验中，target validation/test 均不得用于源域策略调参。

### 3.3 指标与统计

| 指标 | 作用 | 注意事项 |
| --- | --- | --- |
| Dice、IoU | 二值区域重叠 | 所有方法必须使用同一二值化与 resize 协议 |
| AJI | 细胞核实例质量 | 只有在统一、预注册的实例拆分后才报告；二值 mask 需经所有方法共享的 watershed/postprocess |
| object recall / merge rate | dense-instance 覆盖与粘连 | 解释 Dice 无法揭示的错误 |
| point precision / component coverage | matcher 与正点质量 | 不受 segmenter 输出混淆 |
| 平均正负点数、cluster ratio、SAM calls | 可解释性与成本 | 同时报请求数与实际落点数 |
| wall-clock runtime、VRAM | 部署代价 | 必须计入 matching、候选构造、policy 和全部 SAM calls |

主比较使用成对 bootstrap CI、胜/平/负和适当的 paired test。多 reference episode 不是独立样本；统计单位必须是 query 或患者，而不是 episode 或 patch 的简单总数。

## 4. 论文主实验资产

下面五个资产构成论文的完整论证链：**性能 → 组合泛化 → 决策机制 → 可解释性 → 直观证据**。表格中的数值在实验结束前均为空；不得使用目标值或示例值替代真实结果。

### Table 1. Comparison with matching-based prompt-generation methods

**目标：** 在 MoNuSeg 主协议中检验“学习 Prompt Policy”是否优于同一 matcher / segmenter 条件下的固定规则。

| Method | Prompt source | Positive | Negative | Adaptive policy | Dice ↑ | IoU ↑ | AJI ↑ | Prompt cost ↓ |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Segmenter single point | simulated/manual reference | yes | no | no |  |  |  |  |
| Segmenter box prompt | simulated/manual reference | box | no | no |  |  |  |  |
| Matching Top-1 | automatic | yes | no | no |  |  |  |  |
| Matching Top-5 / Top-K | automatic | yes | no | no |  |  |  |  |
| Matching + fixed cluster | automatic | yes | no | no |  |  |  |  |
| Matching + fixed pos/neg | automatic | yes | yes | no |  |  |  |  |
| Closest reproducible prior method | method-specific | method-specific | method-specific | method-specific |  |  |  |  |
| **Ours: Adaptive Prompt Policy** | automatic | yes | yes | yes |  |  |  |  |

**公平性约束：** “manual/simulated”点或框若由 GT 中心和 GT box 构造，只能标为 privileged upper-bound，不可与自动方法混写成同等输入条件。第三方方法只有在相同数据、同类 prompt 接口、无 test 调参条件下才进入数值比较。

Table 1 只有在对每个可比 baseline 都完成同协议重跑后，才能承担“优于现有 matching-based baselines”的结论；不能预先写成 SOTA。

### Table 2. Matching × segmenter generalization

**目标：** 测试接口兼容性和迁移范围，不预设 universal / model-agnostic 结论。

| Matching backbone | Segmentation backbone | Fixed-rule baseline Dice | Policy Dice | Δ Dice | Transfer setting | Same frozen policy? |
| --- | --- | ---: | ---: | ---: | --- | --- |
| DINOv2 | SAM |  |  |  | in-domain |  |
| DINOv2 | SAM2 |  |  |  | in-domain |  |
| DINOv2 | MedSAM / medical segmenter |  |  |  | in-domain |  |
| CLIP | SAM |  |  |  | in-domain |  |
| CLIP | SAM2 |  |  |  | in-domain |  |
| MAE / pathology foundation model | SAM |  |  |  | in-domain |  |
| MAE / pathology foundation model | SAM2 |  |  |  | in-domain |  |
| feature from segmenter encoder | compatible segmenter |  |  |  | in-domain |  |

每个组合必须和**该组合自身最强 validation-selected fixed-rule baseline**比较。结果分为两层报告：

1. **plug-compatible：** 同一候选与策略接口可运行，但允许为该组合重新训练 policy；
2. **policy transfer：** 在不重训、不看 target 标签下，源组合冻结 policy 仍产生正向或非劣收益。

只有第二层在预先指定组合上稳定成立，才可写“prompt policy transfers across models”。第一层仅支持“plug-compatible”。

### Table 3. Ablation of prompt decisions

**目标：** 证明性能不是单一模块偶然贡献，而是正点、负点与布局决策各自具有可测增益。

| Configuration | Adaptive positive | Adaptive negative | Adaptive layout | Dice ↑ | AJI ↑ | Prompt cost ↓ |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Fixed Top-K matching | no | no | no |  |  |  |
| + adaptive positive budget/selection | yes | no | no |  |  |  |
| + adaptive negative selection | yes | yes | no |  |  |  |
| + adaptive layout routing | yes | yes | yes |  |  |  |

同一节还必须报告三组受控子实验：

| Positive strategy | Dice ↑ | Negative strategy | Dice ↑ | Layout strategy | Dice ↑ |
| --- | ---: | --- | ---: | --- | ---: |
| Fixed K=1 |  | None |  | None |  |
| Fixed K=3 |  | Fixed bottom-3 |  | KMeans |  |
| Fixed K=5 |  | Fixed bottom-5 |  | DBSCAN |  |
| Fixed K=10 |  | Hard-background rule |  | MeanShift |  |
| Adaptive positive |  | Adaptive negative |  | Adaptive layout |  |

所有 ablation 共享 action grid、reference episodes、训练预算和 segmenter calls。不得只改变模块却悄悄提高候选数、调用数或 test-time 搜索规模。

### Table 4. Prompt statistics and cross-domain behavior

**目标：** 展示策略实际生成什么，而不是只汇报最终 Dice。

| Dataset | Avg. positive | Avg. negative | Cluster ratio | Avg. clusters | Prompt cost | Dice ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MoNuSeg |  |  |  |  |  |  |
| GlaS |  |  |  |  |  |  |
| CPM-17 / external nuclei set |  |  |  |  |  |  |

表中差异仅证明 policy 在不同数据上具有不同**行为**。要主张“学到了 dataset prompt prior”，还需同时满足：

1. 统计差异在独立 test cases 上稳定；
2. shared policy 与 domain-conditioned policy 的对比显示这种差异带来性能收益；
3. 不能由单纯的图像大小、目标数量或预算上限解释。

### Figure 4. Qualitative prompt decision process

每个案例采用相同排版列，展示：

```text
(a) reference image and mask
(b) foreground / background matching maps
(c) positive and negative candidate pools
(d) selected positive points
(e) selected negative points
(f) selected layout / final prompt
(g) predicted segmentation
(h) ground truth
```

每行选择一个固定对照：random / fixed Top-K / matching + fixed cluster / adaptive policy / GT。至少包含三类案例：

1. policy 用较少点而保持性能的简单图；
2. policy 用更多分散正点改善密集覆盖的图；
3. policy 选择负点或回退，以避免背景扩张的失败图。

定性图必须同时展示成功与失败；不能只选择最漂亮的案例。

## 5. 预注册的风险控制

| 风险 | 设计上的控制 |
| --- | --- |
| test-set 调参 | 所有动作空间、anchor、阈值和模型选择仅由 train/validation 决定 |
| reference 偶然性 | 多 reference episode、训练时随机/分层采样、test 时固定规则聚合 |
| 复杂策略仅因更多搜索获益 | 报告 SAM calls、候选数、prompt 数；固定相同调用预算的比较 |
| AJI 不可比 | 使用所有方法共享的实例化后处理，或只报告区域指标 |
| “泛化”被误写成重训后的效果 | 明确区分 plug-compatible、in-domain retraining 与 frozen-policy transfer |
| 近邻工作不可复现 | 只报告真正同协议复现的数值；其余写入相关工作与限制 |
| 多重比较偶然显著 | 预先指定主指标、主数据集和主比较；其余作为次要分析并报告完整结果 |

## 6. 最小可执行顺序

```text
1. 数据审计与患者/病例隔离 split
2. 单一 matcher × segmenter 的 reference-matching baseline
3. 冻结 candidate pool 与 action grid
4. 固定 Top-K / 固定负点 / 固定布局基线
5. train-only action utility 数据生成
6. adaptive positive → adaptive negative → adaptive layout 的逐步训练
7. Table 1、Table 3 与 Figure 4 完整闭环
8. 跨 matcher × segmenter 的 plug-compatible 与 frozen-transfer 实验
9. 多数据集 Table 4 与 shared/domain-conditioned policy 对照
10. 近邻方法公平复现、端到端成本表、论文写作
```

在第 7 步前不要声称完整方法有效；在第 8 步前不要声称 plug-and-play；在第 9 步前不要声称 dataset prompt prior 或跨域 policy。

## 7. 论文叙事与标题

推荐主叙事：

> **Prompt should be learned, not hand-crafted.** Given a reference-defined task, Adaptive Prompt Policy learns how to allocate positive evidence, negative evidence and spatial organization for a frozen segmentation foundation model.

可选英文标题：

```text
Adaptive Prompt Policy Learning for Matching-Based Segmentation Foundation Models
```

只有在 Table 1 的全部可比 baseline 与 Table 2 的冻结迁移实验均有强证据时，才将摘要或标题升级为“generalizable”或“plug-and-play”。否则使用更准确的限定语：`reference-conditioned`, `matching-based`, `for dense medical segmentation`。

