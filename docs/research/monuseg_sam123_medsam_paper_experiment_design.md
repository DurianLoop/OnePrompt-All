# MonuSeg 主线论文实验设计：Baseline 对比、辅助证明实验与消融实验

日期：2026-07-09

## 0. 本次修正

这份文档重写了上一版中不专业、容易引起误解的部分。

首先，早期关于 SAM3 text prompt 的旁支验证不再作为主线内容。它和我们现在的 MonuSeg reference-guided point prompting 主线关系很弱，后续论文方案中不再使用它作为核心证据，也不再进入主实验、辅助实验或消融实验。

其次，实验设计改成论文中更常见的结构：

```text
主实验：和 baseline 正面对比，回答方法是否有效。
辅助证明实验：解释为什么有效，证明机制成立。
消融实验：拆掉关键模块，证明每一部分都有贡献。
泛化/展示实验：在 SAM / SAM2 / SAM3 / MedSAM 上统一展示。
```

最后，明确区分：

```text
已有实验 = 冻结模型 + 网格搜索 + 评价
未来论文方法 = 可学习 point/prompt policy + baseline 对比 + 消融
```

当前还没有训练模型，所以不能把已有结果写成 learned method。

## 1. 本地模型版本确认

这一节只确认“本地模型对应哪个公开模型体系”，不夸大成“已经使用论文最强配置”。

| 名称 | 本地 checkpoint | 对应公开模型/论文体系 | 当前状态 | 需要注意 |
| --- | --- | --- | --- | --- |
| SAM ViT-B | `assets/checkpoints/sam_vit_b.pth` | Segment Anything / SAM | 已接入并完整跑过 MonuSeg dense point 实验 | 是 SAM 的 ViT-B，不是 ViT-H 最强版本 |
| SAM ViT-H | `assets/checkpoints/sam_vit_h_4b8939.pth` | Segment Anything / SAM | 本地有权重，尚未纳入本轮 MonuSeg dense point grid | 可作为更强 SAM baseline 补充 |
| SAM2.1 Hiera Tiny | `assets/checkpoints/sam2.1_hiera_tiny.pt` | SAM 2 / SAM 2.1 | demo 中已有接口，尚未纳入本轮 MonuSeg dense point grid | Tiny 是轻量模型，不代表 SAM2 最强配置 |
| SAM3 | `assets/checkpoints/sam3_modelscope/sam3.pt` | SAM 3 | 已接入并完整跑过 SAM3 point grid | 本地是 ModelScope 镜像权重，调用的是 interactive point prompt 路径，不等价于论文完整 text/concept 评测 |
| MedSAM ViT-B | `assets/checkpoints/medsam_vit_b.pth` | MedSAM | demo 中已有 box prompt 接口，尚未纳入本轮 dense point grid | MedSAM 主设定更偏 box prompt，直接拿 point prompt 比不公平 |

结论：

```text
这些模型都能对应公开论文中的模型家族，但本地使用的是具体 checkpoint 和具体 prompt 接口。
论文中比较时必须写清楚：模型版本、checkpoint、prompt 类型、是否为最强配置。
```

参考来源：

- SAM: Segment Anything, arXiv 2304.02643, official repository `facebookresearch/segment-anything`。
- SAM2: SAM 2: Segment Anything in Images and Videos, arXiv 2408.00714, official repository `facebookresearch/sam2`。
- SAM3: official repository `facebookresearch/sam3` and local ModelScope checkpoint metadata。
- MedSAM: Segment anything in medical images, Nature Communications 2024, official repository `bowang-lab/MedSAM`。

## 2. 论文主线应该是什么

不要把论文写成：

```text
SAM3 某些无关 prompt 现象。
```

也不要写成：

```text
SAM3 不如 SAM。
```

这两个都不是主线，而且会显得偏题。

更合理的主线是：

```text
给定一张带 mask 的 reference 图，如何把 reference mask 中的目标信息转化为 query 图上的有效 prompt，从而提升 SAM-family 模型在 MonuSeg all-nuclei segmentation 上的表现。
```

核心问题是：

```text
reference mask -> matched points -> prompt policy -> segmenter output
```

我们真正要研究的是 prompt policy：

1. 点从哪里来？
2. 保留多少个点？
3. 点是否需要分散、聚类或选局部峰值？
4. support mask 外的背景信息是否有用？
5. 不同 segmenter 是否需要不同 prompt policy？

更适合的论文题目方向：

```text
Reference-Guided Learnable Point Prompting for Nuclei Segmentation
```

或：

```text
Learning Model-Aware Point Prompts from One Reference Mask for SAM-family Medical Segmentation
```

## 3. 当前已有结果如何放进论文

当前已有两组完整实验。

### 3.1 SAM ViT-B dense point 实验

| 项目 | 数值 |
| --- | ---: |
| cases | 20 |
| candidates | 340 |
| best K | 20 |
| best lambda | 0.25 |
| best layout | local maxima |
| best fixed-strategy Dice | 0.3861 |
| candidate oracle Dice | 0.4464 |

这说明：

```text
在 SAM ViT-B 上，reference-guided dense matched points 能形成比旧 Stage3 更强的候选池。
```

### 3.2 SAM3 dense point 重测

| 项目 | 数值 |
| --- | ---: |
| cases | 20 |
| candidates | 1460 |
| best K | 10 |
| best lambda | 0.75 |
| best layout | DBSCAN medoid |
| best fixed-strategy Dice | 0.0742 |
| candidate oracle Dice | 0.1287 |

这说明：

```text
同样的 dense positive point 思路不能直接迁移到 SAM3 interactive point prompt。
```

这个结果不应该被写成“SAM3 差”，而应该被写成：

```text
prompt policy 是 model-dependent 的。同一组 reference-guided points 在 SAM 和 SAM3 上响应差异很大，因此需要学习 model-aware prompt policy。
```

## 4. 实验总设计

实验应该分四类。

```text
A. 主实验：证明我们的方法比 baseline 好。
B. 辅助证明实验：证明 reference matching、dense points、background suppression 确实有效。
C. 消融实验：拆掉模块，看性能下降。
D. 跨模型实验：SAM / SAM2 / SAM3 / MedSAM 统一比较。
```

## 5. 主实验：和 baseline 正面对比

主实验只回答一个问题：

```text
我们的方法是否比合理 baseline 更好？
```

### 5.1 Baseline 分组

#### Group 1：无 reference 的普通 prompt baseline

| Baseline | Prompt 来源 | 训练 | 意义 |
| --- | --- | --- | --- |
| center point | 图像中心或目标先验中心 | no | 最弱固定点 baseline |
| grid point | 固定网格点 | no | 测试“密集点但不匹配”是否有效 |
| random point | 随机点 | no | 随机提示下限 |
| text prompt | `nuclei` / `cell nucleus` | no | 只适合 SAM3 或支持 text 的模型 |
| box prompt | 粗 box / full image box / GT box oracle | no | MedSAM/SAM 常用强 baseline |

注意：

```text
GT box / GT foreground random point 只能作为 oracle upper bound，不能作为公平实际方法。
```

#### Group 2：reference matcher baseline

| Baseline | 说明 |
| --- | --- |
| fg-only top-k | 只用 support mask 内前景 prototype |
| fg-bg top-k | 加入 background prototype |
| fg-bg local maxima | 当前 SAM ViT-B 表现较好的布局 |
| fg-bg DBSCAN | 当前 SAM3 表现相对较好的布局 |
| full grid oracle | 在测试集上事后选最优，只作为上限 |

#### Group 3：学习型方法

| Method | 训练内容 | 输出 |
| --- | --- | --- |
| learned selector | 学会从 candidates 中选一个 mask | proposal score |
| learned K/lambda policy | 学会选择 K 和 lambda | K, lambda |
| learned layout policy | 学会选择 top-k / local maxima / cluster | layout |
| full learned policy | 同时选择 K、lambda、layout、proposal | final mask |

第一篇论文建议先做轻量学习，不要微调 SAM/SAM3 主干。

### 5.2 主表建议

最终主表应该像这样：

| Method | Segmenter | Reference mask | Train policy | candidates/case | Dice | IoU | Obj recall | Merge error |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| center point | SAM | no | no | 1 | 待填 | 待填 | 待填 | 待填 |
| grid point | SAM | no | no | 1 | 待填 | 待填 | 待填 | 待填 |
| fg-only matcher | SAM | yes | no | 1 | 待填 | 待填 | 待填 | 待填 |
| fg-bg matcher | SAM | yes | no | 1 | 待填 | 待填 | 待填 | 待填 |
| learned policy | SAM | yes | yes | 1-3 | 待填 | 待填 | 待填 | 待填 |
| learned policy | SAM2 | yes | yes | 1-3 | 待填 | 待填 | 待填 | 待填 |
| learned policy | SAM3 | yes | yes | 1-3 | 待填 | 待填 | 待填 | 待填 |
| learned policy | MedSAM | yes | yes | 1-3 | 待填 | 待填 | 待填 | 待填 |
| nnU-Net | none | no | full supervision | 0 | 待填 | 待填 | 待填 | 待填 |

nnU-Net 必须标注为 fully supervised upper bound，不是同等 prompt setting。

## 6. 辅助证明实验

辅助证明实验不是和 baseline 争第一，而是解释为什么方法有效。

### 6.1 Reference matching 是否真的找到目标

实验：

```text
support mask -> DINO fg/bg prototype -> query score map
```

评价：

1. top-k points 中落在 GT mask 内的比例。
2. top-k points 覆盖了多少 GT connected components。
3. score map 与 downsampled GT 的 AUC / AP / Dice。

表格：

| Matching | point precision@K | component recall@K | score-map AP | 说明 |
| --- | ---: | ---: | ---: | --- |
| fg-only | 待填 | 待填 | 待填 | 只看前景相似度 |
| fg-bg | 待填 | 待填 | 待填 | 加背景抑制 |
| learned score | 待填 | 待填 | 待填 | 学习权重 |

### 6.2 Dense point 是否适合 MonuSeg

实验：

```text
K = 1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 75, 100
```

评价：

1. Dice / IoU。
2. foreground recall。
3. object recall。
4. predicted area。
5. merge error。

要证明的不是“K 越大越好”，而是：

```text
MonuSeg 需要 task-specific point budget。
```

### 6.3 Background suppression 是否有用

实验：

```text
lambda = 0, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2
```

评价：

1. Dice 是否提升。
2. false positive / predicted area 是否下降。
3. foreground recall 是否损失过大。

要证明的是：

```text
support mask 外的背景信息不是废信息，它可以帮助抑制 query 中相似背景。
```

### 6.4 Model response analysis

实验：同一批 matched points 输入不同模型：

```text
SAM, SAM2, SAM3, MedSAM
```

评价不同模型对同一 prompt 的响应差异：

| Model | dense point 是否有效 | 是否容易扩张 | 是否偏 instance | 推荐 prompt |
| --- | --- | --- | --- | --- |
| SAM | 待填 | 待填 | 待填 | point / box |
| SAM2 | 待填 | 待填 | 待填 | point / box |
| SAM3 | 待填 | 待填 | 待填 | text / point / exemplar |
| MedSAM | 待填 | 待填 | 待填 | box |

这部分用于解释为什么需要 model-aware prompt policy。

## 7. 消融实验

消融实验要围绕我们自己的方法拆模块。

### 7.1 消融表设计

| Variant | fg proto | bg proto | K policy | layout policy | selector | Dice | 说明 |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| full method | yes | yes | learned | learned | learned | 待填 | 完整方法 |
| w/o bg | yes | no | learned | learned | learned | 待填 | 去掉背景抑制 |
| fixed K | yes | yes | fixed | learned | learned | 待填 | 不学习点数 |
| fixed layout | yes | yes | learned | fixed | learned | 待填 | 不学习布局 |
| no selector | yes | yes | learned | learned | no | 待填 | 直接用模型 score |
| fg-only top-k | yes | no | fixed | top-k | no | 待填 | 最基本 matcher |

### 7.2 必须回答的问题

消融实验要能回答：

1. background prototype 是否有贡献？
2. K 是否需要学习？
3. layout 是否需要学习？
4. proposal selector 是否比 SAM score 更可靠？
5. model-aware policy 是否比单一固定策略更好？

## 8. 训练模块设计

当前还没有训练。论文方法建议训练三个轻量模块，按优先级推进。

### 8.1 模块一：proposal selector

这是最容易先做出来的训练模块。

输入：每个 proposal 的特征。

| 特征 | 含义 |
| --- | --- |
| K | prompt 点数量 |
| lambda | 背景抑制强度 |
| layout | 点布局策略 |
| mean point score | prompt 点平均 matcher 分数 |
| score std | prompt 点质量是否稳定 |
| fg-bg gap | 前景和背景可分性 |
| SAM score | segmenter 自信度 |
| mask area | 预测面积 |
| bbox area | 外接框面积 |
| component count | 连通域数量 |
| prompt coverage | prompt 点是否被 mask 覆盖 |

输出：

```text
proposal quality score
```

训练标签：

```text
proposal Dice 或 accept/reject 标签
```

推荐损失：排序损失。

同一个 case 内，如果 proposal A 的 Dice 高于 proposal B：

```text
q(A) > q(B)
```

损失：

```text
L_rank = max(0, margin - q_A + q_B)
```

也可以同时加回归损失：

```text
L_reg = SmoothL1(q, Dice)
```

总损失：

```text
L_selector = L_rank + beta * L_reg
```

### 8.2 模块二：K / lambda / layout policy

目标：不再跑大网格，而是根据 score map 自动选择 prompt policy。

输入：score map statistics。

| 特征 | 含义 |
| --- | --- |
| heatmap entropy | 高分区域是否分散 |
| top-k score mean/std | 高分点是否稳定 |
| local maxima count | 可能目标数量 |
| fg-bg separation | 前景背景区分度 |
| high-score area ratio | 高响应区域面积 |
| score gradient mean | 是否有明显边界 |

输出：

```text
K class
lambda class
layout class
```

训练标签来自训练集网格搜索 oracle：

```text
oracle_K, oracle_lambda, oracle_layout = argmax Dice on train case
```

损失：

```text
L_policy = CE(pred_K, oracle_K)
         + CE(pred_lambda, oracle_lambda)
         + CE(pred_layout, oracle_layout)
```

如果担心 oracle label 太硬，可以用 soft label：

```text
p(strategy) = softmax(Dice(strategy) / temperature)
```

损失改为 KL divergence：

```text
L_policy = KL(pred_distribution || oracle_soft_distribution)
```

### 8.3 模块三：score re-weighting

把手写 score：

```text
score(x) = z(s_fg(x)) - lambda * z(s_bg(x))
```

扩展为可学习：

```text
score(x) = w_fg * z(s_fg(x))
         - w_bg * z(s_bg(x))
         + w_peak * peakness(x)
         + w_edge * edge_score(x)
         + w_spread * diversity_prior(x)
```

监督方式：把 query GT 下采样到 feature map 尺寸。

损失：

```text
L_bce = BCE(sigmoid(score_map), downsampled_GT)
L_dice = DiceLoss(sigmoid(score_map), downsampled_GT)
L_score = L_bce + L_dice
```

还可以加入 point ranking loss：

```text
L_point = max(0, margin - score_pos + score_neg)
```

总损失：

```text
L = L_score + alpha * L_point + beta * L_policy + gamma * L_selector
```

第一版不要一次性全训。建议顺序：

```text
proposal selector -> K/lambda/layout policy -> score re-weighting
```

## 9. 训练 / 测试划分

必须使用 case-level split，不能把同一个 case 的 proposals 拆到 train 和 test 两边。

当前 20 个 MonuSeg case：

```text
5-fold cross validation
每折 16 cases train，4 cases test
同一个 case 的所有 proposals 只属于 train 或 test
```

如果后续扩展到 50 个 case：

```text
5-fold cross validation
每折 40 cases train，10 cases test
```

评价时报告：

1. 每折 test Dice。
2. 5-fold mean/std。
3. oracle gap。
4. candidates per case。
5. 和固定 baseline 的差值。

## 10. 跨模型实验怎么做才公平

不同模型支持的 prompt 类型不同，不能强行用一个 prompt 类型说谁强谁弱。

建议分两张表。

### 10.1 同一 prompt policy 表

目的：看同一批 points 输入不同模型的响应差异。

| Model | Prompt | Dice | IoU | Obj recall | 说明 |
| --- | --- | ---: | ---: | ---: | --- |
| SAM | same points | 待填 | 待填 | 待填 | point prompt baseline |
| SAM2 | same points | 待填 | 待填 | 待填 | image point prompt |
| SAM3 | same points | 待填 | 待填 | 待填 | interactive point prompt |
| MedSAM | same box | 待填 | 待填 | 待填 | MedSAM 不适合强行 point，对应用 box |

### 10.2 各模型最佳合理 prompt 表

目的：公平展示每个模型在合理 prompt 下的最佳能力。

| Model | Reasonable prompt | Policy | Dice | 说明 |
| --- | --- | --- | ---: | --- |
| SAM | point / box | learned point or box | 待填 | SAM 常规交互 prompt |
| SAM2 | point / box | learned point or box | 待填 | SAM2 image predictor |
| SAM3 | text / point / exemplar | model-aware policy | 待填 | 不强行只用 dense point |
| MedSAM | box | learned/refined box | 待填 | MedSAM 论文主设定偏 box |

这样可以避免误导：

```text
同一 prompt policy 是响应分析；各模型最佳合理 prompt 才是模型能力展示。
```

## 11. 推荐后续执行顺序

### P0：建立 model zoo 与下载清单

先不要直接把所有模型都放进完整大实验。更合理的是建立一个 model zoo，然后做小规模 screening。

当前本地已有：

| 模型 | 本地状态 | 是否进入 screening |
| --- | --- | --- |
| SAM ViT-B | 已有 | 是 |
| SAM ViT-H | 已有 | 是 |
| SAM2.1 Hiera Tiny | 已有 | 是 |
| SAM3 | 已有 | 是 |
| MedSAM ViT-B | 已有 | 是 |
| Lite-MedSAM | 已有 | 可作为轻量医学 baseline |
| medical adapter zoo 中的 microscope/fundus 权重 | 已有 | 可作为医学适配模型候选 |

建议补充下载：

| 模型 | 目的 | 优先级 |
| --- | --- | --- |
| SAM ViT-L | 补齐 SAM B/L/H 三档，避免只比较 B/H | P0 |
| SAM2.1 Hiera Small | 比 tiny 更强但仍较省资源 | P0 |
| SAM2.1 Hiera Base+ | 中等强度 SAM2 baseline | P1 |
| SAM2.1 Hiera Large | SAM2 强 baseline，但显存/速度成本高 | P1 |
| SAM3 官方 HuggingFace 权重副本 | 与 ModelScope 镜像交叉确认 | P1 |
| MedSAM2 / Medical SAM2 | 如果后续扩展到 video/3D 或跨切片，再纳入 | P2 |
| SAM-Med2D / Medical-SAM-Adapter 权重 | 医学适配 baseline，适合与 prompt policy 对比 | P2 |

下载后要维护一个 `model_zoo_manifest.csv`：

| 字段 | 含义 |
| --- | --- |
| model_key | 例如 `sam_vit_l`, `sam2p1_hiera_s` |
| family | SAM / SAM2 / SAM3 / MedSAM / adapter |
| checkpoint | 本地权重路径 |
| config | SAM2/SAM3 等需要的配置 |
| prompt_types | point / box / text / mask |
| source | official / mirror / third-party |
| paper | 对应论文 |
| status | downloaded / runnable / failed |
| notes | 是否是最强配置、是否镜像、是否需要授权 |

这样后续实验不会乱。

### P1：model zoo screening

先用小规模筛选，不直接上完整实验。

建议设置：

```text
cases = 5 个 MonuSeg validation cases
K = {1, 5, 10, 20, 50}
lambda = {0, 0.25, 0.75, 1.5}
layout = {top-k, local maxima}
```

每个模型先跑少量候选，得到：

| Model | Prompt type | screening Dice | speed | memory | 是否进入 full experiment |
| --- | --- | ---: | ---: | ---: | --- |
| SAM ViT-B | point | 待填 | 待填 | 待填 | 待填 |
| SAM ViT-L | point | 待填 | 待填 | 待填 | 待填 |
| SAM ViT-H | point | 待填 | 待填 | 待填 | 待填 |
| SAM2.1 Tiny | point | 待填 | 待填 | 待填 | 待填 |
| SAM2.1 Small | point | 待填 | 待填 | 待填 | 待填 |
| SAM3 | point/text/exemplar | 待填 | 待填 | 待填 | 待填 |
| MedSAM | box | 待填 | 待填 | 待填 | 待填 |

筛选规则要预先写清楚：

```text
每个 family 选 1-2 个代表模型进入完整实验：
1. validation Dice 最高；
2. 如果性能接近，优先选择速度/显存更合理的版本；
3. 如果某模型是论文中更常用或更强的配置，即使慢，也保留一个强 baseline。
```

这样主文可以放少数效果最好的模型，但附录必须放 screening 全表，避免被质疑 cherry-pick。

### P2：修正报告叙事

把已有报告中的早期旁支验证降级为附录或删除，不再作为主线。

### P3：补齐基础 baseline

先在 SAM ViT-B 上补：

1. center point。
2. grid point。
3. fg-only matcher。
4. fg-bg matcher。
5. GT box oracle。
6. random GT foreground point oracle。

### P4：补 SAM2 / MedSAM 同协议评测

优先补：

```text
SAM2.1 Hiera Tiny point/box
MedSAM ViT-B box
SAM ViT-H point/box
```

### P5：训练 proposal selector

先用已有 candidates.csv，不需要重新跑模型。

输出：

```text
selector 5-fold test table
```

### P6：训练 K/lambda/layout policy

目标是减少 candidates/case，同时接近 grid oracle。

### P7：统一网页报告

生成：

```text
docs/reports/monuseg_paper_experiment_plan/index.html
```

包含：

1. 模型版本确认。
2. baseline 对比。
3. 辅助证明实验。
4. 消融实验。
5. training plan。
6. expected paper tables。

## 12. 直接按当前文档实施前还需要改进什么

当前文档已经比上一版更像论文方案，但直接实施前还需要补三件事。

### 12.1 明确 train/validation/test，而不是只做 5-fold

如果要训练 policy，最好不要只写 5-fold。更稳的是：

```text
训练集：训练 selector / policy
验证集：选择模型版本、K/lambda 搜索空间、早停和超参数
测试集：只做最终报告
```

如果样本只有 20 个，可以用 nested cross-validation 或 repeated 5-fold：

```text
outer fold: test
inner fold: validation/model selection
```

这样“挑出几个效果最好的模型”就发生在 validation，而不是 test。

### 12.2 把 candidates/case 作为公平性指标

如果一个方法生成 1460 个 candidates，另一个只生成 340 个 candidates，直接比 oracle 不公平。

主表应该同时报告：

```text
Dice
oracle Dice
candidates/case
runtime/case
GPU memory
```

可以增加 budget curve：

| candidates/case | Method A Dice | Method B Dice |
| ---: | ---: | ---: |
| 1 | 待填 | 待填 |
| 3 | 待填 | 待填 |
| 5 | 待填 | 待填 |
| 10 | 待填 | 待填 |
| full grid | 待填 | 待填 |

这样能说明我们的方法是否真的高效，而不只是靠更多 candidates 堆出来。

### 12.3 明确每个模型的合理 prompt 类型

不要强行让所有模型都只用 point prompt。

建议分两条线：

```text
同 prompt 对照：同一批 points 输入 SAM/SAM2/SAM3，分析模型响应差异。
最佳合理 prompt：每个模型使用它论文中更合理的 prompt 类型，展示实际可用能力。
```

例如：

| 模型 | 同 prompt 对照 | 最佳合理 prompt |
| --- | --- | --- |
| SAM | point | point/box |
| SAM2 | point | point/box |
| SAM3 | point | text/exemplar/point hybrid |
| MedSAM | 不强行 point | box/refined box |

这样比法更专业。

## 13. 最终论文故事线

可以这样讲：

```text
MonuSeg 这类 dense all-nuclei segmentation 不是简单的单点交互问题。给定一个 support mask，我们可以用 foreground-background matching 在 query 图中找到候选目标点。但点的数量、布局、背景抑制和模型响应都强烈影响最终 mask。

因此，我们提出 reference-guided model-aware prompt policy：先从 support mask 生成 foreground/background score map，再学习选择 K、lambda、layout 和 proposal。实验通过 baseline 对比证明方法有效，通过辅助实验证明 reference/background/dense points 的作用，通过消融实验证明各模块贡献，并在 SAM/SAM2/SAM3/MedSAM 上展示不同模型需要不同 prompt policy。
```

这比“某个 prompt 失败”更像论文，也更容易向老师解释。
