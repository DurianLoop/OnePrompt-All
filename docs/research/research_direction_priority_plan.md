# 研究方向优先级方案：SAM3 + Matcher + 医学 Promptable Segmentation

日期：2026-06-24

本方案基于当前代码进度、老师提出的问题、已有最小验证结果，以及本地论文包 `docs/papers2/` 中的 SAM3、Matcher、Medical SAM、One-Prompt、UniverSeg 等论文梳理得出。目标是给出下一步最值得推进的研究方向，而不是泛泛列一堆可做点。

## 1. 当前项目事实基础

已经完成：

1. SAM3 本地可通过 demo 路径做 text 和 point/interactive 推理。
2. 网页 demo 已加入 `cumulative/latest` point 控制，解决“只用最新点”的交互问题。
3. 已完成老师猜想的最小验证闭环：
   - `execute` 在自然图上返回 0；
   - `white girl` 在人物图上返回 0；
   - `girl` 在同一人物图上返回 3 个实例；
   - elephant 场景中累计点和正负点提升 mask score。
4. 已有 Matcher 代码和论文阅读材料，方向上和老师说的 Stage 1/2/3 高度对应。

关键文档：

| 文档 | 用途 |
| --- | --- |
| `docs/reports/sam3/sam3_teacher_minimal_validation_report.md` | 老师猜想的最小验证闭环。 |
| `docs/research/sam3_matcher_teacher_direction_report.md` | 老师原始问题逐项解释。 |
| `docs/ops/code_entrypoints.md` | 当前代码入口。 |
| `runs/teacher_minimal_validation/` | 最小验证结果和 overlay。 |

## 2. 论文梳理得到的核心判断

### 2.1 SAM3：text/concept 是主能力，但复杂文本不是万能

SAM3 论文把模型定位为 promptable concept segmentation，也就是从 SAM/SAM2 的点、框、mask 进一步走向 concept/text prompt。但 SAM3 model card 明确提到它不支持复杂文本查询或 referring expressions，且 niche domains 可能需要 fine-tuning。

这与我们当前验证一致：

- `elephant`、`girl` 这类简单名词可以工作。
- `white girl`、`execute` 失败。
- 所以我们的研究不应该写成“SAM3 text 不行”，而应该写成：

```text
SAM3 text 对简单 concept 有效，但对复合属性、动作词、指代表达和领域特定目标不稳定；需要 visual prompt / feedback / matching 作为 fallback。
```

### 2.2 Matcher：老师提到的 Stage 1/2/3 基本对应论文主流程

Matcher 的三段核心非常适合接到老师的问题上：

1. Correspondence Matrix Extraction：用 DINOv2 计算 reference 和 target patch feature 相似度。
2. Prompts Generation：双向匹配过滤 outlier，再对 matched points 聚类并采样 prompt。
3. Controllable Masks Generation：把多组 prompt 送进 SAM 得到 proposals，再做 instance-level matching，使用 EMD / purity / coverage 等指标筛选 mask。

这说明老师的方向不是随意想法，而是可以直接沿着 Matcher 主流程做增强。我们的切入点应该是：

```text
在 Matcher 的 matched points -> prompt sampling -> instance filtering 链条里加入少量人工 feedback，让 sampling 更稳定，让多实例 K 估计更可靠。
```

### 2.3 医学 SAM 论文：不能只讲自然图，需要回到医学域问题

医学 benchmark 论文反复强调 raw SAM 在医学图像上不稳定，尤其受以下因素影响：

- 目标大小；
- 对比度；
- 2D/3D 维度；
- 模态；
- segmentation difficulty；
- prompt 类型和交互成本。

Medical SAM Adapter / Med-SA 等论文说明：适配医学域可以提升性能，但训练成本更高。One-Prompt 和 UniverSeg 则说明医学分割里 support/example 的选择本身就是一个重要研究轴。

因此当前最合适的路线不是马上 fine-tune，而是先做低成本、训练-free 或轻训练的 prompt/matching/feedback 机制。

## 3. 候选研究方向池

### 方向 A：Feedback-Guided Cluster-Aware Prompt Sampling

中文名：反馈引导的聚类感知 prompt 采样。

核心想法：

```text
Matcher Stage 1 给出一批 matched points；
用户只标少量正确点和错误点；
用 feedback 修正 matched point 分布；
对修正后的 points 聚类；
每个 cluster 单独采样 prompt；
送入 SAM3/SAM 得到 mask proposals；
最后用 instance-level matching 过滤。
```

对应老师问题：

- “Stage 1 给 100 个 points，手动选择 a 个正确点 b 个错误点，更新 target feature，matching 是否更好？”
- “sample N% points 送进 SAM，重复 m 次得到 m 个分割是否可行？”
- “多个物体很多时，怎么知道有 K 个物体？”
- “instance-level matching threshold 怎么做？”

优点：

- 最贴近老师问题。
- 最贴近 Matcher 论文主流程。
- 不需要马上训练新模型。
- 当前已有 SAM3 point、Matcher、demo 基础。
- 很容易形成论文式方法图：feedback -> clustering -> sampling -> filtering。

风险：

- 需要把 Matcher 的中间 matched points 暴露出来。
- 没有 GT 时只能先做可视化和 proxy metrics。
- feature feedback 公式需要谨慎设计，不能只做 UI 交互。

优先级：P0，最推荐主线。

### 方向 B：SAM3 Text Failure Diagnosis and Visual Prompt Fallback

中文名：SAM3 文本失败诊断与视觉提示回退。

核心想法：

```text
先判断 SAM3 text prompt 是否可靠；
如果 text 返回 0、实例过多、score 低或面积异常；
自动切换到 Matcher/SAM3 point prompt；
必要时用用户少量 feedback 修正。
```

对应老师问题：

- “SAM3 文本提示失效了。”
- “point 效果比较好。”

优点：

- 已经有最小验证证据。
- 可以快速形成 demo 效果。
- 容易和 SAM3 论文的 complex query limitation 对齐。

风险：

- 只做 text failure taxonomy 容易偏工程，不够方法创新。
- 如果不接 Matcher/feedback，论文贡献偏弱。

优先级：P1，适合作为 P0 主线的动机和前置模块。

### 方向 C：Sliding-Window No-Downsample Matching

中文名：不降采样大图滑窗匹配。

核心想法：

```text
大图不能 resize 时，切成 tile；
每个 tile 内做 DINO matching 和 SAM/SAM3 mask；
坐标映射回原图；
跨 tile 做 mask merge / NMS。
```

对应老师问题：

- “如果图的大小大于 DINO 规定大小，且不允许降采样怎么办？”

优点：

- 是必须解决的技术问题。
- 对医学大图、病理图、高清自然图都有意义。

风险：

- 偏系统工程。
- 如果单独作为论文主线，创新性不如 feedback + sampling。
- tile overlap、边界对象、跨 tile 合并会引入很多细节。

优先级：P1.5，作为 P0 方法的工程支撑，不单独做主线。

### 方向 D：Medical One-Prompt / Support Selection for SAM3-Matcher

中文名：医学单样本提示/支持样本选择。

核心想法：

```text
参考 One-Prompt 和 UniverSeg；
用一个 support image/mask 或少量 support examples 定义医学任务；
用 Matcher/SAM3 迁移到 query image；
重点研究 support 选择、prompt 类型和少量 feedback。
```

优点：

- 与医学域更贴合。
- 能和 UniverSeg/One-Prompt 对话。
- 可扩展为“医学 reference prompt selection”。

风险：

- 数据组织、support split、医学 GT 会增加工作量。
- 当前老师问题更集中在 SAM3/Matcher mechanics，先做这个可能绕远。

优先级：P2，适合 P0 成功后的医学化扩展。

### 方向 E：Adapter / LoRA / Fine-tuning SAM3 or SAM

中文名：医学域轻量适配。

核心想法：

```text
参考 MedSAM、Medical SAM Adapter、SAMed；
冻结主干，只训练 adapter/LoRA；
解决医学域边界、低对比、小目标问题。
```

优点：

- 医学性能上限更高。
- 容易和 MedSAM/Med-SA 等论文比较。

风险：

- 训练成本高。
- 需要稳定医学数据集和 GT。
- 与老师当前反馈的 Stage 1/2/3 sampling 问题不是同一层。

优先级：P3，后续有数据和算力后再做。

## 4. 推荐主线

我建议主线定为：

```text
Feedback-Guided Cluster-Aware Prompt Sampling for SAM3/Matcher
```

中文可写：

```text
面向 SAM3/Matcher 的反馈引导聚类感知提示采样方法
```

一句话定义：

```text
当 SAM3 text prompt 对复杂概念失败时，用 Matcher 产生候选 matched points，再通过少量正负反馈修正 feature matching，基于聚类进行多实例 prompt sampling，最后用 instance-level matching 过滤 SAM3/SAM proposals。
```

这条主线的优势是把老师所有问题串成一个闭环：

| 老师问题 | 主线中的位置 |
| --- | --- |
| text `white girl` / `execute` 失败 | motivation：text failure。 |
| point foreground/background 效果好 | visual prompt fallback。 |
| demo 只用最新点 | 已修：支持 cumulative feedback。 |
| Stage 1 matching 怎么做 | DINO/Matcher correspondence。 |
| 手动正负点更新 target feature | feedback-guided matching。 |
| sample N% points 重复 m 次 | prompt sampling module。 |
| 多物体很多怎么办 | cluster-aware grouping。 |
| 代码知道 K 个物体 | cluster + mask NMS + instance threshold。 |
| instance-level threshold | proposal filtering。 |
| 大图不能降采样 | sliding-window matching extension。 |

## 5. 优先级实验方案

### P0-1：Stage 1 matched points 可视化与导出

目标：先证明我们能复现 Matcher 的第一阶段。

当前进度：已完成第一版导出脚本和 elephant smoke case。报告见 `docs/reports/matcher/matcher_stage1_export_report.md`，输出目录为 `runs/research_direction/stage1_matching/elephant_pair/`。

要做：

1. 从 `code/matcher/` 或 `Matcher/` 中找出 DINO feature matching 输出位置。
2. 导出 top-k matched points、similarity、坐标。
3. 生成 target image 上的 point overlay 和 heatmap。
4. 对 top-20、top-50、top-100 分别保存结果。

输出：

```text
runs/research_direction/stage1_matching/
```

指标：

| 指标 | 目的 |
| --- | --- |
| top-k point overlay | 看 matched points 是否落在目标物体上。 |
| similarity heatmap | 看 feature matching 是否集中。 |
| point spread | 判断是否需要聚类。 |
| manual positive/negative labels | 为 feedback 准备。 |

成功标准：

```text
至少在 2-3 张自然图上，top matched points 的主要区域与目标相关；并能看到 outlier。
```

### P0-2：正负 feedback 是否改善 matching

目标：正面验证老师最关心的“选正确点/错误点后 matching 是否更好”。

最小方法：

1. 用户或脚本指定少量 positive points 和 negative points。
2. 正点对应区域 feature 拉近 target prototype。
3. 负点对应区域 feature 加惩罚或从 candidate set 中降权。
4. 重新排序 top-k matched points。
5. 比较 feedback 前后的点分布。

可先做两版：

| 版本 | 方法 | 用途 |
| --- | --- | --- |
| V1 简单重排 | 正点附近加权，负点附近降权 | 快速验证现象。 |
| V2 feature prototype update | 更新 target/reference prototype | 更像论文方法。 |

指标：

| 指标 | 解释 |
| --- | --- |
| top-k precision proxy | top-k 中落在目标区域的比例；没有 GT 时先人工框/粗 mask。 |
| mean distance to positive | top-k 是否向正确点靠拢。 |
| mean distance to negative | top-k 是否远离错误点。 |
| heatmap concentration | matching 是否更集中。 |

成功标准：

```text
feedback 后 top-k points 更集中在目标实例，outlier 减少，并且可视化上明显更好。
```

当前进度：已完成 V1 最小闭环验证。报告见 `docs/reports/matcher/matcher_stage1_feedback_report.md`，输出目录为 `runs/research_direction/stage1_feedback/elephant_pair/`。

关键结果：

| 指标 | Feedback 前 | Feedback 后 |
| --- | ---: | ---: |
| threshold=0.55 selected count | 9 | 43 |
| top-k 到正点平均距离 | 128.379 | 83.822 |
| top-k 70px 正点邻域点数 | 22 | 43 |
| top-k 70px 负点邻域点数 | 16 | 0 |

阶段判断：该结果支持老师关于“少量正负 feedback 可以改善 Stage 1 matching”的猜想，但目前仍是空间 reranking 版本；下一版应做 feature/prototype update，并接入 Stage 2 prompt sampling。

### P0-3：Stage 2 多次 prompt sampling

目标：验证 Matcher 论文中的 robust prompt sampler 是否适合我们的 SAM3/SAM3 point 路径。

要做：

1. 取 feedback 后 top N% matched points。
2. 对点做 DBSCAN 或 K-means 聚类。
3. 每个 cluster 内 sample 多组 points。
4. 每组 points 送进 SAM3 interactive 或 SAM。
5. 生成 m 个 mask proposals。

实验变量：

| 变量 | 值 |
| --- | --- |
| top ratio | 5%、10%、20% |
| sample size | 1、2、3、5 points |
| cluster method | none、K-means、DBSCAN |
| feedback | no feedback、positive-only、positive+negative |

指标：

| 指标 | 解释 |
| --- | --- |
| proposal count | m 次采样产生多少有效 mask。 |
| mask area distribution | 是否过小/过大。 |
| proposal diversity | masks 是否都一样。 |
| best score | SAM3/SAM 内部分数。 |
| visual quality | overlay 人工检查。 |

成功标准：

```text
相比单点，cluster-aware multi-point sampling 能产生更完整、更稳定、更多样的 proposals。
```

当前进度：已完成 Stage 2 聚类采样最小闭环。报告见 `docs/reports/matcher/matcher_stage2_sampling_report.md`，输出目录为 `runs/research_direction/stage2_sampling/elephant_pair/`。

关键结果：

| 指标 | 结果 |
| --- | ---: |
| feedback 后 selected candidate points | 43 |
| anchor-distance 过滤后 candidate points | 40 |
| DBSCAN cluster count / estimated K | 1 |
| noise points | 0 |
| prompt group count | 12 |

阶段判断：在单目标 elephant case 中，feedback 后的点可以形成合理的实例级 cluster，`K_pred=1` 符合预期，并且已经能生成 1/3/5 点 prompt groups。下一步应将 `prompt_groups.json` 逐组送入 SAM3/SAM 生成 mask proposals，验证 “重复 m 次得到 m 个分割” 的后半段。

P0-3b 进度：已将 12 组 prompt groups 真实送入本地 SAM ViT-B，生成 12 个 mask proposals。报告见 `docs/reports/matcher/matcher_stage2_sam_proposals_report.md`，输出目录为 `runs/research_direction/stage2_proposals/elephant_pair/`。

关键结果：

| 指标 | 结果 |
| --- | ---: |
| proposal count | 12 |
| SAM score mean/min/max | 0.905409 / 0.709140 / 0.987186 |
| mask area mean/min/max | 33.3608% / 1.0176% / 70.9880% |
| sample size 1 area mean | 21.3485% |
| sample size 3 area mean | 24.5443% |
| sample size 5 area mean | 54.1897% |

阶段判断：老师提出的 “重复 m 次得到 m 个分割” 已经跑通。现象上，3 点 prompts 相对更稳；5 点 prompts 虽然 SAM score 更高，但容易过分扩张到背景，说明 Stage 3 不能只按 SAM score 选 mask，必须加入 instance-level matching、面积过滤和 NMS。

### P0-4：Stage 3 instance-level filtering 和 K 估计

目标：回应“代码知道 K 个物体存在，然后一个个和 target match”。

要做：

1. 对 m 个 proposals 做 mask NMS，去掉重复。
2. 对每个 proposal 提取 DINO/SAM feature。
3. 与 reference/target prototype 做 instance-level similarity。
4. 设置 threshold，保留高分 masks。
5. 输出估计实例数 K。

初版可以先不用 EMD，先做简单版：

```text
proposal feature mean pooling -> cosine similarity -> threshold -> NMS
```

第二版再补 Matcher 论文的 EMD / purity / coverage。

指标：

| 指标 | 解释 |
| --- | --- |
| K_pred | 估计实例数。 |
| retained masks | threshold 后保留的 proposal。 |
| duplicate rate | NMS 前后重复率。 |
| instance similarity | 每个 proposal 与 target/reference 的相似度。 |
| Dice/IoU | 有 GT 时计算。 |

成功标准：

```text
在多实例自然图上，能够从多个 proposals 中筛出目标实例；在医学图上能减少明显 false positives。
```

当前进度：已完成 Stage 3 mask-level instance filtering 最小闭环。报告见 `docs/reports/matcher/matcher_stage3_filtering_report.md`，输出目录为 `runs/research_direction/stage3_filtering/elephant_pair/`。

关键结果：

| 指标 | 结果 |
| --- | ---: |
| input proposals | 12 |
| threshold 后 candidates | 1 |
| NMS 后 retained masks | 1 |
| `K_pred` | 1 |
| retained group id | 5 |
| retained SAM score | 0.936279 |
| retained mask area | 12.7992% |
| retained bbox area | 19.0690% |
| retained cluster coverage | 0.5750 |

阶段判断：老师提出的 Stage 1/2/3 最小链路已经跑通。当前 Stage 3 能用 score、面积、bbox、cluster coverage、prompt coverage 和 NMS 从 12 个 SAM proposals 中保留 1 个实例，`K_pred=1` 与单目标 elephant case 一致。但这仍是 mask-level rule 版本；下一步应升级为 DINO/Matcher feature-level instance matching。

P0-4b 进度：已完成 DINO/Matcher feature-level instance matching。报告见 `docs/reports/matcher/matcher_stage3_feature_matching_report.md`，输出目录为 `runs/research_direction/stage3_feature_matching/elephant_pair/`。

关键结果：

| 指标 | 结果 |
| --- | ---: |
| input proposals | 12 |
| feature threshold 后 candidates | 1 |
| NMS 后 retained masks | 1 |
| `K_pred` | 1 |
| retained group id | 5 |
| retained instance similarity | 0.784939 |
| retained feature score | 0.986067 |
| sample size 3 similarity mean | 0.701390 |
| sample size 5 similarity mean | 0.658832 |

阶段判断：P0-4b 比 P0-4 更接近老师说的 instance-level matching。它不只依赖 SAM score，而是把每个 proposal 的 DINO mask prototype 和 reference object prototype 做 cosine similarity。实验显示 5 点 prompts 虽然 SAM score 高，但 DINO instance similarity 和面积约束并不支持直接保留；最终保留的 3 点 group 5 在 feature similarity、面积和 coverage 上更均衡。

多实例自然图补充验证：已在 horse-head case 上验证 `K_pred > 1`。报告见 `docs/reports/matcher/matcher_multi_instance_horse_validation_report.md`，输出目录为 `runs/research_direction/multi_instance_horse_head/`。

关键结果：

| 指标 | 结果 |
| --- | ---: |
| Stage 1 selected points | 26 |
| Stage 2 DBSCAN clusters | 2 |
| Stage 2 noise points | 5 |
| SAM proposals | 16 |
| feature candidates | 6 |
| retained masks | 2 |
| `K_pred` | 2 |
| retained group ids | 7, 12 |

阶段判断：该多实例 case 说明 “matched points -> clustering -> per-cluster prompt sampling -> SAM proposals -> feature-level matching” 可以从一张含两匹马的自然图中估计出 `K_pred=2`。同时发现每个 cluster 内可能产生多个局部 proposal，因此新增 `--max-retained-per-cluster 1`，用于按 cluster 控制最终实例数，避免 K 被同一实例的多个局部 mask 高估。

### P1：SAM3 text failure taxonomy + fallback trigger

目标：把最小验证扩成一个小而扎实的 failure taxonomy。

要做：

1. 收集 20-50 个自然图/医学图 prompt。
2. prompt 分为：
   - 简单名词；
   - 复合属性；
   - 动作词；
   - referring expression；
   - 医学术语；
   - 同义词/缩写。
3. 记录 SAM3 text 是否返回 0、是否多实例、是否面积异常。
4. 设计 fallback trigger：
   - instances = 0；
   - score 低；
   - area 太小/太大；
   - box 不合理；
   - 多实例但用户只要一个实例。

输出：

```text
runs/research_direction/text_failure_taxonomy/
docs/sam3_text_failure_taxonomy.md
```

这部分作为论文动机，不建议单独做主线。

### P1.5：大图 sliding-window no-downsample

目标：回答 DINO input size 限制。

最小闭环：

```text
large image -> tile -> per-tile matching -> coordinate restore -> merge points/masks
```

优先做在 Stage 1 matching 层，不急着接完整 SAM proposal。

成功标准：

```text
不 resize 原图，也能输出全图坐标系下的 matched points 和 heatmap。
```

### P2：医学小数据验证

目标：把自然图上的机制迁移回医学域。

数据建议：

1. 先选 1-2 个已有医学小数据集。
2. 每个数据集只取 10-20 张，先做小闭环。
3. 按 target size、contrast、difficulty 分层记录。

对比：

| 方法 | 用途 |
| --- | --- |
| SAM3 text | 检查医学术语 prompt 是否失败。 |
| SAM/SAM3 point | 交互 baseline。 |
| Matcher original | 原始 one-shot baseline。 |
| feedback-guided Matcher | 我们的方法。 |
| MedSAM box | 医学强 baseline。 |
| nnU-Net | fully supervised upper/reference baseline。 |

指标：

Dice、IoU、HD95、interaction count、runtime、failure type。

### P3：Adapter / LoRA 后续扩展

只有当 P0-P2 证明 prompt/matching 机制有价值后，再考虑训练 adapter。否则会过早进入训练泥潭。

## 6. 预期论文叙事

可以形成这样的论文主线：

```text
SAM3 improves concept-level segmentation, but complex text prompts and domain-specific expressions remain unstable.
Matcher can transform reference examples into point prompts, but matched points contain outliers and mix multiple instances.
We propose a feedback-guided, cluster-aware prompt sampling framework that uses a few positive/negative clicks to refine feature matching, samples instance-aware prompts, and filters proposals with instance-level matching.
This creates a low-cost fallback path from unreliable text prompts to controllable visual prompts, especially useful for multi-instance and medical segmentation scenarios.
```

中文版本：

```text
SAM3 将分割从点/框推进到概念文本提示，但复杂文本和医学领域表达仍不稳定。
Matcher 能把 reference example 转成点提示，但 matched points 容易包含 outlier，并且多实例场景会混在一起。
我们提出反馈引导的聚类感知 prompt 采样方法，用少量正负点修正 feature matching，再按实例聚类采样 prompt，并通过 instance-level matching 过滤 mask proposals。
该方法为 SAM3 文本失败时提供低成本、可控的视觉提示回退路径，尤其适用于多实例和医学分割场景。
```

## 7. 推荐执行顺序

| 优先级 | 任务 | 预计产物 |
| --- | --- | --- |
| P0-1 | 导出 Matcher Stage 1 matched points | heatmap、top-k overlay、json/csv |
| P0-2 | 加正负点 feedback，验证 matching 是否改善 | before/after heatmap、top-k precision proxy |
| P0-3 | top N% points 聚类采样，重复送入 SAM3/SAM | m 个 proposals、overlay、score 表 |
| P0-4 | instance-level filtering 和 K 估计 | retained masks、K_pred、threshold 曲线 |
| P1 | 扩展 SAM3 text failure taxonomy | prompt failure 表和 fallback trigger |
| P1.5 | 大图 sliding-window matching | 全图坐标 matched points |
| P2 | 医学小数据验证 | Dice/IoU/交互成本表 |
| P3 | adapter/LoRA | 训练型后续方向 |

## 8. 我建议现在立刻做什么

下一步不要继续扩 text prompt benchmark，也不要马上 fine-tune。最优先应该做：

```text
P0-1：把 Matcher Stage 1 的 top matched points、similarity heatmap、top-k overlay 导出来。
```

原因：

1. 这是老师后续所有问题的起点。
2. 没有 Stage 1 点集，就没法验证 feedback、sampling、clustering、K 估计。
3. 这个任务最小、可视化强、最容易判断方向是否走得通。

最小成功截图应该包含：

1. reference image + support mask/box；
2. target image；
3. target 上 top-100 matched points；
4. feedback 前后 heatmap；
5. top points 聚类结果。

完成 P0-1 后，再推进 P0-2 feedback，这样整个项目会从“老师猜想”变成“可复现方法链条”。
