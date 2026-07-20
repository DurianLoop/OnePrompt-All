# 中文论文工作稿：面向密集多实例分割的参考条件响应门控提示

状态：已具备中文论文写作所需的主方法、严格主结果、消融、外部验证、第二 segmenter 复核与失败边界；投稿前仍需完成 FoB 病理核适配和运行成本表。  
更新：2026-07-16

## 暂定标题

**Response-Gated Reference Prompting for Dense Multi-Instance Medical Image Segmentation**

中文：**面向密集多实例医学图像分割的参考条件响应门控提示**

备选标题：**Learning When to Accept Background Prompts in Reference-Guided Dense Segmentation**

## 摘要

通用提示分割模型能够根据点提示输出高质量局部 mask，但点本身不能说明用户希望分割一个实例、若干实例还是全部同类实例。在细胞核等密集多实例场景中，reference image-mask pair 可以通过前景和背景外观定义任务意图；然而固定数量的背景负点并不总是有利：少量负点可能抑制过分割，过多负点又会扩大预测面积或损害区域重叠。本文提出一种冻结 matcher 与 SAM 的参考条件响应门控方法。首先，从 reference mask 内外分别提取 DINOv2-L 前景与背景原型，在 query 中生成 40 个“像前景且不像背景”的正点；随后比较 validation 选择的安全锚点与一个背景负点 challenger，只在冻结 SAM 的预测面积和内部质量分数表明可能获益时接受 challenger。整个策略只需两次 SAM 响应，不读取测试标注。

在 MoNuSeg 官方 37 个 source-train case 上，我们建立 5-fold nested-CV：每个 case 恰好外测一次，每个 query 使用 5 个当前折 train-only references，共 185 个外测 episode，官方 14-case test 不参与该稳健性分析。前景/背景匹配相对仅前景匹配将 query-level Dice 从 0.4733 提高到 0.5380，增益为 +0.0647，95% CI=[0.0470,0.0841]。在 SAM ViT-B 上，学习门控相对每折 validation 选择的固定锚点将 Dice 从 0.5375 提高到 0.5476，增益为 +0.0101，95% CI=[0.0039,0.0188]；相对全局固定 N5 的增益仍为 +0.0092，95% CI=[0.0004,0.0193]。门控将 fixed-to-oracle gap 缩小 66.1%。相同 N0/N5 gate 在 SAM-Med2D 上为 -0.00033 Dice，表明收益依赖分割器响应。在 patient-disjoint MoNuSAC 上，冻结 MoNuSeg gate 相对 N0 提升 +0.0079 Dice；MoNuSAC 域内训练 gate 在患者级提升 +0.0056，95% CI=[0.0006,0.0113]。但两者均未超过该外部数据集事后观察到的全局固定 N5，说明提示策略具有可迁移的纠错能力，却尚非跨域统一最优预算。结果支持将 reference 前景/背景证据与响应门控结合，同时揭示了 dense all-instance prompting 的模型依赖和域依赖边界。

## 1. 引言

### 1.1 研究问题

SAM 系列在“给定明确局部意图”时已经很强，但其原始接口只有点、框、mask 或文本等有限提示形式。一个正点通常表达“这里是什么”，不能完整表达“请分割所有同类目标”。在 MoNuSeg 一张图中可能存在数百个细胞核：

- 少量正点容易只覆盖局部实例；
- 多个正点 joint 输入可能扩大为连通背景；
- 每点独立分割再合并又会受点预算限制，难以覆盖全部核；
- 背景负点可以抑制扩张，但其数量和位置同样影响结果。

本文将问题从“继续增强 SAM 主干”改写为：给定一张带人工 mask 的 reference，如何把 reference 中的目标和背景证据转换为 query 的正负点，并根据冻结 SAM 对提示的响应判断是否接受背景约束。

### 1.2 核心假设

本文检验四个可证伪假设：

1. reference mask 外的背景特征能够帮助 matcher 排除外观相近的非目标区域；
2. dense all-nuclei 场景不存在对所有 query 都最优的固定负点预算；
3. train-only gate 可以利用无 GT 的 SAM 响应，缩小固定策略与 per-query oracle 的差距；
4. 同一 gate 可以跨数据集纠正部分失败，但最优预算仍可能受领域和 segmenter 控制。

### 1.3 贡献

1. 提出 reference foreground-background matching 与 response-gated negative prompting 的受控组合：固定 matcher、正点数、正点位置规则和 SAM，只学习是否接受背景负点响应。
2. 采用 anchored two-candidate policy。validation 先确定安全锚点，测试时只运行锚点和一个 challenger；gate 预测收益未超过 validation 阈值时回退到锚点。
3. 建立无测试调参的 37-query MoNuSeg nested-CV 和 patient-disjoint MoNuSAC 协议，并按 query 或 patient 而非相关 episode/patch 做统计。
4. 同时报告正结果和边界：背景证据稳定有效；门控在 MoNuSeg 显著缩小 oracle gap；外部域中 gate 能纠正 N0，但未胜过事后最佳固定 N5；固定策略也不对所有 SAM 版本普适。

## 2. 相关工作与创新边界

### 2.1 Reference-guided segmentation

Matcher、One-Prompt、UniverSeg、VRP-SAM 等工作已经说明 support/reference image-mask pair 可以定义新任务。Segment Any Tissue（SAT, MedIA 2025）进一步使用 one-shot reference、特征与物理空间提示、循环 feedback 和逐图参数选择。因此，本文不声称首次使用 reference、首次动态选点或首次将 SAM 用于 one-shot 医学分割。

### 2.2 Automatic prompt optimization

Plug-and-Play PPO、AlignSAM、Point Prompt Defender 和 Instruction-Focus-Prompt 已研究自动点提示生成、强化学习优化或结构提示。本文的区别不是“让模型自动点”，而是把动作压缩为可审计的背景预算决策，并用严格 train-only anchor/gate 协议区分真实泛化和测试集候选 oracle。

### 2.3 Background-centric prompting

FoB（CVPR 2026）是最接近工作。FoB 从 support mask 学习背景 prompt localization，并建模前景-背景空间依赖和背景结构约束。它已经证明背景提示本身不是空白方向。本文可辩护的区别是 dense all-instance pathology、DINO 前景/背景证据、冻结主干后的响应接受/拒绝门控，以及 nested-CV/patient-level 的严格诊断。

FoB 官方代码已固定在 `external/FoB_SAM` commit `40d0e23...`。公开权重仅覆盖 CHAOST2/SABS，未提供 MoNuSeg/MoNuSAC 权重或病理核 loader，故当前不能宣称本文数值优于 FoB。完整审计见 `docs/research/fob_baseline_reproducibility_audit_20260716.md`。

## 3. 方法

### 3.1 问题定义

给定 reference 图像 `R`、reference 二值 mask `M_R` 和 query 图像 `Q`，冻结特征骨干 `F` 与分割器 `S`。策略输出正负点集合 `P`：

```text
Y = S(Q, P)
```

训练 query 的 GT 只用于产生 policy reward；validation 只用于选 anchor、树叶大小和接受阈值；outer-test/official-test GT 不参与动作选择。

### 3.2 前景与背景原型

将 `M_R` 映射到 DINO patch 网格，分别对 mask 内外特征加权平均：

```text
p_fg = normalize(sum_i w_fg(i) F_R(i))
p_bg = normalize(sum_i w_bg(i) F_R(i))
```

query 位置 `x` 的匹配分数为：

```text
s(x) = zscore(cos(F_Q(x), p_fg)) - lambda * zscore(cos(F_Q(x), p_bg))
```

主实验固定 DINOv2-L/14、`lambda=0.75`，取 `s(x)` 最高的 K=40 个 patch 中心作为正点。该设置来自训练/验证阶段的受控筛查，不在 outer-test 上逐例重选。

### 3.3 背景负点候选

从 `s(x)` 最低的位置依次选择背景点，并要求其与正点和已选负点保持至少 18 像素距离。动作空间为：

```text
A = {N0, N5, N10, N20}
```

这里的 N 表示请求预算；小 patch 若受空间排除限制，实际放置数可能小于请求数。CSV 同时保存 `requested_negative_count` 和 `negative_count`，策略动作按请求预算定义，响应特征使用实际数量。

### 3.4 Validation anchor

在每个训练折中，先根据 inner-validation 的平均 Dice 选择固定 action `a0`。这一步模拟部署前选择一个全局默认配置。测试中所有学习决策都相对 `a0` 表达，避免一个不确定的学习器任意替换可靠固定策略。

### 3.5 两阶段响应门控

对每个训练 episode 和 action，计算相对 anchor 的真实训练收益：

```text
Delta r(e,a) = Dice(e,a) - Dice(e,a0)
```

pre-SAM ExtraTrees regressor 使用 matcher 统计和动作差预测收益，从四个预算中选出一个 challenger，并强制保留 anchor。当前控制变量设置下，matcher/正点相同，pre-stage 的有效变化主要是负点数量。

随后只对 anchor 与 challenger 运行 SAM。post-SAM regressor增加：

- `delta_sam_score`：challenger 与 anchor 的 SAM 内部 mask quality score 差；
- `delta_pred_area_pct`：两者预测前景面积占比之差；
- `delta_negative_count`：实际负点数差；
- 其余 matcher/点集差值特征。

两阶段模型均为 200 棵 ExtraTrees 回归树，默认平方误差目标，随机种子 20260715。`min_samples_leaf in {1,2,4,8}` 和接受阈值 `tau in {-0.02,0,0.005,0.01,0.02,0.05}` 只在 validation 上选择：

```text
output = challenger, if predicted_gain > tau
         anchor, otherwise
```

部署预算为两次 SAM，不需要运行四个候选后看 GT 选最优。

### 3.6 为什么需要响应特征

只看点的匹配分数无法预知 SAM 会把这些点合成多大区域。负点加入后可能出现三种响应：

1. 去掉大片假阳性，预测面积下降且 Dice 上升；
2. 几乎不改变 mask，此时保留 anchor/challenger差异很小；
3. 错误抑制真实目标或改变 multimask 选择，Dice 下降。

因此 gate 学的不是“负点一般有没有用”，而是“当前 query 上，SAM 对这组负点的响应是否值得接受”。MoNuSeg nested-CV 的 post-stage 主要依赖预测面积变化和 SAM score 变化，符合这一机制解释。

## 4. 实验协议

### 4.1 MoNuSeg

本地官方完整包为 37 train + 14 test = 51 case。主稳健性实验只在 37 个 source-train 内做 5-fold nested-CV：

- 每折 outer-test：7 或 8 个 case；
- inner-validation：固定 6 个 case；
- inner-train：23 或 24 个 case；
- 每个 query：5 个当前折 inner-train references；
- 共 37 个独立 outer-test query、185 个 reference episodes；
- 五折后每个 case 恰好 outer-test 一次；
- 官方 14-case test 完全不参与 nested-CV。

统计时先对同一 query 的 5 个 references 求平均，再对 37 个 query 做 paired bootstrap 和 sign test，不能把 185 个相关 episode 当作 185 个独立样本。

### 4.2 MoNuSAC

整理后共有 310 个 patch。患者审计发现两个 patient 同时存在于官方 train/test，因此从可训练池排除对应的 9 个 train patches，保留完整 101 个 official-test patches：

- train：154 patches，35 patients；
- validation：46 patches，9 patients；
- test：101 patches，25 patients；
- support patient 必须与 query patient 不同。

报告 patch-level 和 patient-aggregated 结果。患者级 bootstrap 以 25 个 patient 为推断单位。

### 4.3 指标与统计

主指标为 binary Dice。辅助指标包括 IoU、matched-point precision、component coverage、object recall、merge error 和 predicted area。所有成对置信区间使用 10,000 次 bootstrap，seed=20260715；同时报告胜/平/负和 exact sign test。CPU/GPU 一致性审计覆盖 128 次同提示比较，坐标完全相同，最大 Dice 数值差为 1.45e-4，明显小于主效应。

## 5. 结果

### 5.1 前景/背景证据是稳定有效的 matcher 改进

| MoNuSeg nested-CV | Dice | 相对 FG-only | 95% CI | 胜/平/负 |
| --- | ---: | ---: | ---: | ---: |
| FG-only K40 | 0.473325 | - | - | - |
| FG-BG K40, lambda=0.75, N0 | **0.538050** | **+0.064725** | [0.046994, 0.084141] | 36/0/1 |

这说明 reference mask 外区域提供了可重复的负证据。它先改善“点落在哪里”，再由负点 gate 处理“是否接受 SAM 的背景约束响应”。

### 5.2 固定负点越多，实例召回越高，但 Dice 和面积恶化

| 请求负点数 | Dice | Object recall | Pred area (%) |
| ---: | ---: | ---: | ---: |
| 0 | 0.538263 | 0.951922 | 61.90 |
| 5 | **0.538372** | 0.965711 | 63.70 |
| 10 | 0.528942 | 0.968563 | 66.37 |
| 20 | 0.517184 | **0.979520** | 70.79 |

N 增大确实覆盖更多核，但同时使预测前景面积持续增加。dense all-instance intent 不是单纯最大化 object recall，必须控制过分割。

### 5.3 MoNuSeg nested-CV 主结果

| 方法 | Query 数 | Dice | 增益 | 95% CI | 胜/平/负 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 每折 validation fixed anchor | 37 | 0.537458 | - | - | - |
| Learned negative-response gate | 37 | **0.547555** | **+0.010097** | [0.003870, 0.018777] | 15/21/1 |
| 全局固定 N5 | 37 | 0.538372 | - | - | - |
| Gate vs global N5 | 37 | 0.547555 | **+0.009183** | [0.000437, 0.019333] | 14/21/2 |
| Per-query oracle | 37 | 0.552270 | - | - | - |

对每折 anchor 的 exact sign-test `p=0.000519`；对 global N5 的 `p=0.004181`。gate 将 global-N5 到 oracle 的 gap 从 0.013898 缩小到 0.004715，关闭 66.1%。

五折增益分别为 +0.016459、0、+0.009506、+0.015080 和 +0.009974。策略并非靠单一异常 fold 支撑。

### 5.4 Gate 行为审计

185 个 outer-test episodes 中，gate 选择 N5 123 次、N0 62 次。相对 fold anchor 实际发生 49 次切换，136 次回退：

- 切换 episode 平均增益 +0.03812；
- 42 次切换成功，7 次失败；
- 46 次切到 N5，平均增益 +0.02630；
- 3 次切回 N0，平均增益 +0.21946，全部成功。

这说明主要收益来自少量有把握的纠错，而不是全局把 N 改成 5。

### 5.5 MoNuSAC 外部验证

| 方法 | Patch Dice | 相对 N0 | Patch 95% CI | Patient Dice | Patient 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| N0 | 0.528250 | - | - | 0.534951 | - |
| Frozen MoNuSeg gate | **0.536154** | +0.007904 | [0.001635,0.015518] | **0.542568** | [0.001808,0.014621] |
| MoNuSAC within-domain gate | 0.534201 | +0.005951 | [-0.000732,0.013891] | 0.540510 | [0.000625,0.011330] |
| Global fixed N5, post-hoc | **0.544167** | +0.015917 | 未作正式选参 CI | 未聚合 | - |

零样本 gate 和域内 gate 都能纠正 N0 的一部分失败，但域内重训没有超过零样本 gate，二者也没有超过事后观察的 global N5。因此：

- 支持“响应 gate 具有跨域纠错信号”；
- 不支持“每个领域训练一次即可得到该领域最优策略”；
- 不支持“同一个 gate 普适优于所有固定预算”。

### 5.6 跨 segmenter 边界

在早期固定 official-test screening 中，固定 K40/lambda0.75 的 fg-bg 相对 fg-only：

- SAM ViT-B：+0.1153；
- SAM ViT-L：+0.0071；
- SAM ViT-H：-0.0040；
- SAM2.1 Hiera-L：-0.0002。

ViT-L/H 与 SAM2.1 在 dense joint points 下常输出接近整图的 mask。该结果只作机制边界，不作为 nested-CV 主表：当前证据支持 model-dependent prompting，不支持 model-agnostic universal policy。

进一步在 SAM-Med2D 上完成完全相同的 37-query、5-reference nested-CV，只保留 N0/N5 两动作。每折 validation anchor 的 pooled Dice 为 0.414726，learned gate 为 0.414397，delta=-0.000330，95% CI=[-0.000872,0.000024]，2/30/5。五折中四折 gate 完全回退 anchor，仅 fold4 的切换产生轻微负收益。该严格负结果说明 SAM ViT-B 上由 predicted-area/SAM-score 学到的接受规则不能无条件迁移到 SAM-Med2D；未来的跨执行器 policy 必须显式输入 segmenter identity、校准其 quality score，或分别训练。

## 6. 消融与辅助实验

### 6.1 Joint 与 per-point union

3-case K40 smoke 中，joint Dice=0.5296，逐点独立解码再并集 Dice=0.1398。逐点输出往往只对应局部实例，40 个点不足以覆盖数百个核。因此本文使用 joint prompt，不把 per-point union 当作默认方案。

### 6.2 早期宽动作策略的负结果

早期 V7 同时选择 K、lambda 和布局，5-reference 主评估增益接近 0；pairwise V8 进一步显著退化。这些负结果说明小样本下同时学习过多变量会使 prefilter regret 主导。当前方法因此固定 matcher/K/lambda/layout，只学习背景响应是否被接受。

### 6.3 Oracle action 分布

MoNuSeg 37-query oracle 为 N0/N5/N10/N20：16/12/2/7。它证明固定预算存在剩余可利用空间，也说明 N10/N20 对少数 query 仍有价值。当前 gate 最终只稳定使用 N0/N5，未来工作应改进 challenger 预筛，而不是盲目扩大动作表。

### 6.4 推理成本

在 RTX 5090 D v2、同一 1000×1000 query、40 正点和 N0/N5 两候选上重复 20 次 mask decode。以下只统计 segmenter，不含 DINO matching 与磁盘 I/O：

| Segmenter | 参数量 | Model load | Image embedding | 两候选 decode | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| SAM ViT-B | 93.735 M | 1.525 s | 0.261 s | 0.0083 s | 2772 MiB |
| SAM-Med2D | 271.242 M | 3.154 s | 0.272 s | 0.0097 s | 1122 MiB |

模型加载可在服务进程中摊销。相对固定单候选，gate 多一次 prompt decode；由于 image embedding 复用，额外开销约 4–5 ms，而不是再跑一次完整图像编码。完整 matcher 端到端时间需在最终部署实现中另行报告。

## 7. 定性分析

论文图位于：

- 主结果：`runs/research_direction/paper_artifacts_20260715/final_main_results.png`
- MoNuSeg 成功与失败：`.../monuseg_nested_gate_qualitative.png`
- MoNuSAC 成功与失败：`.../monusac_within_domain_gate_qualitative.png`

定性图严格分栏显示 reference image、reference 人工 mask、query+GT、N0 与 N5。绿色为 TP，红色为 FP，蓝色为 FN，黄色为正点，青色叉号为负点。MoNuSeg 成功例中 N5 主要去掉大片红色假阳性；失败例中 N5 同时增加蓝色漏分。MoNuSAC 极端成功例从 Dice 0.100 提高到 0.370，但失败例从 0.498 降到 0.407，直观对应“响应门控仍会误接受”的剩余问题。

## 8. 讨论

### 8.1 为什么 MoNuSeg 有效

MoNuSeg 的主要错误是 dense joint points 引起的背景连接和前景面积扩张。低匹配分数位置作为负点能改变 SAM 的 multimask 响应；预测面积与 SAM score 差正好能检测一部分有利变化。anchor fallback 又限制了错误切换的数量。

### 8.2 为什么 MoNuSAC 没有胜过固定 N5

MoNuSAC patch 更小、核密度和染色分布不同，N5 在 101 个 test patches 上恰好形成较强的全局偏置。域内 validation 只有 46 patches/9 patients，却选择 N0 为 anchor；训练 gate 只能在 N0 附近纠错，无法把全局 anchor 改成事后最优 N5。这是 validation 方差、domain shift 和小样本动作选择共同造成的边界。

### 8.3 与 FoB 的差异和风险

FoB 已经把背景提示做成可训练 localization 网络，并在三个医学数据集展示强结果。本文若仅强调“背景点有用”，创新不足。论文必须把重点放在：dense all-instance intent、响应接受/拒绝、严格 oracle-gap 诊断和跨域失败边界。投稿前应完成 FoB 的 MoNuSeg episodic 适配；若 FoB 显著更强，本文更适合定位为轻量、冻结主干、两次 SAM 的效率与诊断方法。

## 9. 局限性

1. SAM-Med2D 已完成同协议 nested-CV，但 gate 未带来收益；当前正结论仍限于 DINOv2-L + SAM ViT-B。
2. MoNuSeg 只有 37 个可用于 nested-CV 的 source-train case，置信区间虽不跨 0，仍需更多病理数据复核。
3. MoNuSAC 域内训练未优于零样本 gate 或事后固定 N5，不能宣称 universal/domain-optimal policy。
4. 当前只处理 binary all-nuclei intent，没有同时验证“单实例/子集/全部实例”三种显式意图标签。
5. FoB 病理核同协议基线尚未完成，不能宣称 SOTA。
6. policy 的 pre-stage 在控制变量后主要根据 N 形成 challenger，真正的图像自适应主要发生在 post-SAM 响应阶段。

## 10. 投稿前必须补齐

| 优先级 | 项目 | 当前状态 |
| --- | --- | --- |
| P0 | FoB MoNuSeg loader 与至少一折 smoke/成本评估 | 官方代码已审计，待适配 |
| P0 | 冻结最终主表、定性图、统计脚本和运行命令 | 已完成初版 |
| P1 | learned gate 在第二个 segmenter 上训练/验证 | 已完成 SAM-Med2D，结果为非正向 |
| P1 | runtime、显存、参数量和 SAM 调用数表 | segmenter-only 表已完成；完整 DINO 端到端待部署版补充 |
| P1 | 英文稿、图注与参考文献 BibTeX | 中文稿完成后转换 |
| P2 | 更多 patient-disjoint pathology dataset | 未开始 |

## 11. 当前可写结论

本文已经可以开始正式写作，最稳妥的中心结论是：

> Reference 前景/背景证据能显著改善 dense all-nuclei point matching；在此基础上，学习是否接受背景负点引起的 SAM 响应，能够在 MoNuSeg nested-CV 中稳定超过固定预算并关闭约三分之二 oracle gap。该 gate 在 MoNuSAC 中仍能纠正 N0，但未超过外部域的事后最佳固定 N5，说明有效提示策略需要显式处理领域与分割器条件，而不是依赖一个普适固定规则。

该结论有主实验、辅助对照、消融、统计、定性成功/失败和外部负边界共同支撑；目前尚不应写成“SOTA”或“model-agnostic universal prompt generator”。
