# Phase3 顶会升级规划：OCSAM-X

## 0. 当前判断

我们现在已经完成了一个初步闭环：

```text
医学图像数据集
-> 多种 prompt 协议
-> SAM / MedSAM / SAM2 推理
-> 自动 prompt 生成
-> prompt selector
-> uncertainty-aware reliability selector
```

但是，如果目标是尽力冲 MICCAI / BIBM / 甚至更高层级会议，目前方案还不够。

原因很直接：

1. 当前提升幅度还小。
2. 当前 selector 的统计显著性还不够稳。
3. 当前还不能宣称超过近几年强医学分割基线。
4. 当前方法更像“prompt 选择模块”，还不是完整医学 SAM 适配框架。

所以 Phase3 不能停在 `prompt_selector_v3`，需要升级为一个更完整的方法：

```text
OCSAM-X:
Support-Guided Uncertainty-Aware Prompt Adaptation
for Medical Segment Anything Models
```

中文可以暂定为：

```text
面向医学 SAM 的支持集引导不确定性感知自动提示适配框架
```

## 1. 我们真正想解决的问题

通用 SAM / SAM2 / SAM3 在自然图像中很强，但医学图像有几个特殊困难：

1. 目标小  
   例如 IDRID 眼底病灶，病灶区域小、分散、低对比。

2. 多实例密集  
   例如 MonuSeg 细胞核，目标数量多、互相粘连、边界模糊。

3. prompt 质量极其敏感  
   box 稍微偏一点，point 落在错误区域，结果就会明显下降。

4. 医学图像边界本身不确定  
   同一个区域可能专家标注都不完全一致，单次 prompt 输出一个 deterministic mask 不够稳。

5. 医学场景人工 prompt 成本高  
   如果每张图都依赖医生手动画 box/点，就很难规模化。

所以我们的核心问题不是：

```text
如何再复现一个更大的 SAM？
```

而是：

```text
如何让 SAM-family 模型在医学图像中自动生成更可靠的 prompt，
并估计这些 prompt/mask 的可靠性，最后得到更稳定的分割结果？
```

## 2. 当前结果说明了什么

当前 40-case phase2-lite 上的结果是：

| 方法 | Mean Dice |
| --- | ---: |
| fixed `auto_box` | 0.3118 |
| `data_aware_selector_v1` | 0.3234 |
| `prompt_selector_v2` | 0.3263 |
| `prompt_selector_v3` | 0.3279 |
| `oracle_auto_family` | 0.3584 |

这个结果说明：

1. 固定单一 `auto_box` 不是最优。
2. data-aware selector 有正向趋势。
3. reliability selector 能继续提升。
4. uncertainty / mask stability 特征有小幅正收益。
5. oracle 上限还明显更高，说明还有研究空间。

但也必须诚实承认：

```text
当前提升还不能作为强 SOTA claim。
```

`prompt_selector_v3` 相对 fixed `auto_box` 的 paired audit：

```text
mean delta: +0.0160 Dice
95% CI: [-0.0001, +0.0340]
p-value: 0.0689
```

所以当前最稳的说法是：

```text
OCSAM 在医学图像自动 prompt reliability 上显示出正向收益，
但仍需要更强组件和更大规模验证。
```

## 3. 现有方案为什么还差点意思

当前方案主要在已有候选 prompt 里选择：

```text
auto_box
auto_medical_box_v1
auto_medical_point_v1
```

这有价值，但还不够顶会。

顶会更希望看到：

1. 更有医学先验的 prompt 生成。
2. 更有理论/实验支撑的不确定性估计。
3. 更强的跨模型/跨数据域泛化。
4. 不只是选择 prompt，而是能 refinement。
5. 和强医学基线比较，而不是只和 SAM prompt ablation 比。

因此升级方向应该从：

```text
prompt selector
```

升级为：

```text
support-guided prompt generation
+ uncertainty-aware prompt ensemble
+ reliability-based prompt/mask selection
+ lightweight mask refinement
```

## 4. OCSAM-X 总体框架

建议总流程如下：

```text
输入医学图像
  |
  |-- 图像启发式 prompt generator
  |-- support-set guided prompt generator
  |-- prompt perturbation sampler
  |
候选 prompts
  |
SAM / MedSAM / SAM2 / Matcher 推理
  |
候选 masks
  |
uncertainty + stability estimation
  |
mask quality predictor / prompt selector
  |
lightweight mask refinement
  |
最终 mask + confidence + explanation
```

这个框架比当前 selector 更完整，也更像论文方法。

## 5. 组件 A：Support-set Guided Prompt Generation

### 5.1 动机

医学图像往往同类结构之间相似。

例如：

- IDRID 中病灶有典型颜色和位置分布。
- MonuSeg 中细胞核形态和染色模式有相似性。
- 同一数据集内 support image/mask 可以提供先验。

如果只看 query image 本身，很容易产生错误 prompt。

因此可以引入 support set：

```text
query image + support image/mask pool
-> 检索相似 support cases
-> 转移粗候选区域
-> 生成 pseudo mask / pseudo box / pseudo point
-> 调用 SAM-family 模型
```

### 5.2 可实现版本

第一版可以不训练深模型，先做轻量版本：

1. 从同一 dataset 中取 support pool。
2. 用简单图像 embedding 检索相似图：
   - 颜色直方图；
   - DINOv2 embedding，如果本地可用；
   - lesion/nuclei 统计特征。
3. 根据 support mask 的面积、位置、形态，约束 query 的 candidate prompt。
4. 输出：

```text
support_box_v1
support_point_v1
support_similarity
support_case_id
```

### 5.3 价值

这个组件比纯手工 heuristic 更强。

它可以自然连接：

- Matcher：one-shot support image/mask。
- SAM2 memory：跨帧/跨切片 support memory。
- 医学小样本分割：support-guided prompting。

### 5.4 风险

1. support 检索不准会带偏 prompt。
2. IDRID 病灶分散，support mask 迁移可能困难。
3. 如果只用很浅的颜色特征，创新性可能不够。

### 5.5 判断标准

必须回答：

```text
support_box_v1 / support_point_v1 是否超过 image-only auto_box？
```

如果不能超过，也要看它是否和 `auto_box` 互补，能否被 selector 利用。

## 6. 组件 B：Prompt-induced Uncertainty Ensemble

### 6.1 动机

医学边界和 prompt 都不稳定。

单次 prompt 输出一个 mask，无法知道这个 mask 是否可靠。

更合理的是：

```text
对同一候选 prompt 做多次扰动
-> 得到多个 mask
-> 计算 mask variance / stability / uncertainty
-> 用 uncertainty 指导选择或融合
```

### 6.2 当前已有基础

我们已经做了 `prompt_selector_v3`：

- 比较 auto-family masks 之间的一致性；
- 计算 prompt uncertainty；
- 用 mask stability 做 reliability reranking。

结果：

```text
prompt_selector_v2: 0.3263
prompt_selector_v3: 0.3279
```

说明 uncertainty 方向有用，但当前还不够强。

### 6.3 下一版做法

`prompt_selector_v4` 应加入 perturbation-based uncertainty：

1. 对 box 做 jitter：
   - 平移；
   - 扩张；
   - 收缩；
   - 边界扰动。

2. 对 point 做 jitter：
   - 正点轻微移动；
   - 加入/删除部分点；
   - 负点位置扰动。

3. 对每个扰动 prompt 跑 SAM/SAM2。

4. 计算：
   - pairwise mask IoU；
   - mean mask；
   - variance map；
   - entropy map；
   - boundary stability；
   - area stability；
   - SAM score stability。

5. 把这些特征送入 reliability predictor。

### 6.4 价值

这个组件更像论文里的 uncertainty-aware module。

它可以支撑 claim：

```text
我们不仅自动生成 prompt，还估计 prompt-induced uncertainty，
并用不确定性提升医学 SAM 分割可靠性。
```

## 7. 组件 C：Reliability Predictor / Prompt Selector

### 7.1 当前版本

当前已经有：

- `data_aware_selector_v1`
- `prompt_selector_v2`
- `prompt_selector_v3`

v2/v3 都是 leave-one-case-out 验证：

```text
训练折可以用 Dice 学可靠性；
测试 case 不用 GT，只用非 GT 特征选择 prompt。
```

这是合理的验证方式。

### 7.2 下一步升级

需要从 ridge regression 升级为更稳的 scorer：

候选方案：

1. Logistic / regression quality head。
2. Gradient boosting / random forest。
3. Tiny MLP。
4. Pairwise ranking loss。

输入特征：

- dataset/domain；
- model；
- prompt type；
- support similarity；
- candidate area ratio；
- box aspect ratio；
- positive/negative point count；
- SAM score；
- predicted mask area；
- uncertainty；
- stability；
- boundary complexity。

输出：

```text
predicted reliability
selected prompt
confidence
reason
```

### 7.3 顶会感在哪里

如果只是一个规则 selector，创新不够。

如果它是：

```text
support-aware + uncertainty-aware + cross-model calibrated reliability predictor
```

那就更像一个方法模块。

## 8. 组件 D：Lightweight Mask Refinement

### 8.1 为什么需要 refinement

仅靠 prompt selection 可能无法超过强医学模型。

强基线通常会有：

- adapter；
- LoRA；
- mask decoder adaptation；
- refinement head；
- task-specific segmentation network。

如果我们要和近几年强基线比较，只做 prompt selection 可能不够。

### 8.2 低成本版本

先做一个轻量 mask refinement head：

输入：

```text
RGB image
candidate SAM mask
uncertainty map
saliency/candidate map
```

输出：

```text
refined mask
```

模型可以很小：

- shallow U-Net；
- tiny CNN；
- depthwise conv refinement；
- CRF-like post-process baseline。

### 8.3 训练策略

先在 IDRID + MonuSeg 上做小规模训练：

- train/val split；
- no leakage；
- compare with no-refinement；
- report per-dataset gain；
- report failure cases。

### 8.4 风险

1. 数据太少会过拟合。
2. 如果 refinement 用 GT 训练，就从 prompt method 变成 supervised segmentation method，需要重新定义实验公平性。
3. 要和 nnU-Net / U-Net baseline 比，否则别人会质疑为什么不用普通分割网络。

### 8.5 建议

refinement 可以作为 Phase3 后半段，不要一开始就上。

推荐顺序：

```text
support prompt
-> perturbation uncertainty
-> reliability selector
-> refinement head
```

## 9. 组件 E：跨模型组合

当前我们已经看到模型互补性。

固定单模型：

| 方法 | Mean Dice |
| --- | ---: |
| fixed SAM ViT-B auto_box | 0.3170 |
| fixed SAM2 Tiny auto_box | 0.3154 |
| fixed MedSAM auto_box | 0.3032 |

模型组合上限：

| 方法 | Mean Dice |
| --- | ---: |
| oracle best model auto_box | 0.3449 |
| oracle best model v3 selector | 0.4002 |
| portfolio v3 predicted reliability | 0.3538 |

这说明：

```text
SAM / MedSAM / SAM2 之间存在互补性。
```

但是 portfolio v3 的 paired test 还不稳：

```text
mean delta vs fixed SAM ViT-B auto_box: +0.0368
95% CI: [-0.0119, +0.0888]
p-value: 0.1690
```

因此跨模型组合是潜力点，但需要校准。

后续需要：

```text
cross-model reliability calibration
```

否则不同模型的 score 不可直接比较。

## 10. 强基线问题

必须非常重视这个问题。

如果想冲好会议，不能只和：

```text
SAM ViT-B
MedSAM
SAM2 Tiny
```

比较。

还需要对比近几年强医学模型：

1. MedSAM / MedSAM2。
2. SAM-Med2D。
3. Medical SAM Adapter / Med-SA。
4. SAMed / SAMUS / AutoSAMUS 类方法，如果能跑。
5. nnU-Net / U-Net / TransUNet 等任务专用 baseline。
6. Matcher / UniverSeg / one-shot medical segmentation baseline。

当前我们不能 claim：

```text
超过近几年强医学分割基线。
```

当前只能 claim：

```text
提升 SAM-family 自动 prompt reliability。
```

如果要升级 claim，必须完成：

```text
更大数据集 + 强基线复现 + 统计检验
```

## 11. 数据集扩展计划

当前：

- IDRID。
- MonuSeg。

下一步建议扩到至少 3-5 个医学域：

1. 眼底：IDRID / REFUGE。
2. 病理：MonuSeg。
3. 息肉：Kvasir-SEG / CVC-ClinicDB。
4. 皮肤：ISIC。
5. 超声：BUSI。
6. CT/MRI：后续用于 2.5D / SAM2 memory。

最低可接受下一阶段：

```text
100-200 cases
至少 3 个医学域
```

如果想冲更高：

```text
500+ cases
5+ domains
包含至少一个强 supervised baseline
```

## 12. 实验设计

### 12.1 Ablation

必须包括：

| 实验 | 目的 |
| --- | --- |
| fixed auto_box | 基础自动 prompt baseline |
| medical prompt only | 验证医学启发式 prompt |
| support prompt only | 验证 support set 贡献 |
| uncertainty only | 验证 perturbation uncertainty |
| selector v1/v2/v3/v4 | 验证 reliability predictor |
| selector + refinement | 验证最终方法 |
| oracle auto-family | 估计 prompt 上限 |
| oracle model portfolio | 估计模型组合上限 |

### 12.2 指标

不能只看 Dice。

需要：

- Dice。
- IoU。
- Boundary F-score。
- lesion-level recall，尤其 IDRID。
- false positive area ratio。
- prompt count。
- interaction cost。
- failure rate。
- robustness under perturbation。
- calibration error，如果做 confidence。

### 12.3 统计检验

至少要做：

- paired bootstrap CI；
- sign-flip test；
- per-dataset report；
- per-model report；
- failure case gallery。

## 13. 论文故事线

不建议写：

```text
我们提出一个更好的 SAM。
```

建议写：

```text
我们发现医学图像中的 SAM-family 模型主要受 prompt 质量、任务域和不确定性影响。
为此提出 OCSAM-X，一个支持集引导、不确定性感知的自动 prompt 适配框架。
它通过 support-guided prompt generation、prompt-induced uncertainty estimation
和 lightweight refinement，提高医学图像 promptable segmentation 的可靠性和交互效率。
```

更强 claim 可以是：

```text
Structure-aware and uncertainty-aware automatic prompting can close part of the gap between general SAM-family models and medical segmentation requirements without fully retraining a foundation model.
```

## 14. 近期执行路线

### Step 1：Support Prompt v1

实现：

```text
code/benchmarks/support_prompt_generator.py
```

输出：

```text
support_box_v1
support_point_v1
support_similarity
support_case_id
```

目标：

```text
验证 support prompt 是否超过 image-only prompt，或与其互补。
```

### Step 2：Perturbation Uncertainty v1

实现：

```text
code/benchmarks/prompt_uncertainty_sampler.py
```

输出：

```text
stability_score
variance_map
entropy_map
mean_mask
boundary_uncertainty
```

目标：

```text
让 prompt_selector_v4 超过 prompt_selector_v3 的 0.3279。
```

### Step 3：Cross-model Calibration

目标：

```text
让 portfolio_v4 的 predicted reliability 稳定超过 fixed single-model baseline。
```

当前 portfolio v3 还不显著，因此必须校准。

### Step 4：扩数据

至少扩到：

```text
100-200 cases
3 个医学域
```

### Step 5：强基线

优先顺序：

1. MedSAM2 / Medical SAM2。
2. SAM-Med2D。
3. Med-SA / Medical SAM Adapter。
4. nnU-Net / U-Net。
5. Matcher / UniverSeg。

## 15. 我对可行性的判断

### 可行部分

1. 自动 prompt + selector 路线已经跑通。
2. uncertainty / stability 有正收益。
3. 模型组合有明显 oracle 空间。
4. 支持集引导和 Matcher/SAM2 memory 可以自然接上。
5. 不需要一开始训练大 foundation model。

### 风险部分

1. 当前提升幅度小。
2. 当前统计显著性不足。
3. 强基线可能很强，单靠 prompt selector 不够。
4. support prompt 如果检索不准，会引入噪声。
5. refinement head 可能变成普通 supervised segmentation，需要设计公平实验。

### 最值得赌的方向

我认为最值得继续的是：

```text
support-guided prompt generation
+ perturbation-based uncertainty
+ reliability selector
```

如果这三者组合能在 100-200 cases 上显著超过 fixed auto prompt，并接近 oracle auto-family，论文就有比较稳的贡献点。

如果再加 lightweight refinement 能明显提升，就可以进一步冲更高。

## 16. 最终建议

不要把当前工作定位成“小项目”。

但也不要现在就 claim SOTA。

合理定位是：

```text
从 SAM-family 医学 prompt failure analysis 出发，
提出支持集引导和不确定性感知的自动 prompt 适配框架，
系统提升自动 promptable medical segmentation 的可靠性。
```

接下来真正要做的不是继续微调 selector 的小数点，而是：

```text
Support Prompt
Perturbation Uncertainty
Cross-model Calibration
Lightweight Refinement
Strong Baseline Comparison
```

这条路线才有机会从 phase2-lite 走向真正有竞争力的 Phase3。
