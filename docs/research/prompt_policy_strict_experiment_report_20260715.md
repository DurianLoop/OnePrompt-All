# Reference-Conditioned Adaptive Prompt Policy：严格实验阶段报告

更新日期：2026-07-15

## 1. 研究问题

给定一张带人工核 mask 的 reference 图，先用 reference 前景与背景在 query 图上计算匹配分数，再由 Prompt Policy 决定 K、背景抑制系数 lambda 和点布局，最后把少量候选 prompt 输入冻结的 SAM。研究目标不是微调 SAM，而是验证：在密集多核场景中，学习的 reference-conditioned prompt policy 能否用 1 或 3 次 SAM 尝试超过固定 K/lambda 策略。

当前第一阶段只研究 MoNuSeg all-nuclei intent，固定 matcher 为 DINOv2-L，固定 segmenter 为 SAM ViT-B。

## 2. 严格数据协议

本地核验的官方 MoNuSeg 包含 37 个 source-train case 和 14 个 source-test case，共 51 个非空 mask。

| 角色 | case 数 | 用途 |
| --- | ---: | --- |
| train | 29 | 训练 reward/policy model，产生 train-only oracle 标签 |
| validation | 8 | 选择固定基线、模型复杂度、候选预算与门控阈值 |
| test | 14 | 冻结后只评估，不参与 K、lambda、reference、模型或阈值选择 |

单 reference 协议使用 `TCGA-FG-A87N-01Z-00-DX1` 作为 support；该 case 只来自 train。初始训练有 28 个 query episode，validation 有 8 个，test 有 14 个。

## 3. 已修正的两个协议问题

### 3.1 test 内选择 Round3 参数的泄漏

早期 runner 会在当前 manifest 上按 Dice 选择 Round3 K/lambda；若 manifest 是 test，这等价于用 test mask 调参。现已加入 `--round3-fixed-k` 与 `--round3-fixed-lambda`，严格 test 固定使用 validation 选出的 K=40、lambda=0.75。早期 test 目录只保留为探索证据，不进入论文结果。

### 3.2 同名 Round3 动作的跨 split 语义漂移

旧 train grid 的 `local_maxima` 使用 train 内选出的 K=75，而 test 的同名动作使用 validation 冻结的 K=40。动作名相同但实际 prompt 数量不同，导致策略训练与测试定义不一致。

当前论文主策略先只使用 Round1/2 中显式编码 K/lambda 的动作。validation 在稳定动作中选择 `contrast_topk_K40_lam0.75` 为固定锚点。Round3 layout 后续必须在所有 split 上重跑同一冻结 K/lambda 后才能重新加入。

## 4. 候选动作与特征

完整单 reference 网格每个 episode 有 46 个候选：

- Round1：8 个 foreground-only K；
- Round2：8 个 K × 4 个 lambda，共 32 个 fg-bg 候选；
- Round3：6 个布局/聚类候选。

策略输入不含 GT、Dice 或 IoU。pre-SAM 特征包括 score map 均值、标准差、最大值、top-quarter 均值、归一化熵，候选点分数/前背景间隔/空间距离，以及 K、lambda 和布局 one-hot。post-SAM 选择器额外读取 SAM 自评分、预测面积、预测连通分量数和 prompt 数量。

监督信号只从 train candidate grid 的 mask 与 GT 计算，包括绝对 Dice、case-centered Dice、case-normalized Dice、top-action 标签，以及候选相对 validation-best anchor 的 Dice gain。

## 5. 策略版本与严格结果

下表均使用同一 29/8/14 case 划分。test 只在 validation 冻结配置后评估。

| 版本 | 决策方式 | Test Dice | 相对固定策略 |
| --- | --- | ---: | ---: |
| fixed stable action | validation-best `K40, lambda=0.75` | 0.515079 | 0 |
| nearest-centroid | matcher 统计 -> 单动作类别 | 0.457698 | -0.057381 |
| ridge reward | pre-SAM 特征回归 Dice，选 1 个 | 0.386593 | -0.128486 |
| ExtraTrees reward | pre-SAM 特征回归 Dice，选 1 个 | 0.474580 | -0.040499 |
| top-3 + post-SAM | pre-SAM 保留 3 个，post-SAM 选 1 个 | 0.503872 | -0.011207 |
| anchored top-3 | 始终保留固定锚点，只在预测 gain 超阈值时切换 | **0.516529** | **+0.001450** |

anchored top-3 的 validation Dice 为 0.527405，固定策略为 0.523353。validation 选择的 gain threshold 为 0.01。

## 6. 配对统计与诚实结论

14 个 test case 中，anchored policy 相对固定策略：

- 2 个 case 提升；
- 11 个 case 回退到固定锚点，结果持平；
- 1 个 case 下降；
- 平均 Dice 增益为 +0.001450；
- 10,000 次 paired bootstrap 的 95% CI 为 [-0.000807, 0.004740]。

因此当前只能说“anchored policy 在严格 test 上首次得到非负且略高的 pilot”，不能声称统计显著，也不能作为论文最终主结论。

三次主动切换分别为：

| Test case | 选择动作 | Delta Dice |
| --- | --- | ---: |
| `TCGA-GL-6846-01A-01-BS1` | K75, lambda=0.75 | +0.020100 |
| `TCGA-HT-8564-01Z-00-DX1` | K75, lambda=0.75 | +0.005850 |
| `TCGA-ZF-A9R5-01A-01-TS1` | K75, lambda=0.25 | -0.005646 |

## 7. 为什么需要多 reference episodic training

单 reference 只有 28 个训练 episode。ExtraTrees 在 train 上可以接近 oracle，但在 validation/test 上明显退化；这说明问题不是树模型容量不够，而是训练 episode 对 reference-query 变化覆盖不足。

新的 multiref5 协议保持 29/8/14 case 划分不变。对每个 train query，从其他 train case 中按 mask density 分层选 5 个 support，构成 29 × 5 = 145 个 train-only episode。validation/test 仍沿用原冻结 reference，因此评价定义不变。

多 reference 初版只保留 15 个稳定 Round2 动作：

- K = {5, 10, 20, 40, 75}；
- lambda = {0.25, 0.75, 1.5}；
- 共 15 个 K/lambda 组合。

该子集完整覆盖 train/val oracle 动作；在已存在的 test grid 中也覆盖每个 case 的最佳 Round2 动作。训练候选生成目录为：

`runs/research_direction/prompt_policy_oracle/train_multiref5_reduced_cpu_dinov2l_samvitb/`

## 8. 下一步验收顺序

1. 完成 145 episode × 15 Round2 actions 的候选表，并核验每个 episode 恰有 15 个 Round2 候选。
2. 按 `episode_id` 生成相对 anchor 的 train-only gain；不得把同一 query 的 5 个 reference 合并成一个动作组。
3. 只在 8-case validation 上选择 ExtraTrees leaf、top-3 budget 和 gain threshold。
4. 在 14-case test 上一次性评估 fixed、single-reference anchored、multiref5 anchored。
5. 输出 paired bootstrap CI、胜负平、切换动作、top-3 oracle、pre-filter regret 和 post-selector regret。
6. 若 multiref5 仍未显著超过固定策略，下一步优先增加训练 reference episode 或采用 query-grouped cross-validation，不继续堆叠任意 proposal score。
7. 只有主方法稳定后，再做 fg-only、去 anchor、去 post-SAM、候选预算 1/3/5、DINOv2-B/L 与 SAM-family transfer 消融。

## 9. 可复现入口

```powershell
# 单 reference 稳定动作策略
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python -B code\benchmarks\train_adaptive_prompt_policy.py `
  --allowed-rounds round1,round2 `
  --out-dir runs\research_direction\prompt_policy\adaptive_policy_models_v6_stable_actions

# 配对统计
python -B code\benchmarks\analyze_prompt_policy_results.py `
  --fixed runs\research_direction\prompt_policy\adaptive_policy_models_v6_stable_actions\test_fixed_predictions.csv `
  --learned runs\research_direction\prompt_policy\adaptive_policy_models_v6_stable_actions\test_anchored_predictions.csv `
  --out-dir runs\research_direction\prompt_policy\adaptive_policy_models_v6_stable_actions\paired_analysis
```

## 10. 当前论文就绪度

目前已经具备可复现协议、严格 train/val/test、多个学习策略负结果、首个 anchored 正向 pilot 和统计分析代码。尚未达到论文主结果就绪，主要缺口是：multiref5 主结果、nested/grouped 稳健性、机制指标、消融、跨 backbone/segmenter 学习迁移，以及最终定性图。
