# AdaPromptSet Execution Specification

> **用途**：本文件是交给 Codex 的唯一方法、实现、实验与论文生成规范。Codex 应按本文件建立代码仓库、生成实验配置、执行实验、汇总结果，并将图表与正文编译为约 8 页双栏 PDF。
>
> **项目暂定名称**：**AdaPromptSet: Cost-Aware Prompt Set Selection for Reference-Guided Dense Medical Segmentation**
>
> **核心原则**：冻结 matcher 与 segmenter；只学习一个轻量、可审计的 prompt-set utility predictor；所有方法共享相同 reference-query episodes、候选池、动作库、后处理与预算定义。
>
> **重要警告**：本文件中所有带有 `SIMULATED PLACEHOLDER` 标记的数值均为排版和代码联调用的模拟数据，不能出现在最终实验结论中。真实实验完成后必须由脚本自动替换，并在 PDF 中移除占位水印。

## Goal-mode 决策状态（2026-07-17）

当前执行以 `AdaPromptSet_Codex_Response.md` 为最高优先级裁决文件。本规范仍作为长期蓝图和工程细则使用；若与该裁决文件冲突，按以下已锁定边界执行：

- 第一轮只做 `AdaPromptSet-MVP`：独立可复现实验包 + 严格 manifest/audit/log/gate 框架 + 小规模 oracle-gap pilot，不能直接跳到完整论文矩阵。
- 主开发数据为 MoNuSAC，MoNuSeg 作为 sanity/historical check，GlaS 作为第一优先级形态迁移扩展；DDR/IDRID/CPM-17 均不得在 MVP 中驱动调参。
- MVP matcher/segmenter 锁定为 DINOv2-L + SAM-B；SAM-L/H/SAM2/SAM3 只在中心 gate 通过后扩展。
- Gate 失败时先补充预声明的诊断/扩展实验；若 evidence 不支持 adaptive policy，则允许 pivot 到 deterministic action 或 oracle/representation analysis，不得强行包装正结果。
- “0.5 Dice points” 在本项目中按绝对 Dice 的 0.005 解释；所有报告必须同时写清 absolute Dice 和 percentage-point 表达。

## 当前工作区可行性附录（2026-07-17）

本规范在 `D:\SAM` 当前状态下**总体可行，但不能按“全新大工程一次性做完”的方式执行**。必须先复用并核对已有 `FSPL` 资产，完成一个小规模 pilot gate，再决定是否进入完整 AdaPromptSet policy 与论文流水线。

### 本地已具备的基础

- 可复用代码基础：`D:\SAM\FSPL\fspl` 已实现 reference-conditioned prompt-policy 实验的核心相邻模块，包括数据发现、DINOv2/SAM wrapper、candidate pools、fixed rules、utility policy、多参考协议、统计与报告脚本。AdaPromptSet 应优先 fork/迁移这些稳定模块，而不是从零重写。
- 本地可用数据：`D:\SAM\code\datasets\MonuSeg\official`、`D:\SAM\code\datasets\MoNuSAC\processed`、`D:\SAM\code\datasets\IDRID`、`D:\SAM\code\ddr_datasets\DDR-dataset-01\lesion_segmentation`、`D:\SAM\external\research_data\extracted\GlaS`、`D:\SAM\external\research_data\raw\CPM17` 已存在。
- 本地可用权重：`D:\SAM\assets\checkpoints\dinov2_vitl14_pretrain.pth`、`dinov2_vitb14_pretrain.pth`、`sam_vit_b.pth`、`sam_vit_l_0b3195.pth`、`sam_vit_h_4b8939.pth`、`sam2.1_hiera_*.pt` 等已存在。
- 本地运行环境：优先使用 `D:\SAM\conda_envs\sam_gpu\python.exe`，并先执行短测试/小 pilot。

### 当前不应默认成立的假设

- GlaS 与 CPM-17 不在 `D:\SAM\code\datasets` 下，而在 `D:\SAM\external\research_data` 下；GlaS 已解压成完整 85/60/20 image-mask pairs，可作为扩展验证数据接入。CPM-17 当前目录只确认到 `cpm15` 的 15 image + 15 `.mat` labels，现有 `build_cpm17_strict_policy_splits.py` 预期 32 train / 32 test，必须先修复/补齐数据后才能作为正式 gate。
- SAM3 在本工作区属于 Windows 适配式 bring-up，不应作为第一版 AdaPromptSet 的主 segmenter；主线先使用 SAM ViT-B，SAM-L/H 作为复现实验，SAM2/SAM3 作为通过主 gate 后的扩展。
- 已有 FSPL 证据显示：固定 spatial layout/radius-3 prompt rule 在 MoNuSAC 上有稳定正效应，但已有 adaptive selector 分支在 MoNuSAC→IDRiD 外部测试上失败。因此 AdaPromptSet 不能预设“policy 一定优于 fixed rule”；必须由 oracle gap 与 validation gate 决定。
- 最终 8 页双栏论文、跨数据集表、GlaS/CPM-17 transfer 和 SAM2/SAM3 矩阵实验只有在 pilot 和主验证通过后才执行。

### 推荐执行收缩版

第一轮只做 `AdaPromptSet-MVP`：

1. 在 `adapromptset/` 内建立独立包，但复用/迁移 `FSPL` 的候选点、SAM wrapper、fixed protocol、policy 和统计测试。
2. 以 `MoNuSAC processed` 为主开发数据，`MoNuSeg official` 为小规模 sanity/cross-check；GlaS 作为第一优先级形态迁移扩展；DDR/IDRiD 作为眼底病灶域的边界测试；CPM-17 只有在数据补齐并通过 manifest gate 后才进入正式实验。IDRiD 因已有 selector 负结果和测试标签观察历史，只能作为记录性参考，不能继续用来调参或宣布新的外部正结果。
3. 默认 matcher/segmenter 为 DINOv2-L/14 + SAM ViT-B；SAM-L/H 只做固定协议复现；SAM2/SAM3 先不进入 MVP。
4. 先跑 10-20 个 query 的 action-bank/oracle-gap pilot。若 oracle gap 小、oracle action 高度集中，或 policy 无法超过 validation best fixed rule，则按 checklist 的 STOP/PIVOT 分支收缩为 fixed-rule 或 action-oracle 分析论文。
5. 只有 `[GATE-F]` 与 `[GATE-H]` 通过后，才扩展到完整 train action grid、cross-dataset、PDF 论文和提交级结果。

### 文档解释优先级

下文原始规范仍作为长期目标；若它与本附录冲突，以本附录为当前 `D:\SAM` 工作区的执行边界。尤其是数据集范围、segmenter 选择、运行命令和论文主张，都必须先服从本附录。

完整的不确定点、需要用户确认的问题、以及 gate 失败后的预声明扩展实验规则，见 `open_questions_and_gate_extensions.md`。Goal 模式全面展开前应先处理该文档中的确认项。

---

## 0. Codex 必须产出的最终工件

Codex 应在仓库中生成并维护以下工件：

```text
adapromptset/
├── README.md
├── execution.md
├── checklist.md
├── pyproject.toml
├── environment.yml
├── configs/
│   ├── data/
│   │   ├── monuseg.yaml
│   │   ├── glas.yaml
│   │   └── cpm17.yaml
│   ├── matcher/
│   │   ├── dinov2_vitl14.yaml
│   │   └── pathology_encoder.yaml
│   ├── segmenter/
│   │   ├── sam_vit_h.yaml
│   │   └── sam2_hiera_large.yaml
│   ├── actions/
│   │   └── prompt_set_bank_v1.yaml
│   ├── policy/
│   │   └── set_utility_ranker.yaml
│   └── experiments/
│       ├── e0_data_audit.yaml
│       ├── e1_fixed_baselines.yaml
│       ├── e2_oracle_gap.yaml
│       ├── e3_policy_train.yaml
│       ├── e4_main_test.yaml
│       ├── e5_ablation.yaml
│       ├── e6_cross_dataset.yaml
│       ├── e7_reference_sensitivity.yaml
│       └── e8_efficiency.yaml
├── src/adapromptset/
│   ├── data/
│   ├── matching/
│   ├── candidates/
│   ├── proposals/
│   ├── segmenters/
│   ├── policy/
│   ├── training/
│   ├── evaluation/
│   ├── statistics/
│   └── visualization/
├── scripts/
│   ├── audit_data.py
│   ├── build_episodes.py
│   ├── cache_features.py
│   ├── build_action_utility_dataset.py
│   ├── train_policy.py
│   ├── evaluate.py
│   ├── run_bootstrap.py
│   ├── export_tables.py
│   ├── export_figures.py
│   └── build_paper.sh
├── tests/
│   ├── unit/
│   ├── integration/
│   └── leakage/
├── artifacts/
│   ├── manifests/
│   ├── feature_cache/
│   ├── action_utility/
│   ├── checkpoints/
│   ├── predictions/
│   ├── metrics/
│   └── figures/
└── paper/
    ├── main.tex
    ├── references.bib
    ├── sections/
    ├── tables/
    ├── figures/
    ├── generated/
    └── main.pdf
```

最终验收必须至少包括：

1. 可复现的数据审计与 episode manifest；
2. 固定规则 baseline、GF-SAM-style baseline、SynPo-style baseline；
3. 冻结动作库上的 oracle 分析；
4. 训练完成的 set-level utility policy；
5. 测试集统计、置信区间、成本分析和定性图；
6. 可由单一命令编译的约 8 页双栏 PDF；
7. 完整更新并勾选的 `checklist.md`。

---

# 1. 科学定位：与相似研究的区别

## 1.1 任务定义

给定一张带标注的 reference/support image-mask pair

\[
(R, M_R),
\]

以及一张未标注 query image

\[
Q,
\]

目标是在不微调 matcher 和 segmentation foundation model 的前提下，分割 query 中由 reference mask 所定义的同类目标。

传统流程通常是：

```text
reference mask
→ foreground feature extraction
→ reference-query matching map
→ fixed Top-K / fixed negative rule / fixed clustering
→ SAM or SAM2
→ mask fusion
```

本研究不把重点放在发明新的 matcher，也不微调 SAM。研究对象是 matcher 与 segmenter 之间长期由人工启发式承担的决策层：

```text
matching evidence
→ 选择哪些正点
→ 是否需要负点及选择哪些负点
→ 哪些点应组成同一个 prompt group
→ 允许调用多少次 segmenter
→ 输出完整 structured prompt set
```

## 1.2 四篇最相似工作的准确边界

### 1.2.1 GF-SAM / Bridge the Points

**正式出处**：Anqi Zhang et al., *Bridge the Points: Graph-based Few-shot Segment Anything Semantically*, NeurIPS 2024. 论文主页与摘要：<https://proceedings.neurips.cc/paper_files/paper/2024/hash/3a2ef31a1e45908901adc0ca853a8faf-Abstract-Conference.html>

GF-SAM 已经解决：

- reference foreground/background matching；
- 动态选择候选点；
- 将 point-mask pairs 建图；
- 根据某个 SAM mask 是否覆盖其他候选点构建有向边；
- 使用弱连通分量获得自然 point-mask clusters；
- 使用 positive/overshooting gating 过滤错误 masks；
- 减少固定聚类参数和冗余 mask generation。

GF-SAM 没有解决：

- 从多个完整 prompt-set actions 中学习 query-specific utility；
- 显式联合优化 SAM 正点与 SAM 负点；
- 在统一的调用预算下预测哪种 prompt grouping 最值得执行；
- 学习成本—准确率 Pareto 决策；
- 通过 oracle gap 分析证明固定规则存在可利用的 query-level action heterogeneity。

**与本研究的关系**：GF-SAM 是最重要的结构化 grouping baseline。本研究不能再声称“现有方法都使用固定聚类”，而应声称 GF-SAM 仍是一个确定性 graph rule；本研究学习在包含 GF-SAM-style proposal 的统一动作库中，按 query 和预算选择完整 prompt set。

### 1.2.2 SynPo

**正式出处**：Yufei Liu et al., *SynPo: Boosting Training-Free Few-Shot Medical Segmentation via High-Quality Negative Prompts*, MICCAI 2025. 论文页面：<https://papers.miccai.org/miccai-2025/0891-Paper1376.html>

SynPo 已经解决：

- 医学 few-shot segmentation 中的 reference-query prompt generation；
- 融合 DINOv2 语义特征与 SAM-ViT 空间特征形成 confidence synergy map；
- 采用 Top-K 候选作为正点集合；
- 对 background confidence 建模，避免把明显图像背景作为无效负点；
- 对正负候选分别使用 K-means；
- 使用 morphology 与 SAM 进行 noise-aware refinement。

SynPo 没有解决：

- 正点数量和负点数量仍由固定超参数指定；
- 每张 query 是否需要负点仍由统一规则决定；
- K-means 与 refinement 是固定流程；
- 不学习完整 prompt set 的边际分割价值；
- 不在严格等调用预算下做 query-specific action selection。

**与本研究的关系**：SynPo 是“高质量医学负点规则”的最强直接 baseline。本研究不能把贡献写成“第一次在医学 segmentation 中使用负点”，而应检验：负点的数量、位置及其与正点和 grouping 的组合是否应按 episode 学习。

### 1.2.3 AlignSAM

**正式出处**：Duojun Huang et al., *AlignSAM: Aligning Segment Anything Model to Open Context via Reinforcement Learning*, CVPR 2024. 官方页面：<https://openaccess.thecvf.com/content/CVPR2024/html/Huang_AlignSAM_Aligning_Segment_Anything_Model_to_Open_Context_via_Reinforcement_CVPR_2024_paper.html>

AlignSAM 已经解决：

- 将自动 prompting 表述为 sequential decision-making；
- 使用 patch-center action space；
- 将 SAM image embedding 与上一轮 mask prediction 组合为 state；
- 使用 actor-critic / PPO 式强化学习迭代选择 prompt location；
- 通过 semantic recalibration 模块判断所选点的正负标签；
- 冻结 SAM 主干并在多个开放语义任务中自适应 prompting。

AlignSAM 没有解决：

- 它不是标准的 labeled-reference/query matching 问题；
- action 不由 reference matching evidence 限定；
- 主 reward 更侧重所选位置是否属于目标，而不是完整 prompt set 的实际成本约束效用；
- 不显式解决密集多实例 prompt grouping；
- 不以 oracle action bank 和 Dice–cost Pareto 为核心分析。

**与本研究的关系**：本研究不能声称“第一次学习 prompt policy”。为避免与 AlignSAM 正面重叠，第一版采用 one-shot、offline、set-level utility ranking，而不是 sequential RL。

### 1.2.4 Point Prompt Defender

**正式出处**：Xueyu Liu et al., *Attack for Defense: Adversarial Agents for Point Prompt Optimization Empowering Segment Anything Model*, CVPR 2026. 官方页面：<https://openaccess.thecvf.com/content/CVPR2026/html/Liu_Attack_for_Defense_Adversarial_Agents_for_Point_Prompt_Optimization_Empowering_CVPR_2026_paper.html>

Point Prompt Defender 已经解决：

- 将 image patches 表示为同时包含物理距离和语义距离的 dual-space graph；
- attacker 激活破坏性 prompt；
- defender 抑制破坏性 prompt；
- 使用 DQN 和基于 segmentation quality variation 的 reward；
- 推理时使用 defender 优化任意 coarse prompt set。

Point Prompt Defender 没有解决：

- 它主要优化已经存在的 coarse prompt set，而不是由 reference evidence 从零构造 structured prompt proposals；
- 不联合决定 prompt group 数与 segmenter call budget；
- 不针对 reference-guided dense medical segmentation；
- 不把 fixed-rule、SynPo-style、GF-SAM-style proposals 放入同一可审计 action bank 比较。

**与本研究的关系**：本研究借鉴“prompt utility 应由真实 segmentation outcome 定义”的思想，但不用 adversarial DQN。训练目标是离线运行完整 actions 后得到的真实 utility ranking。

## 1.3 本研究必须使用的收紧表述

### 允许使用的核心主张

> We formulate reference-guided prompt generation as a budget-constrained structured prompt-set selection problem. A lightweight policy predicts the segmentation utility of complete positive/negative prompt groups generated from frozen matching evidence, without fine-tuning either the matcher or the segmenter.

中文解释：

> 本研究把 reference-guided prompt generation 定义为一个受预算约束的结构化 Prompt Set 选择问题。策略不任意生成坐标，而是在统一、冻结、可审计的完整动作库中，预测由正点、负点和多组 prompt groups 构成的动作在当前 query 上的分割效用。

### 不得在没有实验证据时使用的主张

- 不得写“first to learn prompts for SAM”；
- 不得写“all previous approaches use fixed Top-K or clustering”；
- 不得写“end-to-end differentiable”；
- 不得写“model-agnostic”或“universally transferable”；
- 不得写“plug-and-play across all segmenters”；
- 不得写“state of the art”，除非所有可比 baseline 在同一协议下被复现且统计显著；
- 不得把使用 GT 生成的中心点或框与自动方法视为同等输入条件。

## 1.4 研究问题

本文只围绕以下三个主研究问题展开：

### RQ1：固定 prompt 规则是否存在足够大的 query-specific oracle gap？

在冻结 action bank 中，不同 query 的最优 action 是否显著不同？最强固定动作与每个 query 的 oracle action 之间是否存在足够大的性能差距？

### RQ2：学习策略能否在相同预算下追回 oracle gap？

在相同 segmenter-call budget 和候选空间下，policy 是否稳定优于 validation-selected best fixed rule，并追回可观的 oracle gap？

### RQ3：策略在什么条件下改变 prompt allocation？

匹配图熵、前景—背景 margin、候选密度、目标数量和 reference-query gap，是否能够解释策略选择更多正点、加入负点或拆分多个 prompt groups 的行为？

## 1.5 可证伪假设

- **H1：Action heterogeneity**。在测试 query 中，不存在单一 fixed action 能成为超过 70% query 的 oracle action。
- **H2：Oracle gap**。在主预算下，oracle action 至少比 validation-selected best fixed action 高 2.0 Dice points，或在 AJI 上高 1.5 points。
- **H3：Policy value**。policy 在主预算下至少优于 best fixed action 1.0 Dice point，且 paired bootstrap 95% CI 不跨 0。
- **H4：Oracle recovery**。policy 至少追回 25% 的可用 oracle gap。
- **H5：Cost adaptivity**。在简单 query 上 policy 使用更少 calls，在高熵/高密度 query 上使用更多 groups；这种差异在控制目标数量和图像尺寸后仍显著。

若 H1 和 H2 均不成立，应停止扩展 policy，并转向重新设计 action bank 或缩小为 deterministic cost-aware prompting 分析。详细分支见 `checklist.md`。

---

# 2. 方法总览

## 2.1 一句话方法

冻结 reference-query matcher 和 promptable segmenter；由确定性模块产生有限的 positive/negative candidates 与完整 prompt-set proposals；离线运行每个 proposal 得到真实 segmentation utility；训练一个轻量 set-level ranker，在给定预算下为新 query 选择预测效用最高的完整 structured prompt set。

## 2.2 论文 Figure 1：总览示意图

Codex 应同时实现：

1. 下面的文本/ASCII 图，便于 README 阅读；
2. `paper/figures/method_overview.tex` 中的 TikZ 双栏宽度矢量图；
3. `artifacts/figures/method_overview.pdf`，由 LaTeX 单独编译后供论文引用。

```text
Reference image R + mask M_R                    Query image Q
              │                                       │
              └────────── Frozen matcher F ───────────┘
                                  │
                   foreground/background evidence
                s_fg, s_bg, s_ctr, entropy, gap features
                                  │
              ┌───────────────────┴───────────────────┐
              │                                       │
       positive candidates C+                 negative candidates C-
              │                                       │
              └──────── deterministic proposal builders ────────┐
                                                                 │
             complete structured prompt-set proposal bank A(Q,R)
        {Top-K, pos/neg, local maxima, KMeans groups, graph groups, ...}
                                 │
                Cost-conditioned set utility predictor f_theta
                                 │
              score every legal proposal under call budget B
                                 │
                     A* = argmax f_theta(A_i | R,Q,B)
                                 │
                prompt groups G_1, ..., G_n; one call per group
                                 │
                       Frozen SAM / SAM2 segmenter
                                 │
                  shared mask filtering and instance fusion
                                 │
                         final prediction Y_hat
```

TikZ 图必须通过颜色或线型区分：

- 灰色：冻结模块；
- 蓝色：确定性、无训练模块；
- 橙色：唯一可学习模块；
- 虚线：只在训练阶段存在的 GT utility supervision；
- 实线：训练和测试均存在的数据流。

不要把 GT 画入测试时输入路径。

## 2.3 符号定义

| 符号 | 含义 |
|---|---|
| \(R\) | reference image |
| \(M_R\) | reference binary or instance mask |
| \(Q\) | query image |
| \(Y\) | query ground truth，仅 train utility generation 与 evaluation 使用 |
| \(F\) | frozen matcher / feature encoder |
| \(S\) | frozen promptable segmenter |
| \(C^+\), \(C^-\) | 正点与负点候选池 |
| \(A_i\) | 第 \(i\) 个完整 prompt-set proposal/action |
| \(G_{i,g}\) | action \(A_i\) 中第 \(g\) 个 prompt group |
| \(P^+_{i,g}\), \(P^-_{i,g}\) | group 内正点、负点集合 |
| \(B\) | 最大 segmenter-call budget |
| \(c(A_i)\) | action 的实际调用成本 |
| \(f_\theta\) | 可学习的 set-level utility predictor |
| \(U(A_i)\) | 离线执行后得到的真实 action utility |

## 2.4 冻结的 matching evidence

### 2.4.1 特征提取

将 reference 和 query resize/pad 到统一推理尺寸，但保留原始坐标映射：

\[
F_R = F(R) \in \mathbb{R}^{H_f\times W_f\times d},\qquad
F_Q = F(Q) \in \mathbb{R}^{H_f\times W_f\times d}.
\]

所有特征逐位置 L2 normalize：

\[
\bar F(x)=\frac{F(x)}{\|F(x)\|_2+\epsilon}.
\]

### 2.4.2 单 prototype 主版本

将 reference mask 下采样到 feature grid，得到 \(\widetilde M_R\)。定义前景与背景 prototype：

\[
p_{fg}=\operatorname{norm}\left(
\frac{\sum_x \widetilde M_R(x)F_R(x)}{\sum_x\widetilde M_R(x)+\epsilon}
\right),
\]

\[
p_{bg}=\operatorname{norm}\left(
\frac{\sum_x (1-\widetilde M_R(x))F_R(x)}{\sum_x(1-\widetilde M_R(x))+\epsilon}
\right).
\]

query 上的前景和背景相似图：

\[
s_{fg}(x)=\langle \bar F_Q(x),p_{fg}\rangle,
\qquad
s_{bg}(x)=\langle \bar F_Q(x),p_{bg}\rangle.
\]

使用 robust normalization，而非依赖极值：

\[
\operatorname{rnorm}(s)=
\operatorname{clip}\left(
\frac{s-q_{0.05}(s)}{q_{0.95}(s)-q_{0.05}(s)+\epsilon},0,1
\right).
\]

contrast evidence：

\[
s_{ctr}(x)=\operatorname{rnorm}(s_{fg})(x)
-\lambda_{bg}\operatorname{rnorm}(s_{bg})(x).
\]

默认 \(\lambda_{bg}=0.5\)，只允许在 validation 上选择自 `{0.25, 0.5, 0.75, 1.0}`。test 不得重新选择。

### 2.4.3 可选 multi-prototype 消融

仅作为消融，不进入第一版默认主线。将 reference foreground/background 特征分别聚成 \(K_{fg}\)、\(K_{bg}\) 个 prototypes：

\[
s_{fg}^{multi}(x)=\max_k\langle \bar F_Q(x),p_{fg}^{(k)}\rangle,
\qquad
s_{bg}^{multi}(x)=\max_k\langle \bar F_Q(x),p_{bg}^{(k)}\rangle.
\]

该消融用于检验单一背景 prototype 是否因病理背景异质性而受限。

### 2.4.4 Episode-level global features

utility predictor 不直接读取 GT。它可读取以下由 reference、query 和 matching maps 计算的全局特征：

- \(s_{fg},s_{bg},s_{ctr}\) 的 mean/std/quantiles；
- matching entropy；
- top-1 与 top-5 margin；
- 高分区域面积比例；
- local maxima 数量；
- candidate spatial dispersion；
- reference/query feature mean distance；
- reference mask area ratio、perimeter/area ratio；
- 可选 image-size token、dataset token；
- **默认不使用 segmenter token 与 dataset token作为主模型输入**，以避免学习数据集捷径。它们仅用于消融。

matching entropy 可定义为：

\[
p(x)=\frac{\exp(s_{ctr}(x)/\tau)}{\sum_{x'}\exp(s_{ctr}(x')/\tau)},
\qquad
H_{match}=-\frac{1}{\log N}\sum_x p(x)\log p(x).
\]

## 2.5 候选池

### 2.5.1 正点候选 \(C^+\)

正点候选为以下集合的并集并进行坐标去重：

1. `global_top`: \(s_{ctr}\) 最高的前 \(M_{top}\) 个位置；
2. `local_max`: 经过 max-pooling NMS 后的局部峰值；
3. `coverage`: 将高分连通区域分格后，各格最高分位置；
4. `fg_only`: \(s_{fg}\) 高且 \(s_{bg}\) 不高的位置。

默认：

```yaml
positive_pool:
  max_size: 32
  global_top: 16
  local_max: 16
  nms_radius_feature_cells: 2
  min_contrast_quantile: 0.70
  coordinate_dedup_radius_px: 8
```

每个候选保存：

\[
\phi^+(c)=
[x/W,y/H,s_{fg},s_{bg},s_{ctr},r_{peak},d_{nearest},\rho_{local},h_{local}].
\]

其中 \(r_{peak}\) 为局部峰强度，\(\rho_{local}\) 为邻域候选密度，\(h_{local}\) 为局部 matching entropy。

### 2.5.2 负点候选 \(C^-\)

负点必须是对 segmenter 有潜在辨别价值的 hard background，而不是图像外部或纯黑背景。候选来源：

1. `high_bg`: \(s_{bg}\) 高；
2. `negative_contrast`: \(s_{ctr}<0\) 但非极端低纹理区域；
3. `ambiguous_band`: \(s_{fg}\) 与 \(s_{bg}\) 都较高，margin 小；
4. `positive_ring`: 正点候选周围一定距离环带中的高背景匹配点；
5. 可选 `anatomical_validity`: 使用图像有效区域 mask 排除 padding、空白和玻片纯背景。

默认：

```yaml
negative_pool:
  max_size: 32
  high_bg: 12
  ambiguous_band: 12
  positive_ring: 8
  min_distance_to_positive_px: 12
  max_distance_to_positive_px: 96
  exclude_low_texture_quantile: 0.10
```

每个负点保存：

\[
\phi^-(c)=
[x/W,y/H,s_{fg},s_{bg},s_{ctr},m_{amb},d_{pos},t_{local},v_{valid}].
\]

### 2.5.3 候选池一致性规则

- 所有 baseline 与 policy 必须从同一 `C+`、`C-` 取点；
- 任何 baseline 都不能使用 policy 看不到的额外坐标；
- GT center/box 只能作为 privileged upper bound，不能进入自动方法候选池；
- action bank 冻结后不得根据 test 结果新增动作；
- 生成候选时不得运行 query GT；
- 第一版 one-shot policy 不使用初始 SAM mask 生成负点。

## 2.6 Structured Prompt Set

一个完整 action 定义为：

\[
A_i=\left\{G_{i,1},\ldots,G_{i,n_i}\right\},
\]

其中每个 group：

\[
G_{i,g}=\left(P^+_{i,g},P^-_{i,g}\right).
\]

语义必须固定：

- 一个 group 对应一次 SAM/SAM2 调用；
- 同一 group 中的多个正点表示同一目标区域或同一 mask request；
- group-specific negative points 用于抑制该 mask request 的局部混淆背景；
- 不同 groups 的输出经过共享 fusion 形成最终 semantic/instance prediction；
- action cost：

\[
c_{call}(A_i)=n_i,
\qquad
c_{point}(A_i)=\sum_g(|P^+_{i,g}|+|P^-_{i,g}|).
\]

主实验以 `SAM calls` 作为硬预算，点数作为附加报告。原因是 segmenter image embedding 可缓存，但每个 mask decoding request 仍产生可观成本；实际 runtime 仍应测量。

## 2.7 冻结 Prompt-set Proposal Bank

不要使用独立的 positive head、negative head、layout head 和 stop head。第一版策略只对**完整 actions**评分。这样 action semantics、离线 supervision 和测试推理完全一致。

### 2.7.1 动作家族

Codex 应实现 `prompt_set_bank_v1.yaml`，默认包含以下家族。每个动作必须是确定性的，给定同一 episode 和 config 应完全复现。

#### Family A：Single-group positive-only

- `A_top1_one_group`
- `A_top3_one_group`
- `A_top5_one_group`
- `A_top10_one_group`
- `A_localmax3_one_group`
- `A_localmax5_one_group`

所有点一次输入，cost = 1 call。

#### Family B：Single-group positive + negative

- `B_top3_pos1neg`
- `B_top3_pos3neg`
- `B_top5_pos1neg`
- `B_top5_pos3neg`
- `B_synpo_style_3p3n`
- `B_synpo_style_5p5n`

SynPo-style 使用同一候选池中的 Gaussian-band/hard-background 负点，并分别 K-means 到固定数量。

#### Family C：Independent positive groups

- `C_top3_independent`
- `C_top5_independent`
- `C_localmax3_independent`
- `C_localmax5_independent`

每个正点单独调用一次 segmenter；每组无负点。

#### Family D：KMeans grouped prompts

- `D_kmeans2_groups`
- `D_kmeans3_groups`
- `D_kmeans5_groups`
- `D_kmeans3_groups_localneg`
- `D_kmeans5_groups_localneg`

先从统一高分 candidates 选固定候选，再按坐标 K-means。每个 cluster 选 medoid 或最高分点作为正点；local negative 从 cluster 外邻近 hard-background 中选择。

#### Family E：Density / local-max grouping

- `E_dbscan_groups`
- `E_meanshift_groups`
- `E_connected_peak_groups`

聚类参数只能来自 validation-frozen config。若动作产生 0 groups、超预算或非法 group，应被标记 invalid，不得测试时临时回退到更优动作。可以使用预注册的固定 fallback，但必须计入该动作定义。

#### Family F：GF-SAM-style graph groups

- `F_graph_groups_basic`
- `F_graph_groups_gated`

实现一个同协议、可复现的 GF-SAM-style baseline：

1. 根据 foreground/background alignment 选点；
2. 每个候选点单独运行 segmenter 得到 mask；
3. 若 mask \(m_i\) 覆盖点 \(p_j\)，建立有向边；
4. 弱连通分量形成 groups；
5. basic 版本直接融合；gated 版本加入 positive/overshooting filtering。

注意：该家族需要先调用 segmenter 才能建立 group，因此其真实 calls 必须全部计入。若 graph grouping 的调用已产生最终可用 masks，不得再次无理由重复调用。

#### Family G：Safe anchors

- `G_top1_safe`
- `G_top3_safe`
- `G_empty_or_abstain`（仅研究性消融；若 segmenter 不允许空 prompt，则输出全背景并 cost=0）

safe anchor 用于 utility normalization 和失败回退，但是否允许 abstention 必须在主实验前预注册。

### 2.7.2 动作合法性

动作 \(A_i\) 在预算 \(B\) 下合法，当且仅当：

\[
\operatorname{valid}(A_i)=1,
\qquad
c_{call}(A_i)\le B,
\qquad
|P^+_{i,g}|\ge 1\ \forall g.
\]

负点数允许为 0。

### 2.7.3 Action feature

每个 action 保存：

- action family one-hot；
- group count；
- total positive/negative points；
- per-group point count mean/std/max；
- positive confidence mean/min/max；
- negative hardness mean；
- group spatial coverage；
- inter-group separation；
- intra-group dispersion；
- estimated overlap between groups；
- action-specific deterministic runtime estimate；
- actual segmenter calls；
- invalid reason code。

## 2.8 Segmenter execution and mask fusion

### 2.8.1 Segmenter cache

对同一 query：

- image encoder embedding 只计算一次并缓存；
- 每个 group 单独执行 prompt encoder + mask decoder；
- SAM/SAM2 的 multimask output 使用 validation-frozen 选择规则：默认取 predicted IoU 最大的 mask；
- 不得使用 GT 选择 multimask candidate。

### 2.8.2 Group output

对 group \(G_g\)：

\[
m_g=S(Q,P_g^+,P_g^-).
\]

记录：

- predicted IoU/confidence；
- mask area；
- overlap with other group masks；
- decoding latency。

### 2.8.3 共享 mask fusion

所有自动方法使用完全相同 fusion：

1. 移除面积小于 validation-frozen `min_area` 的 masks；
2. 以 segmenter predicted quality 排序；
3. 使用 mask IoU NMS 去除重复 masks；
4. semantic output 为保留 masks 的 union；
5. instance output 保留 mask IDs，并使用 overlap arbitration；
6. 若统一采用 watershed 将 semantic mask 转 instance，则所有方法共享同一后处理参数。

主文需同时报告：

- raw semantic union 的 Dice/IoU；
- 统一 instance postprocess 后的 AJI/object recall/merge rate。

不得为某个方法单独调后处理。

## 2.9 Cost-conditioned Set Utility Predictor

### 2.9.1 输入编码

策略由三部分组成：

1. **Episode encoder** \(E_{epi}\)：编码全局 matching statistics；
2. **Candidate set encoder** \(E_{cand}\)：使用 DeepSets 或 Set Transformer 编码 \(C^+\)、\(C^-\)；
3. **Action encoder** \(E_{act}\)：编码完整 action statistics 与 group structure。

推荐第一版使用 DeepSets，以减少复杂度：

\[
z_+=\rho_+\left(\sum_{c\in C^+}\phi_+(c)\right),
\qquad
z_-=\rho_-\left(\sum_{c\in C^-}\phi_-(c)\right).
\]

动作中每个 group 的编码：

\[
h_g=\psi_g\left(
\operatorname{pool}_{p\in P_g^+}\phi^+(p),
\operatorname{pool}_{n\in P_g^-}\phi^-(n)
\right),
\]

\[
z_A=\rho_A\left(\sum_{g=1}^{n_A}h_g,\phi_{global}(A)\right).
\]

最终分数：

\[
\hat u_i=f_\theta([z_{epi},z_+,z_-,z_{A_i},e(B)]).
\]

其中 \(e(B)\) 为预算 embedding。主模型参数量目标小于 2M。

### 2.9.2 真实 action utility

主设置将成本作为硬约束，真实 utility 先简单定义为 validation-selected 主分割指标：

\[
U(A_i)=Dice(A_i).
\]

对于 nuclei 主实验，也训练/分析 AJI utility：

\[
U_{AJI}(A_i)=AJI(A_i).
\]

不要在主版本中把大量人工权重混成单一 reward。可选多目标消融：

\[
U_{multi}(A_i)=Dice(A_i)+\eta AJI(A_i)-\gamma OverSeg(A_i),
\]

但 \(\eta,\gamma\) 只能由 validation 选择，并必须做敏感性分析。

### 2.9.3 训练损失

主损失使用 pairwise ranking。对同一 episode 中满足相同预算的 action pairs \((i,j)\)：

\[
y_{ij}=\operatorname{sign}(U_i-U_j),
\]

\[
\mathcal L_{rank}=\log\left(1+\exp[-y_{ij}(\hat u_i-\hat u_j)]\right).
\]

只采样 utility 差异大于 \(\delta\) 的 pairs，避免把数值噪声当作排序标签。默认 \(\delta=0.005\)，在 validation 上从 `{0, 0.0025, 0.005, 0.01}` 选择。

加入辅助回归和 top-action classification：

\[
\mathcal L_{reg}=\operatorname{Huber}(\hat u_i,U_i),
\]

\[
\mathcal L_{top}=CE(\hat p(a^*),a^*).
\]

总损失：

\[
\mathcal L=\mathcal L_{rank}
+\lambda_{reg}\mathcal L_{reg}
+\lambda_{top}\mathcal L_{top}.
\]

默认 \(\lambda_{reg}=0.2\)，\(\lambda_{top}=0.1\)。必须通过 validation 消融确认 ranking 是主要贡献。

### 2.9.4 Utility normalization

为降低不同 episode 难度尺度差异，可使用相对 safe anchor 的 improvement：

\[
\Delta U_i=U(A_i)-U(A_{anchor}).
\]

主实验比较 absolute utility 与 relative utility 两种标签。若 relative label 更稳定，则作为默认，但论文必须报告 anchor 定义。

### 2.9.5 推理

给定预算 \(B\)，在所有合法 actions 中选择：

\[
A^*=\arg\max_{A_i:\,c(A_i)\le B}\hat u_i.
\]

若所有非 anchor 动作 invalid，则选择 safe anchor。不得根据 segmenter 输出在线搜索多个 actions，因为那会增加未计入的 test-time calls。

## 2.10 训练流程

### 2.10.1 Episode 构造

每个 episode 为：

```json
{
  "dataset": "monuseg",
  "reference_case_id": "...",
  "reference_image_id": "...",
  "reference_mask_id": "...",
  "query_case_id": "...",
  "query_image_id": "...",
  "class_or_structure_id": "nucleus",
  "seed": 0,
  "split": "train"
}
```

约束：

- reference 与 query 不得来自同一 patient/case；
- reference 只来自当前允许的 train split；
- validation/test query 的 GT 不参与候选生成或模型输入；
- 每个 query 的多个 reference episodes 在统计时先按预注册规则聚合，不能当作独立测试样本。

### 2.10.2 离线 action-utility 数据生成

伪代码：

```python
for episode in train_episodes:
    R, M_R, Q, Y = load_episode(episode)
    evidence = matcher.compute(R, M_R, Q)          # F frozen
    c_pos, c_neg = build_candidate_pools(evidence)
    actions = proposal_bank.build(c_pos, c_neg, evidence)
    query_embedding = segmenter.encode_image(Q)    # cache once

    records = []
    for action in actions:
        if not action.is_valid:
            continue
        pred, execution_meta = execute_action(
            segmenter=segmenter,
            query_embedding=query_embedding,
            action=action,
            shared_fusion=shared_fusion,
        )
        metrics = evaluate_prediction(pred, Y)
        records.append({
            "episode_id": episode.id,
            "action_id": action.id,
            "action_features": action.features,
            "episode_features": evidence.global_features,
            "dice": metrics.dice,
            "iou": metrics.iou,
            "aji": metrics.aji,
            "object_recall": metrics.object_recall,
            "merge_rate": metrics.merge_rate,
            "calls": execution_meta.sam_calls,
            "points": execution_meta.prompt_points,
            "runtime_ms": execution_meta.runtime_ms,
        })
    save_episode_records(records)
```

数据生成必须支持断点续跑、hash 校验和版本锁定。

### 2.10.3 Policy 训练

```python
for batch in episode_grouped_loader:
    # batch contains multiple actions for each episode
    pred_utility = policy(batch.episode_features,
                          batch.candidate_sets,
                          batch.action_features,
                          batch.budget)
    loss = ranking_loss(pred_utility, batch.true_utility)
    loss += auxiliary_losses(...)
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
```

推荐默认超参数：

```yaml
optimizer: adamw
learning_rate: 3.0e-4
weight_decay: 1.0e-4
batch_unit: episode
batch_size_episodes: 16
max_epochs: 100
early_stop_patience: 15
scheduler: cosine
mixed_precision: true
grad_clip_norm: 1.0
hidden_dim: 128
num_mlp_layers: 3
dropout: 0.1
seed_list: [0, 1, 2, 3, 4]
```

模型选择指标：validation query-level Dice improvement over best fixed action，次指标为 oracle recovery。不得以 test 表现选 checkpoint。

## 2.11 训练、验证、测试的严格边界

| 阶段 | 可使用 query GT | 可运行全部 actions | 可调超参数 | 可选择最好 action |
|---|---:|---:|---:|---:|
| Train utility generation | 是 | 是 | 否，动作库已冻结 | 仅用于标签 |
| Policy training | 通过离线标签间接使用 | 否 | 训练参数可调 | 否 |
| Validation | 是，仅评价 | 是，可用于选择动作库版本和超参 | 是 | 仅 validation |
| Test | 是，仅最终评价 | **不得在线搜索全部 actions** | 否 | 由冻结 policy 预测 |

必须写 leakage tests：

- 检查 patient/case overlap；
- 检查 reference 是否来自 test；
- 检查 test GT 是否被候选构造函数读取；
- 检查输出目录时间戳与 config hash；
- 检查任何 test-specific thresholds。

---

# 3. 实验设计

## 3.1 数据集

### 3.1.0 当前工作区数据优先级

当前 `D:\SAM` 中，`MoNuSAC\processed` 已经支撑了 FSPL 的固定规则、多参考、matcher/segmenter 复现实验；它应作为 AdaPromptSet-MVP 的主开发与 pilot oracle-gap 数据。`MonuSeg\official` 已存在，适合做 sanity check 和小规模 cross-check。`GlaS` 已解压在 `external\research_data\extracted\GlaS`，具备 85 train / 60 testA / 20 testB 图像与 `_anno` 掩码，是第一优先级的形态迁移扩展。

`DDR-dataset-01\lesion_segmentation` 与 `IDRID` 是眼底病灶域数据，适合测试“病理核/腺体规则是否不能直接迁移到小目标病灶”。但 IDRiD 已有 adaptive selector 外部负结果与测试标签观察历史，因此除非重新声明全新的未观察 split，否则不得用它调参或宣称新的外部正结果。`runs\nnunet\nnUNet_raw\Dataset902_OCSAMPhase3Stratified` 是 IDRID + MonuSeg 的监督基线派生数据，可用作强基线/对照检查，不宜混入 AdaPromptSet policy 训练。

`CPM-17` 当前位于 `external\research_data\raw\CPM17`，但本次盘点只确认到 `cpm15` 的 15 image + 15 `.mat` labels；现有 split 构建脚本预期 32 train / 32 test。CPM-17 在补齐、解析 `.mat` 标注并通过 manifest gate 前，不进入正式结果表。

| 数据源 | 当前路径 | 盘点状态 | AdaPromptSet 用途 | 当前 gate |
|---|---|---|---|---|
| MoNuSAC | `code\datasets\MoNuSAC\processed` | 已用于 FSPL 多轮实验 | MVP 主开发、oracle-gap、fixed/policy 主验证 | 可立即审计后接入 |
| MoNuSeg | `code\datasets\MonuSeg\official` | 已存在 | sanity check、小规模 cross-check、历史对齐 | 可立即审计后接入 |
| GlaS | `external\research_data\extracted\GlaS\Warwick QU Dataset...` | 85/60/20 pairs 已确认 | 腺体形态迁移、domain-conditioned policy 扩展 | 需运行/迁移 split builder |
| DDR | `code\ddr_datasets\DDR-dataset-01\lesion_segmentation` | train/valid/test 图像与 4 类 lesion labels 已确认 | 眼底病灶泛化边界、domain-specific candidate 研究 | 需定义多类别 mask 合并规则 |
| IDRID | `code\datasets\IDRID` | train 55 images/54 masks, test 27/27 | 记录性边界测试、已有负结果对照 | 不可再调参式使用 observed test |
| Dataset902 | `runs\nnunet\nnUNet_raw\Dataset902_OCSAMPhase3Stratified` | 80 train，IDRID+MonuSeg stratified | nnU-Net 强基线/对照，不作为 policy 训练主数据 | 只读对照 |
| CPM-17 | `external\research_data\raw\CPM17` | 当前不满足 32/32 脚本假设 | 外部 nuclei 扩展候选 | 先补齐/解析/审计 |

### 3.1.1 MoNuSeg：主数据集

MoNuSeg 2018 包含 30 张训练图像和 14 张测试图像，来自多器官 H&E 病理图像。官方数据页面：<https://monuseg.grand-challenge.org/Data/>；挑战论文：<https://pmc.ncbi.nlm.nih.gov/articles/PMC10439521/>。

用途：

- 主任务；
- 完整 oracle 分析；
- 完整 baseline 与消融；
- dense nuclei multi-instance grouping；
- AJI、object recall、merge rate。

推荐做法：

- 保留官方 test 作为最终 test；
- 从官方 train cases 中按 patient/case 建立 train/validation；
- 若原始文件中 patient 信息不足，使用 TCGA slide/case ID 审计，不得仅随机切图；
- reference 从 train-only pool 中选择；
- query patching 规则必须固定并避免同一 WSI/case 跨 split。

### 3.1.2 GlaS：不同目标形态的次数据集

GlaS challenge 共 165 张 H&E 图像，常用划分为 85 train、60 test A、20 test B，来自 16 个组织切片。挑战论文：<https://wrap.warwick.ac.uk/id/eprint/81337/>。

用途：

- 验证方法不只对密集小核有效；
- 腺体结构更大、形态更不规则；
- 测试负点对抑制腺体外扩张的价值；
- 只做主要 baseline、policy 和行为统计，不必复制全部 MoNuSeg 消融。

### 3.1.3 CPM-17：外部 nuclei 数据集

CPM-17 常用为 32 张训练图像和 32 张测试图像，包含多癌种病理图像。使用前必须核对当前下载版本、标注格式和官方许可，并将文件 hash 写入 manifest。

用途：

- 同类 nuclei 任务的域外验证；
- 测试 MoNuSeg 训练 policy 的 frozen transfer；
- 区分 per-dataset retraining 与真正 policy transfer。

### 3.1.4 数据集表

论文中应展示：

| Dataset | Target | Train/Val/Test policy split | Official test retained? | Primary role |
|---|---|---|---:|---|
| MoNuSeg | nuclei instances | patient-isolated train/val + official test | Yes | main results and ablation |
| GlaS | gland instances | patient/slide-isolated train/val + Test A/B | Yes | morphology shift |
| CPM-17 | nuclei instances | official split; source/target protocols separated | Yes | external-domain transfer |

具体例数由 `audit_data.py` 读取后自动生成，不得手工硬编码到最终表格。

## 3.2 Reference-query episode sampling

主协议：

- train：每个 query 采样 8 个不同 reference，跨 epoch 重采样；
- validation：每个 query 固定 5 个 train-only reference；
- test：每个 query 固定 5 个 train-only reference；
- 所有方法使用同一 `episode_manifest.jsonl`；
- 多 reference 聚合默认：先对每个 reference 产生 prediction，再按 pixel majority/mean probability 聚合；若 segmenter 只输出 binary masks，则使用 validation-frozen majority threshold；
- 另做 `single-reference` 与 `five-reference` 消融。

统计单位：

- 同一 query 的不同 references 不是独立样本；
- 先得到 query-level metric，再以 query 或 patient 做 bootstrap；
- 若多个 query 来自同一 patient，优先 patient-cluster bootstrap。

## 3.3 Baselines

### 自动输入、直接可比

1. `Matching Top-1`；
2. `Matching Top-3`；
3. `Matching Top-5`；
4. `Matching Top-10`；
5. `Fixed positive + bottom-K negative`；
6. `Fixed positive + hard-background negative`；
7. `KMeans grouped prompts`；
8. `SynPo-style positive/negative selection`；
9. `GF-SAM-style graph grouping`；
10. `Best fixed action selected on validation`；
11. `AdaPromptSet policy`；
12. `Oracle best legal action`，只作为分析上界。

### Privileged upper bounds，不得混为自动比较

- GT center point(s)；
- GT bounding box；
- GT-derived one point per instance；
- GT mask prompt。

这些行必须单独分组并标注 `privileged input`。

## 3.4 指标

### 3.4.1 区域指标

\[
Dice=\frac{2|\hat Y\cap Y|}{|\hat Y|+|Y|},
\]

\[
IoU=\frac{|\hat Y\cap Y|}{|\hat Y\cup Y|}.
\]

### 3.4.2 AJI

使用标准 Aggregated Jaccard Index。实现应与官方/公认实现对照单元测试。所有方法使用相同 instance masks 或相同 semantic-to-instance postprocess。

### 3.4.3 Object recall

GT instance 与任一 predicted instance IoU 超过阈值 \(t_o\) 时视为被召回：

\[
Recall_{obj}=\frac{\#\{g: \max_p IoU(g,p)\ge t_o\}}{N_{GT}}.
\]

默认 \(t_o=0.5\)，并附加 0.3/0.7 敏感性。

### 3.4.4 Merge rate

若一个 predicted component 与两个及以上 GT instances 的 overlap 超过预注册阈值，则记为 merge：

\[
MergeRate=\frac{N_{merged\ predicted\ components}}{N_{predicted\ components}+\epsilon}.
\]

### 3.4.5 Point precision 与 component coverage

\[
PointPrecision=\frac{\#\ positive\ points\ inside\ GT}{\#\ positive\ points},
\]

\[
ComponentCoverage=\frac{\#\ GT\ components\ containing\ at\ least\ one\ positive\ point}{N_{GT}}.
\]

它们用于解释 matcher/prompt 质量，不作为主性能指标。

### 3.4.6 成本指标

必须报告：

- 平均正点数；
- 平均负点数；
- 平均 prompt groups；
- 实际 SAM/SAM2 calls；
- image encoding time；
- prompt decoding time；
- total wall-clock time；
- peak VRAM；
- action-utility 数据生成的总 GPU-hours，单独报告为训练成本。

### 3.4.7 Oracle recovery

\[
OracleRecovery=
\frac{D_{policy}-D_{fixed}}
{D_{oracle}-D_{fixed}+\epsilon}.
\]

只在 \(D_{oracle}>D_{fixed}\) 时报告；否则标记 N/A。

## 3.5 统计检验

主比较单位为 query 或 patient。

必须实现：

1. paired cluster bootstrap，10,000 次重采样；
2. 报告 mean difference 与 95% CI；
3. 对主比较 `policy vs best fixed` 使用双侧 paired permutation test 或 Wilcoxon signed-rank；
4. 主指标预注册为 MoNuSeg Dice；次主指标为 AJI；
5. 多预算、多数据集比较使用 Holm–Bonferroni；
6. 报告 win/tie/loss：tie 定义为 query-level absolute Dice difference < 0.005；
7. 至少 5 个 policy seeds，报告 seed mean ± std；
8. baseline 的确定性结果不通过伪造多个 seed 获得方差。

## 3.6 实验列表

### E0：数据审计

输出：

- case/patient/slide IDs；
- 图像尺寸、倍率、mask 类型；
- split overlap report；
- dataset license；
- file hashes；
- final episode manifests。

### E1：Matching 与固定规则 baseline

目的：建立公平、可复现的基础线，验证 candidate pool 和 segmenter execution 正常。

### E2：Oracle gap 与 action heterogeneity

在 train/validation 上运行全部 actions，在 test 上仅为论文分析可离线计算 oracle，但**不能用于选择 policy 或超参数**。

必须回答：

- oracle gap 是否足够大；
- 每个 action 成为 oracle 的频率；
- action entropy；
- 哪些 query 特征与最优 action 相关。

### E3：Policy 训练

比较：

- action global features only；
- + candidate set encoder；
- + group structure encoder；
- absolute utility vs relative anchor utility；
- pairwise ranking vs regression only。

### E4：主测试

在冻结主设置中只评估一次 test。主预算建议 \(B=3\) calls，同时展示 \(B\in\{1,3,5,8\}\)。

### E5：Factorial ablation

正点适应、负点适应、grouping 适应使用完整 2×2×2 factorial，而不是只做累加式消融。

### E6：跨数据集

- 每个数据集重新训练 policy：只能称 `in-domain policy`；
- MoNuSeg policy 直接到 CPM-17：称 `frozen policy transfer`；
- GlaS 主要用于 morphology shift，不强求冻结迁移显著。

### E7：Reference sensitivity

对同一 query 使用多个 reference，比较均值、标准差、最差分位数和 action consistency。

### E8：效率与成本

包括 inference latency、VRAM、calls、GPU-hours 和 Pareto curve。

### 条件性扩展，不进入第一轮主实验

只有主线达到 checklist 中 Gate D/E 后才允许启动：

- one-step response-conditioned refinement；
- multi-prototype matcher；
- SAM ↔ SAM2 或第二 matcher 的兼容性；
- budget-conditioned single policy；
- abstention / uncertainty-aware action；
- sequential RL。

---

# 4. 论文中需要展示的图表

## 4.0 模拟数据使用规则

以下全部数值仅用于：

- 让 `export_tables.py` 和 LaTeX 模板可立即运行；
- 验证列宽、对齐、显著性符号、图形布局；
- 为 Codex 提供结果文件 schema。

必须在模拟表格标题上加入：

```text
SIMULATED PLACEHOLDER — REPLACE WITH REAL EXPERIMENTS
```

`paper/main.tex` 顶部在模拟阶段显示红色横幅：

```latex
\newif\ifsimulated
\simulatedtrue
\ifsimulated
\twocolumn[{\color{red}\bfseries SIMULATED PLACEHOLDER RESULTS — NOT FOR SUBMISSION}\vspace{6pt}]
\fi
```

真实结果导入后改为 `\simulatedfalse`。构建脚本在 `submission` 模式下若检测到 `SIMULATED` 字符串必须失败。

## Table 1：与相似方法的任务和决策边界

此表是定性表，不使用模拟性能数据。

| Method | Reference-guided | Explicit SAM negatives | Learned policy | Multi-instance grouping | Cost-conditioned complete action selection |
|---|---:|---:|---:|---:|---:|
| SynPo | ✓ | ✓ | ✗ | fixed K-means | ✗ |
| GF-SAM | ✓ | mostly negative reference | ✗ | graph WCC | ✗ |
| AlignSAM | ✗ standard support-query | learned label module | ✓ sequential RL | ✗ | ✗ |
| Point Prompt Defender | ✗ | optimizes coarse set | ✓ adversarial DQN | graph environment | ✗ |
| **AdaPromptSet** | ✓ | ✓ | ✓ offline set ranking | ✓ prompt groups | ✓ |

注意：GF-SAM 的 negative 主要是 background reference evidence，不要错误写成与 SynPo 完全相同的显式 negative point prompting。

## Table 2：MoNuSeg 主比较

**SIMULATED PLACEHOLDER — REPLACE WITH REAL EXPERIMENTS**

主预算：最多 3 次 segmenter calls。数值单位为 mean；括号中最终应填 95% CI 或 seed std，不能混用。

| Method | Input | Pos. | Neg. | Calls ↓ | Dice ↑ | IoU ↑ | AJI ↑ | Obj. Recall ↑ | Merge ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Top-1 | auto | 1.0 | 0.0 | 1.0 | 0.681 | 0.516 | 0.501 | 0.642 | 0.184 |
| Top-3 one group | auto | 3.0 | 0.0 | 1.0 | 0.712 | 0.553 | 0.536 | 0.681 | 0.171 |
| Top-5 one group | auto | 5.0 | 0.0 | 1.0 | 0.724 | 0.568 | 0.548 | 0.696 | 0.166 |
| Fixed hard pos/neg | auto | 5.0 | 3.0 | 1.0 | 0.739 | 0.586 | 0.565 | 0.711 | 0.153 |
| KMeans groups | auto | 3.0 | 0.0 | 3.0 | 0.746 | 0.595 | 0.579 | 0.739 | 0.146 |
| SynPo-style | auto | 5.0 | 5.0 | 1.0 | 0.748 | 0.598 | 0.582 | 0.724 | 0.143 |
| GF-SAM-style graph | auto | 4.6 | 0.0 | 3.0 capped | 0.758 | 0.610 | 0.596 | 0.761 | 0.132 |
| Best fixed action (val-selected) | auto | 4.1 | 1.2 | 3.0 | 0.759 | 0.611 | 0.598 | 0.756 | 0.133 |
| **AdaPromptSet** | auto | 4.8 | 1.6 | 2.7 | **0.779** | **0.638** | **0.624** | **0.801** | **0.111** |
| Oracle action | uses test GT only to identify action | 5.0 | 1.8 | 2.9 | 0.812 | 0.684 | 0.665 | 0.842 | 0.094 |
| GT one-point-per-instance | privileged | GT | 0.0 | GT-dependent | 0.851 | 0.741 | 0.713 | 0.902 | 0.072 |

由模拟数值得到的 oracle recovery：

\[
(0.779-0.759)/(0.812-0.759)=37.7\%.
\]

最终表格必须添加显著性标记，并在 caption 中说明 oracle 与 privileged rows 不参与公平排名。

## Table 3：预算—准确率曲线

**SIMULATED PLACEHOLDER — REPLACE WITH REAL EXPERIMENTS**

| Max calls B | Best fixed Dice | Policy Dice | Oracle Dice | Policy AJI | Avg. actual calls | Oracle recovery |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.724 | 0.739 | 0.758 | 0.572 | 1.00 | 44.1% |
| 3 | 0.759 | 0.779 | 0.812 | 0.624 | 2.70 | 37.7% |
| 5 | 0.768 | 0.788 | 0.823 | 0.635 | 3.82 | 36.4% |
| 8 | 0.772 | 0.791 | 0.829 | 0.639 | 4.31 | 33.3% |

该表对应 Figure 3 的 Dice–calls Pareto curve。不要只展示最高预算。

## Table 4：Oracle action heterogeneity

**SIMULATED PLACEHOLDER — REPLACE WITH REAL EXPERIMENTS**

| Oracle action family | Fraction of test queries | Mean target count | Mean matching entropy | Mean fg-bg margin |
|---|---:|---:|---:|---:|
| Single-group positive-only | 18.6% | 12.4 | 0.42 | 0.31 |
| Single-group pos/neg | 16.3% | 15.8 | 0.57 | 0.18 |
| Independent point groups | 14.0% | 26.1 | 0.61 | 0.22 |
| KMeans/local-max groups | 23.3% | 31.7 | 0.68 | 0.16 |
| GF-SAM-style graph groups | 20.9% | 35.2 | 0.71 | 0.14 |
| Safe anchor / abstain | 7.0% | 8.2 | 0.79 | 0.07 |

同时报告最常见单一 action 的 oracle 占比。若超过 70%，H1 不成立，应触发 checklist 的重新评估节点。

## Table 5：2×2×2 Factorial ablation

**SIMULATED PLACEHOLDER — REPLACE WITH REAL EXPERIMENTS**

所有行使用相同最大预算 \(B=3\)。`Adaptive grouping` 指从完整 proposal bank 中选择 group structure，不是独立 layout head。

| Adaptive positive | Adaptive negative | Adaptive grouping | Dice ↑ | AJI ↑ | Actual calls ↓ |
|---:|---:|---:|---:|---:|---:|
| ✗ | ✗ | ✗ | 0.746 | 0.579 | 3.00 |
| ✓ | ✗ | ✗ | 0.758 | 0.591 | 2.96 |
| ✗ | ✓ | ✗ | 0.754 | 0.588 | 2.93 |
| ✗ | ✗ | ✓ | 0.762 | 0.603 | 2.78 |
| ✓ | ✓ | ✗ | 0.769 | 0.608 | 2.91 |
| ✓ | ✗ | ✓ | 0.772 | 0.615 | 2.75 |
| ✗ | ✓ | ✓ | 0.770 | 0.613 | 2.71 |
| ✓ | ✓ | ✓ | **0.779** | **0.624** | **2.70** |

需用 factorial ANOVA 或更稳健的 query-level mixed/paired analysis报告主效应与交互效应；若分布假设不满足，用 bootstrap interaction contrast。

## Table 6：Policy architecture 与 supervision 消融

**SIMULATED PLACEHOLDER — REPLACE WITH REAL EXPERIMENTS**

| Episode features | Candidate set encoder | Group encoder | Objective | Dice | Oracle recovery |
|---|---:|---:|---|---:|---:|
| global only | ✗ | ✗ | regression | 0.763 | 7.5% |
| global only | ✗ | ✗ | pairwise | 0.768 | 17.0% |
| global | ✓ | ✗ | pairwise | 0.772 | 24.5% |
| global | ✓ | ✓ | regression | 0.773 | 26.4% |
| global | ✓ | ✓ | pairwise | 0.777 | 34.0% |
| global | ✓ | ✓ | pairwise + aux | **0.779** | **37.7%** |

## Table 7：跨数据集结果

**SIMULATED PLACEHOLDER — REPLACE WITH REAL EXPERIMENTS**

此表的 `Policy` 默认表示各数据集 train/validation 上重新训练的 in-domain policy；不可据此声称 frozen transfer。

| Dataset | Best fixed Dice | GF-SAM-style Dice | In-domain policy Dice | Δ vs best fixed | AJI / Object F1 | Avg calls |
|---|---:|---:|---:|---:|---:|---:|
| MoNuSeg | 0.759 | 0.758 | **0.779** | +0.020 | AJI 0.624 | 2.70 |
| GlaS | 0.812 | 0.821 | **0.836** | +0.024 | Obj. F1 0.781 | 1.62 |
| CPM-17 | 0.722 | 0.736 | **0.751** | +0.029 | AJI 0.587 | 3.08 |

## Table 8：Frozen policy transfer

**SIMULATED PLACEHOLDER — REPLACE WITH REAL EXPERIMENTS**

| Source policy | Target | Target best fixed | Frozen source policy | Target-retrained policy | Frozen Δ | Claim allowed |
|---|---|---:|---:|---:|---:|---|
| MoNuSeg | CPM-17 | 0.722 | 0.731 | 0.751 | +0.009 | limited same-task transfer |
| MoNuSeg | GlaS | 0.812 | 0.808 | 0.836 | -0.004 | no transfer claim |
| GlaS | MoNuSeg | 0.759 | 0.761 | 0.779 | +0.002 | non-inferior only if CI supports |

只有 `Frozen source policy` 不使用 target labels 重新训练时，才可讨论 transfer。

## Table 9：Prompt behavior statistics

**SIMULATED PLACEHOLDER — REPLACE WITH REAL EXPERIMENTS**

| Dataset | Avg positive | Avg negative | Avg groups | Neg-used queries | Graph/grouped action ratio | Dice |
|---|---:|---:|---:|---:|---:|---:|
| MoNuSeg | 4.8 | 1.6 | 2.70 | 48.2% | 57.4% | 0.779 |
| GlaS | 3.2 | 2.4 | 1.62 | 66.8% | 31.5% | 0.836 |
| CPM-17 | 5.1 | 1.9 | 3.08 | 52.7% | 61.3% | 0.751 |

该表只能说明行为差异。要声称 policy 学到 dataset prior，必须有 shared vs domain-conditioned 对照并排除目标数量、图像尺寸等混杂因素。

## Table 10：Reference sensitivity

**SIMULATED PLACEHOLDER — REPLACE WITH REAL EXPERIMENTS**

| Method | Mean Dice | Across-reference std ↓ | 10th percentile Dice ↑ | Action consistency ↑ | Worst-reference Dice ↑ |
|---|---:|---:|---:|---:|---:|
| Top-5 | 0.724 | 0.043 | 0.667 | 1.000 fixed | 0.641 |
| SynPo-style | 0.748 | 0.040 | 0.693 | 1.000 fixed | 0.671 |
| GF-SAM-style | 0.758 | 0.038 | 0.706 | 0.721 | 0.684 |
| AdaPromptSet | **0.779** | **0.028** | **0.742** | 0.684 | **0.718** |

Action consistency 对 adaptive 方法使用同一 query 不同 reference 下 action family 的一致比例或 normalized mutual information。

## Table 11：效率

**SIMULATED PLACEHOLDER — REPLACE WITH REAL EXPERIMENTS**

在同一 GPU、相同 batch size、warm-up 后测量。总时间必须包含 matching、candidate construction、policy、所有 mask decoder calls 和 fusion。

| Method | Matching ms | Proposal/policy ms | Actual calls | Decoder ms | Total ms ↓ | Peak VRAM GB |
|---|---:|---:|---:|---:|---:|---:|
| Top-5 one group | 46 | 5 | 1.0 | 61 | 116 | 8.3 |
| SynPo-style | 69 | 12 | 1.0 | 61 | 146 | 9.1 |
| KMeans groups | 46 | 8 | 3.0 | 183 | 241 | 8.5 |
| GF-SAM-style | 46 | 18 | 4.3 uncapped raw | 262 | 331 | 8.7 |
| AdaPromptSet | 46 | 11 | 2.7 | 165 | 226 | 8.6 |

若 GF-SAM-style 在主预算中被 cap，应同时报告 uncapped 原始成本和 budget-matched 版本。

## Figure 2：Action bank 结构图

展示同一候选池如何产生：

- Top-K one group；
- pos/neg one group；
- independent groups；
- KMeans groups；
- graph groups；
- 每个 action 的 calls 和 points。

必须直观说明策略选择的是完整 proposal，而不是任意像素。

## Figure 3：Dice–cost Pareto curve

- x 轴：平均实际 SAM calls，或 wall-clock runtime；
- y 轴：Dice；
- 线：best fixed、policy、oracle；
- 每个预算一个 marker；
- 加 query-cluster bootstrap CI；
- 不允许通过连接不对应同一预算定义的点制造虚假 Pareto 优势。

模拟数据使用 Table 3。

## Figure 4：定性决策过程

每个案例固定列：

```text
(a) reference image + mask
(b) foreground/background/contrast maps
(c) C+ and C- candidates
(d) best fixed prompt groups
(e) policy-selected prompt groups
(f) fixed prediction
(g) policy prediction
(h) GT
```

至少四行：

1. 简单 query：policy 选择少量点；
2. 密集 query：policy 拆分多个 groups；
3. 混淆背景：policy 加入 group-specific negative points；
4. 失败案例：matching evidence 错误，policy 无法补救。

所有 prompt：正点绿色圆点、负点红色叉号、不同 groups 用不同轮廓/编号。不得只展示成功案例。

## Figure 5：策略行为解释

建议展示：

- matching entropy vs selected calls；
- target count vs groups；
- fg-bg margin vs probability of using negatives；
- query-level utility predicted vs true utility calibration scatter。

使用 partial dependence 或分箱均值，并报告置信区间。不能把相关性写成因果。

---

# 5. LaTeX 双栏约 8 页 PDF 规范

## 5.1 目标

生成一个可阅读、可投前内部评审的双栏 8 页左右论文草稿。它不必绑定最终投稿会议，但采用 CVPR/IEEE 风格双栏版式，主文目标为 8 页，不含参考文献可额外 1–2 页。

建议标题：

```text
AdaPromptSet: Cost-Aware Prompt Set Selection for
Reference-Guided Dense Medical Segmentation
```

## 5.2 页面预算

| 内容 | 目标页数 |
|---|---:|
| Title, abstract, introduction | 1.0 |
| Related work and problem formulation | 0.8 |
| Method overview and matching/candidates | 1.2 |
| Proposal bank and utility learning | 1.3 |
| Experimental setup | 0.9 |
| Main results and oracle analysis | 1.3 |
| Ablation, behavior, qualitative, limitations | 1.2 |
| Conclusion | 0.3 |
| **Main text total** | **8.0** |

不要通过极小字号、负间距或不可读表格硬压到 8 页。优先把次要表格移至 appendix/supplementary。

## 5.3 LaTeX 文件结构

```text
paper/
├── main.tex
├── references.bib
├── sections/
│   ├── 00_abstract.tex
│   ├── 01_intro.tex
│   ├── 02_related.tex
│   ├── 03_method.tex
│   ├── 04_experiments.tex
│   ├── 05_results.tex
│   ├── 06_limitations.tex
│   └── 07_conclusion.tex
├── tables/
│   ├── table_related.tex
│   ├── table_main.tex
│   ├── table_budget.tex
│   ├── table_oracle.tex
│   ├── table_ablation.tex
│   └── table_crossdata.tex
├── figures/
│   ├── method_overview.tex
│   ├── action_bank.tex
│   ├── pareto.pdf
│   ├── qualitative.pdf
│   └── behavior.pdf
└── generated/
    ├── macros_results.tex
    └── build_metadata.tex
```

## 5.4 推荐包

```latex
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{amsmath,amssymb}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,fit,calc,shapes.geometric}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{hyperref}
```

表格必须使用 `booktabs`，避免竖线。超宽表优先：

- 减少文字列；
- 用符号和脚注；
- `table*` 跨双栏；
- 不要默认使用 `\resizebox{\textwidth}{!}` 把字缩得不可读。

## 5.5 主文建议结构

### Abstract

包含：

- reference-guided frozen foundation segmentation 背景；
- 固定 prompt rules 的 query adaptivity/cost 问题；
- structured prompt-set selection 定义；
- offline utility learning；
- 真实结果完成后填三条关键数字；
- 不写 universal/generalizable，除非 transfer 实验支持。

### Introduction

逻辑：

1. SAM 需要 prompt，reference matching 可自动提供 evidence；
2. GF-SAM/SynPo 等已改进候选与组织，但决策仍由确定性规则或固定预算驱动；
3. 学习 prompting 已有 AlignSAM/PPD，因此新意不是“学习 prompt”；
4. 研究缺口是 reference-conditioned、budget-constrained、complete structured prompt-set selection；
5. 三条贡献。

### Related Work

三段即可：

- training-free/reference-guided few-shot segmentation；
- automatic prompt generation and negative prompts；
- learned/RL prompt optimization。

### Method

按以下顺序：

1. problem formulation；
2. matching evidence and candidate pools；
3. structured prompt proposal bank；
4. set utility predictor；
5. offline action utility supervision；
6. budgeted inference。

### Experiments

- datasets/splits；
- baselines；
- metrics/statistics；
- implementation；
- main comparison；
- oracle gap；
- ablation；
- cross-dataset/efficiency；
- qualitative/limitations。

## 5.6 自动导出真实结果

不要手工复制实验数值到 `.tex`。`export_tables.py` 从标准结果文件生成：

```text
artifacts/metrics/summary.parquet
artifacts/metrics/bootstrap_ci.parquet
artifacts/metrics/significance.parquet
```

并输出：

```text
paper/generated/macros_results.tex
paper/tables/*.tex
```

示例宏：

```latex
\newcommand{\MainPolicyDice}{77.9}
\newcommand{\MainFixedDice}{75.9}
\newcommand{\MainOracleRecovery}{37.7}
```

数值格式统一：

- Dice/IoU/AJI 主表显示百分制一位小数或 0–1 三位小数，全文一致；
- CI 使用 `77.9 [76.4, 79.2]`；
- calls 保留两位；
- runtime 整数 ms；
- p-value 小于 0.001 写 `<0.001`。

## 5.7 编译命令

`scripts/build_paper.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-draft}"

python scripts/export_tables.py --mode "$MODE"
python scripts/export_figures.py --mode "$MODE"

if [[ "$MODE" == "submission" ]]; then
  if grep -R "SIMULATED PLACEHOLDER\|simulatedtrue" paper/; then
    echo "ERROR: simulated placeholders remain"
    exit 1
  fi
fi

cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

验收命令：

```bash
bash scripts/build_paper.sh draft
pdfinfo paper/main.pdf
```

构建必须：

- 无 undefined references；
- 无 missing citations；
- 无 overfull hbox > 5 pt；
- 所有 figure 字体可读；
- draft 模式明确显示模拟结果水印；
- submission 模式拒绝模拟数据。

## 5.8 BibTeX 至少包含

- SAM；
- SAM2；
- DINOv2；
- GF-SAM / Bridge the Points；
- SynPo；
- AlignSAM；
- Point Prompt Defender；
- MoNuSeg challenge；
- GlaS challenge；
- CPM-17 数据来源；
- AJI 指标来源；
- DeepSets 或 Set Transformer；
- paired bootstrap/statistical references（如需要）。

---

# 6. 实现细节与工程要求

## 6.1 统一配置与可复现性

每次运行生成：

```json
{
  "git_commit": "...",
  "config_hash": "...",
  "dataset_manifest_hash": "...",
  "matcher_checkpoint_hash": "...",
  "segmenter_checkpoint_hash": "...",
  "seed": 0,
  "timestamp": "...",
  "cuda": "...",
  "torch": "..."
}
```

所有 metrics 和 predictions 必须带 run ID。禁止覆盖旧结果。

## 6.2 结果文件 schema

`action_utility.parquet` 每行一个 `(episode, action)`：

```text
episode_id
query_case_id
reference_case_id
split
dataset
action_id
action_family
budget
valid
invalid_reason
num_positive
num_negative
num_groups
sam_calls
runtime_ms
dice
iou
aji
object_recall
merge_rate
point_precision
component_coverage
episode_feature_path
action_feature_path
config_hash
```

`test_predictions.parquet` 只保存冻结 policy 的选择，不保存 online oracle selection 作为方法输出。

## 6.3 测试

至少实现：

- feature coordinate round-trip；
- point labels 和 SAM API 输入顺序；
- candidate dedup；
- action cost count；
- group-to-call 一一对应；
- mask fusion deterministic；
- Dice/IoU/AJI known examples；
- patient split leakage；
- policy 无 GT 字段输入；
- budget filtering；
- oracle 不进入 checkpoint selection；
- simulated placeholder submission guard。

## 6.4 运行命令建议

在当前 Windows 工作区，优先使用 PowerShell 与本地 GPU 环境：

```powershell
cd D:\SAM\adapromptset
$env:PYTHONPATH = "D:\SAM\adapromptset\src;D:\SAM\FSPL"
D:\SAM\conda_envs\sam_gpu\python.exe -m pytest
```

MVP 阶段先执行 audit/smoke/pilot，不直接运行完整 action grid。若尚未迁移脚本，下列命令是目标接口；实现时可先由 `D:\SAM\FSPL` 中对应 stage 脚本迁移而来。

```bash
# 1. Audit
python scripts/audit_data.py --config configs/experiments/e0_data_audit.yaml

# 2. Build fixed episode manifests
python scripts/build_episodes.py --dataset monuseg --seed 0

# 3. Cache frozen features
python scripts/cache_features.py --dataset monuseg --matcher dinov2_vitl14

# 4. Generate full train/val action utility
python scripts/build_action_utility_dataset.py \
  --config configs/experiments/e2_oracle_gap.yaml \
  --split train,val

# 5. Train policy
python scripts/train_policy.py --config configs/experiments/e3_policy_train.yaml

# 6. Frozen test evaluation
python scripts/evaluate.py --config configs/experiments/e4_main_test.yaml --split test

# 7. Statistics
python scripts/run_bootstrap.py --experiment e4_main_test

# 8. Tables/figures/PDF
bash scripts/build_paper.sh draft
```

## 6.5 计算预算管理

在开始完整 train action grid 前，先做 10–20 个 query 的 pilot，估算：

\[
GPUHours \approx N_{episodes}\times N_{actions}\times t_{action}/3600.
\]

若预计成本过高，优先：

1. 缓存 image embeddings；
2. 合并动作间重复 group calls；
3. 对完全相同 prompts 使用 content hash 缓存 mask；
4. 先在 train subset 上筛除始终 dominated 的 actions；
5. 只能使用 train/validation 数据做 action pruning；
6. 记录 pruning 前后 oracle gap，确保没有破坏动作上界。

---

# 7. 论文结论边界

## 7.1 当主实验成功时可写

- fixed prompt configurations are suboptimal for heterogeneous reference-query episodes；
- a lightweight offline utility predictor improves segmentation under matched call budgets；
- complete prompt-set selection recovers a measurable portion of the frozen action-bank oracle gap；
- positive, negative, and grouping decisions have complementary effects；
- the method is compatible with the evaluated frozen matcher/segmenter pair。

## 7.2 只有额外证据后可写

- `transfers across datasets`：冻结 source policy 在 target 上正向且 CI 支持；
- `model-agnostic`：多个 matcher/segmenter 上冻结 policy 迁移成功；
- `plug-and-play`：无需 target retraining 且接口/性能稳定；
- `learns dataset priors`：控制混杂因素后，domain behavior 差异仍带来收益；
- `efficient`：不仅 calls 少，真实 wall-clock 和 GPU cost 也更优。

## 7.3 当结果不支持原主张时

按 `checklist.md` 中的 gate 执行，不得通过 test 调参挽救结果。可能的诚实转向包括：

- 若 oracle gap 小：论文转为系统性分析，证明简单 fixed rule 已近最优，或重新设计更有表达力的 prompt groups；
- 若 oracle gap 大但 policy 学不会：研究重点转为 utility representation / action ranking；
- 若负点无增益：删除负点联合主张，集中于 dense multi-instance grouping；
- 若 grouping 无增益：转为 cost-aware positive/negative budget allocation；
- 若只在 MoNuSeg 有效：标题限定为 dense histopathology nuclei segmentation；
- 若 frozen transfer 失败：只主张 in-domain learned selection，不讨论泛化。

---

# 8. Codex 的完成定义

Codex 不应仅生成代码骨架。项目完成需满足：

1. `pytest` 全部通过；
2. 数据 split 与 leakage report 已生成；
3. action bank 有版本号和 hash；
4. train/val action-utility 数据可复现；
5. policy 有至少 5 seeds；
6. test 只由冻结 config 与 checkpoint 执行一次正式报告；
7. 所有主表自动生成；
8. 所有模拟数据已被真实数据替换；
9. `bash scripts/build_paper.sh submission` 成功；
10. `paper/main.pdf` 为约 8 页双栏主文，引用、公式、图表无错误；
11. `checklist.md` 中对应项目均已勾选，并记录所有偏离、失败实验与转向理由。
