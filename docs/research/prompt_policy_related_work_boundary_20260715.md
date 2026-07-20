# Reference-Conditioned Prompt Policy：相关工作边界与可发表定位

更新：2026-07-15

## 1. 结论先行

“学习 point prompt policy”本身不是空白。AlignSAM（CVPR 2024）、Plug-and-Play PPO（CVPR 2025）和 Point Prompt Defender（CVPR 2026）已经分别使用强化学习、异构图策略或攻击-防御 DQN 自动选择/优化 SAM 点提示。更直接地，Segment Any Tissue（SAT，MedIA 2025）已经完成 one-shot medical image-mask reference、feature/physical 双空间提示优化、cyclic feedback、每图超参数选择和多 reference ensemble。因此本项目不能声称首次提出 Prompt Policy Learning、reference-guided medical auto-prompt 或动态 prompt scheme。

当前仍有机会形成的贡献组合是：

1. reference mask 同时提供前景 prototype 与背景 prototype，显式定义 dense all-instance intent；
2. policy 学习的不是从零寻找任意点击，而是在 reference-query matching 产生的可解释 point pool 上联合决定 K、背景抑制 lambda 与候选预算；
3. 冻结 matcher 与 SAM 主干，只训练轻量、可审计的 policy/selector；
4. anchored safety gate 把 validation-best 固定策略作为保底，只在预测相对增益可信时切换；
5. 在密集病理细胞核场景报告 object coverage、merge、面积扩张和 oracle regret，而不只报告单物体 Dice；
6. 研究同一 reference-conditioned policy 在不同 feature backbone 与 SAM-family 响应下何时迁移、何时失败。

如果最终结果只有 MoNuSeg 上极小的 Dice 增益，而没有显著性、机制证据或跨模型结论，则不足以支撑高接受率论文。

## 2. 与最接近工作的区别

| 工作 | 是否使用 support/reference | 是否学习 prompt | 是否冻结 SAM | 主要决策 | 与本项目的关键区别 |
| --- | --- | --- | --- | --- | --- |
| Matcher | 1 个 image-mask reference | 否，training-free sampler | 是 | 匹配点采样、proposal、instance-level matching | 本项目从 Matcher 的固定采样规则前进一步，学习 reference-conditioned K/lambda/budget；但必须把 Matcher 作为直接 baseline |
| Segment Any Tissue, MedIA 2025 | 1 个或多个 image-mask references | training-free cyclic feedback 与每图 prompt-scheme selection | 是 | 双空间 positive/negative/hard prompts、超参数 bank | **最直接先例**；本项目只能以严格 train-only policy、fg/bg prototype 机制、有限候选预算和 dense all-nuclei regret/统计区别，不能把动态 K 本身写成创新 |
| One-Prompt | 1 个 prompted template | 是，但训练新分割网络 | 否，不是原始冻结 SAM policy | template-query feature fusion | 它学习通用分割器；本项目学习冻结模型外部的轻量决策策略 |
| UniverSeg | 多个 image-label support | 隐式学习 support-query 映射 | 不使用 SAM | CrossBlock 通用分割 | 它在 53 个数据集上元训练完整网络；本项目只训练 prompt action policy |
| AlignSAM, CVPR 2024 | 可利用视觉/语言上下文 | 是，RL agent | 是 | 迭代选择点及正负标签 | 已覆盖“冻结 SAM + RL 自动点提示”；本项目必须以 reference fg/bg matching、K/lambda 联合决策和 dense medical intent 区分 |
| Plug-and-Play PPO, CVPR 2025 | 从初始 prompts 和 foundation features 出发 | 是，DRL | 是 | 双空间异构图上的点分布优化 | 是最直接 prompt optimizer baseline；不能把“自适应点分布”写成新颖点 |
| Point Prompt Defender, CVPR 2026 | 任意 coarse prompt set | 是，攻击/防御 DQN | 是 | 抑制破坏性 prompts、恢复准确率 | 已覆盖 task-agnostic prompt refinement；本项目应强调 reference-conditioned task definition 和预算选择，而非泛化 prompt refinement |
| PromptMoE, CVPR 2026 | image + coarse mask | 是，MoE router | 是 | 动态选图像线索、空间多样正点和一个负点 | 已覆盖 learned expert utility、prompt interaction 与 model-agnostic refinement；本项目必须用 reference-defined task、无 coarse query mask、dense all-instance 指标和 train-only action regret 区分 |
| Instruction-Focus-Prompt, CVPR Findings 2026 | visual reference + text instruction | 是，视觉/文本 focus alignment | 是 | DINOv3/CLIP 结构点，迭代 mask union | 已在自然与医学 one-shot 上对 Matcher/PPO 做比较；本项目不应再声称 feature matching 是最强通用路线，只聚焦病理 dense all-nuclei 的受控 policy 学习 |
| SAMIX, CVPR 2026 | image-mask reference pairs | 是，RL reference selection | SAM2 带 semantic adapter | 选择相关且互补的 references | 已覆盖“学习 reference selection”；本项目当前固定/分层采样 reference，只能将其作为后续扩展或对照 |
| Self-Prompt-SAM, 2025 | 无人工 per-image prompt | 是/规则混合 | 含 adapter | 辅助 mask 生成 box 与中心点 | 它依赖辅助分割和模型适配；本项目使用 image-mask reference matching 且冻结 segmenter |
| T-SAM, CVPRW 2025 | 少量 labeled examples | 是 | 轻量学习 prompt representation | 跨样本提示表示 | 与本项目同属 support-conditioned SAM，需要在 related work 中正面比较 |

## 3. 论文问题应如何改写

不再写：

> 我们首次学习如何为 SAM 选择点提示。

建议写：

> 现有自动提示方法通常从图像状态、文本语义或给定 coarse prompts 出发优化点分布，但很少研究由 image-mask reference 显式定义的 dense all-instance task 中，前景相似性、背景排斥、prompt budget 与 frozen segmenter response 应如何联合决策。我们把该问题形式化为 reference-conditioned action selection，并引入 validation-anchored safety gate，以有限候选预算在固定策略与自适应策略之间进行可审计切换。

该表述仍需实验支持以下三点：

1. fg-bg reference matching 确实比 fg-only、随机 reference 和无 reference prompt pool 更可靠；
2. K/lambda 的最优值随 reference-query episode 变化，且学习策略能缩小 fixed-to-oracle gap；
3. anchored gate 的收益不是只来自“多数病例不做任何改变”，而是在足够多病例上准确识别何时切换。

## 4. 必须加入的 baseline

论文主表至少应有：

1. SAM center/random/grid/box；
2. Matcher 原始 pipeline 与 GBMSeg-style/SAT predecessor adapted baseline；
3. fixed fg-only top-K；
4. fixed fg-bg K/lambda；
5. validation-best fixed action；
6. learned one-candidate policy；
7. learned top-3 policy；
8. anchored top-3 policy；
9. full-grid oracle upper bound；
10. 若代码与协议可适配，加入 AlignSAM 或 Plug-and-Play PPO；若不能公平复现，至少在 related work 与限制中明确说明差异；
11. SAT 官方仓库当前未释出实现，不能伪称 official reproduction；使用公开 GBMSeg 前身时必须标为 adapted baseline。

## 5. 必须补的机制实验

| 机制问题 | 指标/图 |
| --- | --- |
| reference 前景是否找对 | point precision@K、component coverage@K、score-map AP/AUC |
| 背景 prototype 是否有用 | fg-only vs fg-bg；背景误触发面积、FP、Dice |
| 为什么 K 随病例变化 | oracle K 与 score entropy、local maxima 数、核密度、reference-query gap 的相关性 |
| pre-filter 是否包含好动作 | top-1/3/5 oracle 与 full-grid oracle；prefilter regret |
| post-selector 是否误杀 | shortlist oracle - selected Dice；selector regret |
| safety gate 是否真的识别风险 | predicted gain calibration、switch precision、coverage-risk curve |
| reference 选择是否敏感 | 单 reference、5 个分层 reference、随机 reference、oracle reference |

## 6. 当前风险

1. 29 个 train query 太少，策略容易记住病例难度而不是动作相对优劣。
2. 单 reference 严格 test 只提升 +0.00145，bootstrap CI 包含 0。
3. 当前 action policy 使用 ExtraTrees，不构成算法创新；需要靠问题形式化、机制、严格协议和强结果支撑。
4. anchored gate 若几乎总回退，只能证明安全，不能证明 adaptive policy 有足够覆盖率。
5. 2024-2026 已有 SAT、PPO、Point Prompt Defender、PromptMoE 与 IFP 等强先例，若不做直接或概念对比，审稿人会认为贡献重复。
6. MoNuSeg 只有 all-nuclei intent，尚不能支撑“通用意图理解”；论文必须限制表述，或用 MoNuSAC class/instance 标注扩展 intent。

## 7. 当前最合理投稿故事

第一篇不追求“通用所有领域”。建议故事为：

> 面向密集病理细胞核的一次示例分割，提出 reference-conditioned foreground-background prompt action space 与安全门控策略，在冻结 feature matcher 和 promptable segmenter 下学习每个 query 的 prompt budget 与背景抑制强度；通过严格 source-test hold-out、episode-level 多 reference 训练、oracle regret 与风险-覆盖分析，证明固定 K/lambda 在密集多实例场景中的局限，并给出何时可安全自适应的机制解释。

只有当 MoNuSeg 主结果稳定后，再扩展到 MoNuSAC 的 class/instance intent 与其他 segmenter。

## 8. 主要原始资料

- Matcher: `docs/papers2/13_Matcher_One_Shot_Feature_Matching.pdf`
- Segment Any Tissue: https://doi.org/10.1016/j.media.2025.103550
- SAT reproducibility audit: `docs/research/sat_baseline_reproducibility_audit_20260715.md`
- GBMSeg code: `external/GBMSeg/` commit `d3e41786ba603bcc85f58a9b267d23de56aa2c87`
- One-Prompt: `docs/papers2/14_One_Prompt_to_Segment_All_Medical_Images.pdf`
- UniverSeg: `docs/papers2/15_UniverSeg_Universal_Medical_Image_Segmentation.pdf`
- AlignSAM: https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_AlignSAM_Aligning_Segment_Anything_Model_to_Open_Context_via_Reinforcement_CVPR_2024_paper.pdf
- Plug-and-Play PPO: https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Plug-and-Play_PPO_An_Adaptive_Point_Prompt_Optimizer_Making_SAM_Greater_CVPR_2025_paper.html
- Point Prompt Defender: https://openaccess.thecvf.com/content/CVPR2026/html/Liu_Attack_for_Defense_Adversarial_Agents_for_Point_Prompt_Optimization_Empowering_CVPR_2026_paper.html
- SAMIX: https://openaccess.thecvf.com/content/CVPR2026/html/Hu_SAMIX_Reinforcing_SAM2_with_Semantic_Adapter_and_Reference_Selecting_Policy_CVPR_2026_paper.html
- PromptMoE: https://openaccess.thecvf.com/content/CVPR2026/html/Price_PromptMoE_A_Segmentation_Refinement_Framework_Leveraging_Mixture_of_Experts_for_CVPR_2026_paper.html
- Instruction-Focus-Prompt: https://openaccess.thecvf.com/content/CVPR2026F/html/Xia_Instruction-Focus-PromptSemantics-Driven_Structural_Prompts_for_Universal_SAM_Segmentation_CVPRF_2026_paper.html
