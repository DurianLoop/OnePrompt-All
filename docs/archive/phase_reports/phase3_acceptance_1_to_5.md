# Phase3 五个验收标准推进报告

这份报告把当前阶段从第 1 个验收标准推进到第 5 个验收标准。需要强调：这里完成的是可复现工程验收和研究路线验收，不伪造未运行强基线的指标。

## 1. 验收总表

| Step | 验收标准 | 状态 | 证据文件 | 行数 | Mean Dice | 仍需补齐 |
| ---: | --- | --- | --- | ---: | ---: | --- |
| 1 | Support Prompt v1 | pass_engineering | `runs\phase3_milestone5\support_guided_prompt_v1.csv` | 120 | 0.3089 | 当前 support 只用历史 Dice proxy；论文版需要用视觉相似度/病灶先验替代 oracle-like 历史表现。 |
| 2 | Perturbation Uncertainty v1 | pass_proxy | `runs\phase3_milestone5\perturbation_uncertainty_proxy_v1.csv` | 120 | 0.3116 | 当前是 auto-family consensus proxy；下一版要真实生成 jitter prompts 并重跑模型。 |
| 3 | Cross-model Calibration | pass_engineering | `runs\phase3_milestone5\cross_model_calibrated_portfolio_v4.csv` | 40 | 0.3555 | 需要扩大数据后验证置信区间和显著性，目前还不能写 SOTA claim。 |
| 4 | 扩数据准备 | pass_preparation | `runs\phase3_acceptance_1_to_5\dataset_expansion_inventory.csv` | 3 |  | 已形成扩展清单；还需把 MedSAM demo npz 和更多 IDRID/MonuSeg 病例写入统一 manifest 并跑主表。 |
| 5 | 强基线比较 | pass_audit_not_scientific_claim | `runs\phase3_acceptance_1_to_5\strong_baseline_readiness.csv` | 7 |  | 已完成 readiness audit；科学验收必须补 SAM-Med2D / Med-SA / MedSAM2 / nnU-Net 中至少两个。 |

## 2. 当前 Phase2-lite 基础

- 已评估 manifest：`runs\phase2_lite\manifests\phase2_lite_40_cases.csv`。
- 当前病例数：40，数据集分布：{'IDRID': 20, 'MonuSeg': 20}。
- 当前有效 metric 行：720。

| 模型 | Mean Dice | 有效行数 |
| --- | ---: | ---: |
| medsam_vit_b | 0.2475 | 160 |
| sam2_tiny | 0.2195 | 280 |
| sam_vit_b | 0.2287 | 280 |

## 3. 扩数据验收结论

扩数据不是简单堆样本，而是要覆盖不同医学边界失败模式：眼底小病灶、病理密集核、超声/内镜/OCT 等低对比和伪影场景。当前已经生成数据库存和 MedSAM demo 适配清单。

| 来源 | 病例/文件概况 | 进入 Phase3 的用途 |
| --- | --- | --- |
| IDRID | image-like=196, mask-like=105 | 扩展 Phase2-lite 到更多眼底病例；适合测试小目标、多实例和低对比病灶。 |
| MonuSeg | image-like=329, mask-like=329 | 扩展核分割病例；适合测试密集实例和边界粘连。 |
| MedSAM demo npz | image-like=10, mask-like=20 | 作为跨模态 sanity set，不适合作为最终主表，但适合快速验收 domain generalization 管线。 |
| FSS-1000 | image-like=0, mask-like=0 | 非医学，只能作为 Matcher/support 机制的工程回归测试，不进入医学主表。 |

## 4. 强基线验收结论

第 5 个验收标准目前完成了强基线 readiness audit：能清楚区分已经可跑、已有权重待接入、仍缺仓库/权重的模型。论文主张仍必须等 P0 强基线补齐后再收紧。

| Baseline | 状态 | 仓库 | 权重/输出 | 下一步 |
| --- | --- | ---: | ---: | --- |
| SAM ViT-B | done_phase2_lite | 0 | 1 | 已进入统一接口和 Phase2-lite。 |
| SAM ViT-H | checkpoint_ready | 0 | 1 | 权重存在；可作为更强 SAM backbone 的下一轮对照。 |
| MedSAM ViT-B | done_phase2_lite | 1 | 1 | 已进入统一接口和 Phase2-lite。 |
| Lite-MedSAM | checkpoint_ready | 1 | 1 | 权重存在；适合速度/轻量化对照。 |
| SAM2 Tiny | done_phase2_lite | 1 | 1 | 已进入统一接口和 Phase2-lite。 |
| Matcher | repo_and_ckpt_ready | 1 | 1 | 已有仓库和 DINOv2/IDRID backbone；需要纳入同一 metric 主表。 |
| SAM3 | demo_ready_not_box_point_baseline | 1 | 1 | 权重存在；更适合作为 text/concept 分割对照，不直接等价 SAM box/point。 |
| SAM-Med2D | missing | 0 | 0 | P0 强基线，需要下载/接入。 |
| Med-SA / Medical SAM Adapter | missing | 0 | 0 | P0 强基线，需要下载/接入。 |
| MedSAM2 / Medical SAM2 | missing | 0 | 0 | P0 视频/3D/跨切片方向强基线，需要下载/接入。 |
| nnU-Net | missing | 0 | 0 | 监督式医学分割强参照；需要训练 split 和推理脚本。 |

## 5. 现在可以站住的结论

OCSAM-X 已经从单一 prompt selector 推进为一个可复现实验链条：support-guided prompt selection、prompt-induced uncertainty proxy、cross-model calibrated portfolio，再加上扩数据和强基线 readiness audit。

但当前还不是顶会级最终结论。真正的 Phase3 连接点是：在更大医学数据、多强基线、真实 prompt jitter uncertainty 下，证明这种 support + uncertainty + cross-model calibration 的组合能稳定优于固定 prompt、单模型 SAM/MedSAM/SAM2，以及至少部分近年医学 SAM adapter baseline。
