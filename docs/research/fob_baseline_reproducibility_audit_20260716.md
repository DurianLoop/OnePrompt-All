# FoB-SAM 强基线复现审计

更新：2026-07-16  
目的：判断 CVPR 2026 FoB 是否能直接作为 MoNuSeg/MoNuSAC 同协议数值基线，以及本文应如何划定创新边界。

## 1. 审计结论

FoB 是当前最接近本文的工作，必须在相关工作中正面比较。它已经提出了从 support mask 学习背景提示、约束 SAM 过分割，并报告了跨域结果。因此，本文不能再把“使用 reference 背景”“自动生成负点”或“背景点改善 SAM”单独表述为创新。

当前官方代码**不能直接在 MoNuSeg/MoNuSAC 上公平推理**，原因不是代码不可用，而是任务协议和已发布权重不匹配：

- 官方仓库：`external/FoB_SAM/`
- 固定版本：commit `40d0e23d75f81f07bb7f34cb641db51ef490d025`
- 官方支持的数据：CHAOST2、SABS、ISIC2018
- Hugging Face 已发布权重：CHAOST2 与 SABS，共约 2.01 GB
- 未发布：MoNuSeg、MoNuSAC 或其他病理核权重
- 代码的数据加载、类别折和 2D/3D slice 协议围绕腹部器官与 ISIC 编写
- `models/encoder.py`、`SAM.py`、`test.py` 中仍需手动替换 ResNet/SAM/实验目录的绝对路径
- 官方默认使用 SAM ViT-H；本文当前主结果使用 SAM ViT-B，不能直接把两个数值放在同一栏宣称优劣

因此，当前可以写“FoB 是最接近工作，官方发布物不含病理核同协议模型”；不能写“本文优于 FoB”。

## 2. FoB 与本文的关系

| 维度 | FoB | 本文当前方法 |
| --- | --- | --- |
| 任务 | few-shot 医学分割，重点抑制边界过分割 | dense all-nuclei reference-conditioned 分割 |
| support 信息 | support image/mask | reference image/mask |
| 背景使用方式 | 学习背景提示定位和结构约束 | DINO 前景/背景原型匹配 + 负点预算门控 |
| 学习对象 | 背景 prompt localization 网络 | 冻结 matcher/SAM 后的 N0/N5 响应 gate |
| 分割器 | SAM ViT-H 与 S-2D | 当前主结果为 SAM ViT-B |
| 测试预算 | 由模型生成背景提示 | 两次 SAM：N0 与一个 challenger |
| 当前数据 | 腹部 MRI/CT、皮肤 | MoNuSeg、MoNuSAC 病理核 |
| 当前主要证据 | 三数据集与腹部跨域 | 37-query nested-CV + patient-disjoint 外部验证 |

本文剩余可辩护的新意不是“背景 prompt”，而是：

1. dense all-instance intent 下，reference 前景/背景证据、固定正点覆盖与背景负点预算的联合机制；
2. 不训练 SAM 和 matcher，只学习是否接受当前 query 的负点响应；
3. 用 validation anchor、case/patient-disjoint split、nested-CV 和 query/patient 统计防止候选 oracle 泄漏；
4. 公开跨域失败边界：零样本 gate 与域内 gate 都未超过 MoNuSAC 的事后全局固定 N5。

## 3. 公平复现的最低要求

若要把 FoB 放入 MoNuSeg 主表，不能直接使用腹部权重。最低公平协议为：

1. 新增 MoNuSeg image/mask episodic loader，support 与 query 必须遵守当前 nested-CV fold；
2. 每折只在 inner-train 训练 FoB，inner-val 选择 checkpoint/阈值，outer-test 只推理一次；
3. 固定同一 SAM checkpoint，优先同时报告 FoB+ViT-H 官方设定与 FoB+ViT-B 同执行器设定；
4. 输出 dense binary all-nuclei mask，并使用本文相同 Dice、IoU、object recall、merge error；
5. 每个 outer query 使用同样 5 个 train-only references，先按 query 聚合再做 37-query 配对统计；
6. 报告训练时长、显存、参数量与每 query SAM 调用数。

## 4. 投稿前状态

| 状态 | 项目 |
| --- | --- |
| [x] | 官方仓库与固定 commit 已保存 |
| [x] | 支持数据、权重和硬编码路径已审计 |
| [x] | 已确认不能直接使用腹部权重产生病理核公平数字 |
| [ ] | MoNuSeg episodic loader 适配 |
| [ ] | 至少完成一个 outer-fold 的训练与 smoke evaluation |
| [ ] | 根据单折成本决定是否完成 5-fold FoB 主表 |

在上述适配完成前，论文中只做方法边界讨论，不做数值胜负声明。

## 5. 来源

- 官方代码：https://github.com/primebo1/FoB_SAM
- 官方权重：https://huggingface.co/PrimeBo1/FoB_SAM
- CVPR 2026 论文：https://openaccess.thecvf.com/content/CVPR2026/html/Bo_Focus_on_Background_Exploring_SAMs_Potential_in_Few-shot_Medical_Image_CVPR_2026_paper.html
