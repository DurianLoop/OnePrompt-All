# 外部依赖与参考文献

**语言：** [English](EXTERNAL_DEPENDENCIES.md) | [中文](EXTERNAL_DEPENDENCIES_zh.md)

本页记录 `external/` 下全部目录的来源和引用要求。它并不表示每个模块都参与 OnePrompt-All 的推理流程。

## 依赖登记表

| 本地目录 | 工作区用途 | 上游代码/数据 | 引用状态 |
|---|---|---|---|
| `assets/` | COCO 快速测试图像 | [COCO](https://cocodataset.org/) | 引用 COCO 数据集 |
| `FoB_SAM/` | 小样本医学基线 | [代码与模型](https://huggingface.co/PrimeBo1/FoB_SAM) | FoB-SAM, CVPR 2026 |
| `FSPL/` | 本地实验工件快照 | 当前副本未记录上游 | 本地资源，不作为外部工作引用 |
| `GBMSeg/` | 肾小球基底膜单样本基线 | [项目代码](https://github.com/SnowRain510/GBMSeg) | Liu et al., MICCAI 2024 |
| `legacy_scripts/` | 本地数据准备/nnU-Net 辅助脚本 | 无独立上游项目 | 引用底层数据集与 nnU-Net |
| `Matcher/` | 跨图单样本匹配基线 | [Matcher](https://github.com/aim-uofa/Matcher) | Liu et al., ICLR 2024 |
| `Medical-SAM-Adapter/` | 医学 SAM 适配基线 | [Medical-SAM-Adapter](https://github.com/KidsWithTokens/Medical-SAM-Adapter) | Wu et al., 2023 |
| `MedSAM/` | 医学 SAM 基线 | [MedSAM](https://github.com/bowang-lab/MedSAM) | Ma et al., 2024 |
| `nnUNet/` | 监督医学分割基线 | [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) | Isensee et al., 2021/2024 |
| `ocsam_demo/` | 历史本地 Demo 快照 | 来自本地 OCSAM/Matcher 工作台 | 本地资源；按需引用 Matcher/SAM/DINOv2 |
| `PPO/` | 点提示优化器基线 | [PPO 代码](https://github.com/SnowRain510/PPO) | Plug-and-Play PPO, CVPR 2025 |
| `research_data/` | 本地整理研究数据 | 下方逐数据集来源 | 引用原始数据集 |
| `SAM-Med2D/` | 医学二维 SAM 基线和 SA-Med2D 数据 | [SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D) | Cheng et al., 2023 |
| `sam2/` | SAM2 图像/视频基线 | [SAM2](https://github.com/facebookresearch/sam2) | Ravi et al., 2024 |
| `sam3/` | SAM3 图像/文本/视觉提示骨干 | [SAM3](https://github.com/facebookresearch/sam3) | Carion et al., 2025 |
| `Segment-Any-Tissue/` | Segment Any Tissue 本地快照 | README 未记录公开代码发布链接 | 等官方发布信息后再正式引用 |

## 推荐引用原则

- 仅引用实际参与某个结果、图表或比较的模块/数据集。
- OnePrompt-All 结果至少引用 **SAM3** 与 **DINOv2**。
- 只有在实验中实际使用时才引用 SAM、SAM2、Matcher 或医学基线。
- 当前仓库是持续开发的软件；不要将 OnePrompt-All 的占位条目当成已发表论文引用。待获得正式预印本或 DOI 后再添加。

英文版包含完整 BibTeX 条目和全部数据集链接，请见 [English dependency registry](EXTERNAL_DEPENDENCIES.md)。
