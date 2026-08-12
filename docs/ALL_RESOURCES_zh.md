# 完整资源安装指南

**语言：** [English](ALL_RESOURCES.md) | [中文](ALL_RESOURCES_zh.md)

本文档覆盖工作台和全部 `external/` 模块的代码、权重和数据集。分类含义：**必需**表示运行 OnePrompt-All 所需；**可选 UI** 表示只用于启用额外工作台模型；**研究专用**表示仅复现某个外部基线，不是 OnePrompt-All 所需。不要提交权重和原始数据。

## A. 工作台核心资源

| 范围 | 组件 | 获取方式 | 本地位置 | 说明 |
|---|---|---|---|---|
| 必需 | SAM3 代码 | `git clone https://github.com/facebookresearch/sam3.git external/sam3` | `external/sam3/` | `pip install -e external/sam3` |
| 必需 | SAM3 图像 checkpoint | [SAM3 HF](https://huggingface.co/facebook/sam3) | `assets/checkpoints/sam3_modelscope/sam3.pt` | 申请权限、`hf auth login`；权重和代码版本必须匹配。 |
| 必需 | DINOv2 ViT-B/14 | [DINOv2](https://github.com/facebookresearch/dinov2#pretrained-models) | `assets/checkpoints/dinov2_vitb14_pretrain.pth` | 下载 backbone，不是分类头。 |
| 可选 UI | SAM ViT-B/L/H | [SAM checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints) | `assets/checkpoints/sam_vit_b.pth`、`sam_vit_l_0b3195.pth`、`sam_vit_h_4b8939.pth` | 启用 SAM Series 选择项。 |
| 可选 UI | SAM2.1 Tiny/Small/Base+/Large | [SAM2 checkpoints](https://github.com/facebookresearch/sam2#download-checkpoints) | `assets/checkpoints/sam2.1_hiera_{tiny,small,base_plus,large}.pt` | 使用匹配的 SAM2 代码。 |
| 可选 UI | MedSAM ViT-B | [MedSAM release](https://github.com/bowang-lab/MedSAM#model-checkpoint) | `assets/checkpoints/medsam_vit_b.pth` | 启用 MedSAM。 |
| 可选 UI | Matcher 权重 | [Matcher](https://github.com/aim-uofa/Matcher) | `assets/checkpoints/swint_only_sam_many2many.pth` | 对应上游 Matcher 版本。 |

## B. external 模块资源矩阵

| 模块 | 代码 | 权重 | 数据与建议位置 | 范围 |
|---|---|---|---|---|
| `FoB_SAM` | [项目页](https://huggingface.co/PrimeBo1/FoB_SAM) | SAM ViT-H、ResNet-101 或 DeepLabV3-ResNet101；可选 FoB HF checkpoint | Abd-MRI/Abd-CT 按[上游流程](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation)预处理；Skin-DS/ISIC 放至 `external/FoB_SAM/data/` | 研究专用 |
| `FSPL` | 本地工件快照 | 未发现独立权重 | 已存工件无需下载；原始数据按 manifest | 本地资源 |
| `GBMSeg` | [GBMSeg](https://github.com/SnowRain510/GBMSeg) | DINOv2 backbone + SAM checkpoint | `external/GBMSeg/data/{reference_images,reference_masks,target_images}/` | 研究专用 |
| `legacy_scripts` | 本地脚本 | 取决于脚本，通常 nnU-Net/SAM/DINO | 按脚本 `-data` 参数和下方数据集表 | 本地资源 |
| `Matcher` | [Matcher](https://github.com/aim-uofa/Matcher) | `swint_only_sam_many2many.pth`、SAM、DINOv2 | 依据 `external/Matcher/datasets/README.md` 下载 COCO-20i、FSS-1000、Pascal-5i、Pascal-Part、LVIS、PACO-Part、MoNuSeg、IDRiD | 可选 UI / 研究 |
| `Medical-SAM-Adapter` | [项目页](https://github.com/KidsWithTokens/Medical-SAM-Adapter) | SAM ViT-B；可选 [Adapter Zoo](https://huggingface.co/KidsWithTokens/Medical-Adapter-Zoo/tree/main) | ISIC、REFUGE、LIDC-IDRI、BTCV；按 `guidance/Dataset.md` | 研究专用 |
| `MedSAM` | [MedSAM](https://github.com/bowang-lab/MedSAM) | [MedSAM checkpoint](https://github.com/bowang-lab/MedSAM#model-checkpoint)；可选原始 SAM ViT-B | [FLARE22 demo](https://zenodo.org/record/7860267) 解压至 `external/MedSAM/data/FLARE22Train/` | 可选 UI / 研究 |
| `nnUNet` | [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) | 无通用预训练权重，按数据集训练或另寻任务权重 | 设置 `nnUNet_raw`、`nnUNet_preprocessed`、`nnUNet_results` | 研究专用 |
| `ocsam_demo` | 本地历史快照 | 由 SAM/DINO/Matcher 配置决定 | 本地 demo 样例，无独立下载 | 本地资源 |
| `PPO` | [PPO](https://github.com/SnowRain510/PPO) | [PPO model](https://drive.google.com/file/d/1elAw4iagw4TYHsD0zWjJZnn9vJUHcZ0H/view) + SAM + DINO | `external/PPO/data/{reference_image,reference_mask,target_image,initial_indices,optimized_indices}/` | 研究专用 |
| `research_data` | 本地整理目录 | 无 | 原始数据到 `external/research_data/raw/`，预处理结果到 `external/research_data/extracted/` | 研究专用 |
| `SAM-Med2D` | [SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D) | 项目发布的 SAM-Med2D checkpoint；可选 base SAM | [SA-Med2D-20M](https://huggingface.co/datasets/OpenGVLab/SA-Med2D-20M) 或 `data_demo`，配置 `data_path` | 研究专用 |
| `sam2` | [SAM2](https://github.com/facebookresearch/sam2) | `checkpoints/download_ckpts.sh` 或 HF 下载 SAM2.1 | 训练/评估时可选 [SA-V](https://ai.meta.com/datasets/segment-anything-video/) | 可选 UI / 研究 |
| `sam3` | [SAM3](https://github.com/facebookresearch/sam3) | [SAM3 HF](https://huggingface.co/facebook/sam3)，SAM3.1 为[独立页面](https://huggingface.co/facebook/sam3.1) | 评测时可选 [SA-Co](https://huggingface.co/datasets/facebook/SACo-Gold) | 必需代码 / 数据可选 |
| `Segment-Any-Tissue` | 本地快照 | 本地 README 未给出可验证公开权重链接 | 本地 README 未给出可验证公开数据清单 | 等待官方发布 |
| `assets` | 已含少量 COCO 图 | 无 | 如需刷新，从 COCO 下载图像并放 `external/assets/coco_test_images/` | 可选 UI |

## C. 数据集下载矩阵

| 数据集 | 下载地址 | 建议目录 | 使用模块 |
|---|---|---|---|
| COCO 2017 | [COCO](https://cocodataset.org/#download) | `datasets/coco/` 或 `external/Matcher/datasets/coco/` | 工作台、Matcher |
| LVIS v1 | [LVIS](https://www.lvisdataset.org/dataset) + COCO 图像 | `datasets/LVIS/` 或 `external/Matcher/datasets/lvis/` | 工作台、Matcher |
| FSS-1000 | [FSS-1000](https://github.com/HKBU-LAG/FSS-1000) | `external/Matcher/datasets/FSS-1000/` | Matcher |
| Pascal VOC / Pascal-Part | [VOC](http://host.robots.ox.ac.uk/pascal/VOC/) / [Pascal-Part](http://roozbehm.info/pascal-parts/pascal-parts.html) | `external/Matcher/datasets/` | Matcher |
| PACO | [PACO](https://github.com/facebookresearch/paco) | `external/Matcher/datasets/PACO/` | Matcher |
| MoNuSeg / MoNuSAC | [MoNuSeg](https://monuseg.grand-challenge.org/) / [MoNuSAC](https://monusac-2020.grand-challenge.org/) | `datasets/medical/MoNuSeg/`、`MoNuSAC/` | 工作台、研究 |
| IDRiD | [IDRiD](https://idrid.grand-challenge.org/) | `datasets/medical/IDRiD/` | 工作台、Matcher |
| GlaS | [Warwick GlaS](https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/) | `external/research_data/raw/GlaS/` | 研究 |
| ISIC / REFUGE / LIDC-IDRI | [ISIC](https://www.isic-archive.com/) / [REFUGE](https://refuge.grand-challenge.org/) / [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/) | 各外部模块 `data/` | FoB-SAM、Adapter |
| BTCV / FLARE22 | [BTCV](https://www.synapse.org/#!Synapse:syn3193805/wiki/217752) / [FLARE22](https://flare22.grand-challenge.org/) | Adapter `data/` / `external/MedSAM/data/FLARE22Train/` | Adapter、MedSAM |
| SA-Med2D-20M | [Hugging Face](https://huggingface.co/datasets/OpenGVLab/SA-Med2D-20M) | 用户配置的 `data_path` | SAM-Med2D |
| SAM2 SA-V / SAM3 SA-Co | [SA-V](https://ai.meta.com/datasets/segment-anything-video/) / [SA-Co Gold](https://huggingface.co/datasets/facebook/SACo-Gold) | 各自上游 data 目录 | SAM2/SAM3 评测 |
| Abd-MRI / Abd-CT / Skin-DS | [FoB-SAM 指南](https://huggingface.co/PrimeBo1/FoB_SAM) | `external/FoB_SAM/data/` | FoB-SAM |
| GBM TEM | 本地快照不重新分发 | `external/GBMSeg/data/` | GBMSeg |

## 全量安装顺序

1. 先只安装计划运行的模块代码与环境。
2. 先下载 OnePrompt-All 必需的 SAM3 和 DINOv2-B/14，再补全 UI/基线所需权重。
3. 每个数据集只为对应的实验下载，并按模块配置更新路径。
4. 先执行上游 README 的最小验证，再运行大规模评测。

所有资源均有独立许可证、注册条件和使用协议；请仅从官方来源下载并遵守其条款。
