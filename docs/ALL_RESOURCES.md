# Complete Resource Installation Guide

**Language:** [English](ALL_RESOURCES.md) | [中文](ALL_RESOURCES_zh.md)

This is the complete acquisition matrix for the workbench and all checked-out
external modules. It distinguishes three scopes:

- **Required**: needed to run OnePrompt-All.
- **Optional UI**: needed only to enable an additional workbench model.
- **Research-only**: needed to reproduce a particular external baseline or its
  published experiment; not needed for OnePrompt-All.

Do not commit checkpoints or raw datasets. The repository `.gitignore` excludes
the standard local locations below.

## A. Core workbench resources

| Scope | Component | Obtain | Local destination | Notes |
|---|---|---|---|---|
| Required | SAM3 code | `git clone https://github.com/facebookresearch/sam3.git external/sam3` | `external/sam3/` | Install with `pip install -e external/sam3`. |
| Required | SAM3 image checkpoint | [SAM3 HF](https://huggingface.co/facebook/sam3) | `assets/checkpoints/sam3_modelscope/sam3.pt` | Request access; use `hf auth login`; code/checkpoint versions must match. |
| Required | DINOv2 ViT-B/14 | [DINOv2](https://github.com/facebookresearch/dinov2#pretrained-models) | `assets/checkpoints/dinov2_vitb14_pretrain.pth` | Backbone checkpoint, not classifier head. |
| Optional UI | SAM ViT-B/L/H | [SAM checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints) | `assets/checkpoints/sam_vit_b.pth`, `sam_vit_l_0b3195.pth`, `sam_vit_h_4b8939.pth` | Needed for SAM Series choices. |
| Optional UI | SAM2.1 Tiny/Small/Base+/Large | [SAM2 checkpoints](https://github.com/facebookresearch/sam2#download-checkpoints) | `assets/checkpoints/sam2.1_hiera_{tiny,small,base_plus,large}.pt` | Use matching SAM2 code. |
| Optional UI | MedSAM ViT-B | [MedSAM Drive release](https://github.com/bowang-lab/MedSAM#model-checkpoint) | `assets/checkpoints/medsam_vit_b.pth` | Enables MedSAM mode. |
| Optional UI | Matcher model | [Matcher](https://github.com/aim-uofa/Matcher) | `assets/checkpoints/swint_only_sam_many2many.pth` | Match the upstream Matcher release. |

## B. External modules: code, weights, and data

| Local module | Code acquisition | Required weights | Required data / local layout | Scope |
|---|---|---|---|---|
| `FoB_SAM` | [FoB-SAM HF project](https://huggingface.co/PrimeBo1/FoB_SAM) | SAM ViT-H: `sam_vit_h_4b8939.pth`; ResNet-101 or DeepLabV3 ResNet-101 from links in its README; optional FoB checkpoints from HF | Abd-MRI/Abd-CT preprocessed per [upstream instructions](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation); Skin-DS/ISIC under `external/FoB_SAM/data/` | Research-only |
| `FSPL` | Local artifact snapshot; no upstream clone metadata | None identified | No download required for stored artifacts; original experiment data is governed by its manifests | Local-only |
| `GBMSeg` | [GBMSeg](https://github.com/SnowRain510/GBMSeg) | DINOv2 backbone + SAM checkpoint; use module config paths | `external/GBMSeg/data/{reference_images,reference_masks,target_images}/` | Research-only |
| `legacy_scripts` | Included local scripts | Depends on invoked script (usually nnU-Net/SAM/DINO) | Follow each script's `-data` / environment arguments; see datasets in Section C | Local-only |
| `Matcher` | [Matcher](https://github.com/aim-uofa/Matcher) | Matcher release weight `swint_only_sam_many2many.pth`; SAM backbone; DINOv2 backbone | Follow `external/Matcher/datasets/README.md` for COCO-20i, FSS-1000, Pascal-5i, Pascal-Part, LVIS, PACO-Part, MoNuSeg, IDRiD | Optional UI / research |
| `Medical-SAM-Adapter` | [Medical-SAM-Adapter](https://github.com/KidsWithTokens/Medical-SAM-Adapter) | SAM ViT-B `sam_vit_b_01ec64.pth`; optional task adapter from [Medical-Adapter-Zoo](https://huggingface.co/KidsWithTokens/Medical-Adapter-Zoo/tree/main) | ISIC, REFUGE, LIDC-IDRI, or BTCV as selected; follow `guidance/Dataset.md` | Research-only |
| `MedSAM` | [MedSAM](https://github.com/bowang-lab/MedSAM) | MedSAM checkpoint from [release](https://github.com/bowang-lab/MedSAM#model-checkpoint); optional original SAM ViT-B | Demo: [FLARE22 Zenodo](https://zenodo.org/record/7860267) to `external/MedSAM/data/FLARE22Train/`; full FLARE via official challenge | Optional UI / research |
| `nnUNet` | [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) | No universal pretrained weight; train per target dataset or acquire task-specific model separately | Set `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`; prepare each dataset per [official guide](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how-to/prepare_a_dataset.md) | Research-only |
| `ocsam_demo` | Local historical snapshot | Depends on SAM/DINO/Matcher configuration | Uses local demo samples; no independent required download | Local-only |
| `PPO` | [PPO](https://github.com/SnowRain510/PPO) | [PPO model](https://drive.google.com/file/d/1elAw4iagw4TYHsD0zWjJZnn9vJUHcZ0H/view); SAM and DINO backbones | `external/PPO/data/{reference_image,reference_mask,target_image,initial_indices,optimized_indices}/` | Research-only |
| `research_data` | Local materialization | None | Download original data listed in Section C into `external/research_data/raw/`, then preprocess to `external/research_data/extracted/` | Research-only |
| `SAM-Med2D` | [SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D) | SAM-Med2D checkpoint from its project release; optional base SAM checkpoint | [SA-Med2D-20M](https://huggingface.co/datasets/OpenGVLab/SA-Med2D-20M) or project `data_demo`; configure `data_path` | Research-only |
| `sam2` | [SAM2](https://github.com/facebookresearch/sam2) | Download desired SAM2.1 checkpoint using `external/sam2/checkpoints/download_ckpts.sh` or HF | Optional [SA-V](https://ai.meta.com/datasets/segment-anything-video/) only for training/evaluation | Optional UI / research |
| `sam3` | [SAM3](https://github.com/facebookresearch/sam3) | [SAM3 HF](https://huggingface.co/facebook/sam3); SAM3.1 is [separate](https://huggingface.co/facebook/sam3.1) | Optional [SA-Co](https://huggingface.co/datasets/facebook/SACo-Gold) variants only for SAM3 evaluation | Required code / research data optional |
| `Segment-Any-Tissue` | Local snapshot; its README states code release is pending | No verified public checkpoint link in local README | No verified public dataset-release manifest in local README | Local-only pending official release |
| `assets` | Included small COCO samples | None | To refresh: download COCO images and copy selected files to `external/assets/coco_test_images/` | Optional UI |

## C. Dataset acquisition matrix

| Dataset | Download / registration | Suggested local target | Used by |
|---|---|---|---|
| COCO 2017 | [COCO download](https://cocodataset.org/#download) | `datasets/coco/` or `external/Matcher/datasets/coco/` | Workbench, Matcher, quick test |
| LVIS v1 | [LVIS dataset](https://www.lvisdataset.org/dataset) + COCO images | `datasets/LVIS/` or `external/Matcher/datasets/lvis/` | Workbench, Matcher |
| FSS-1000 | [FSS-1000](https://github.com/HKBU-LAG/FSS-1000) | `external/Matcher/datasets/FSS-1000/` | Matcher |
| Pascal VOC 2012 | [VOC](http://host.robots.ox.ac.uk/pascal/VOC/) | `external/Matcher/datasets/VOC2012/` | Pascal-5i |
| Pascal-Part | [Pascal-Part](http://roozbehm.info/pascal-parts/pascal-parts.html) | `external/Matcher/datasets/PascalPart/` | Matcher |
| PACO | [PACO](https://github.com/facebookresearch/paco) | `external/Matcher/datasets/PACO/` | Matcher |
| MoNuSeg | [MoNuSeg challenge](https://monuseg.grand-challenge.org/) | `datasets/medical/MoNuSeg/` | Workbench, Matcher, nnU-Net |
| MoNuSAC | [MoNuSAC challenge](https://monusac-2020.grand-challenge.org/) | `datasets/medical/MoNuSAC/` | Research baselines |
| IDRiD | [IDRiD challenge](https://idrid.grand-challenge.org/) | `datasets/medical/IDRiD/` | Workbench, Matcher |
| GlaS | [Warwick GlaS](https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/) | `external/research_data/raw/GlaS/` | Research data |
| ISIC | [ISIC Archive](https://www.isic-archive.com/) | `external/FoB_SAM/data/isic/` or `external/Medical-SAM-Adapter/data/isic/` | FoB-SAM, Adapter |
| REFUGE | [REFUGE](https://refuge.grand-challenge.org/) | `external/Medical-SAM-Adapter/data/REFUGE-MultiRater/` | Adapter |
| LIDC-IDRI | [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/) | User-defined adapter data root | Adapter |
| BTCV | [Synapse BTCV](https://www.synapse.org/#!Synapse:syn3193805/wiki/217752) | `external/Medical-SAM-Adapter/data/` | Adapter |
| FLARE22 | [FLARE22](https://flare22.grand-challenge.org/) / [demo Zenodo](https://zenodo.org/record/7860267) | `external/MedSAM/data/FLARE22Train/` | MedSAM |
| SA-Med2D-20M | [Hugging Face](https://huggingface.co/datasets/OpenGVLab/SA-Med2D-20M) | user-defined `data_path` | SAM-Med2D |
| SAM2 SA-V | [Meta SA-V](https://ai.meta.com/datasets/segment-anything-video/) | `external/sam2/sav_dataset/` per upstream docs | SAM2 training/eval |
| SAM3 SA-Co | [Gold](https://huggingface.co/datasets/facebook/SACo-Gold), [Silver](https://huggingface.co/datasets/facebook/SACo-Silver), [VEval](https://huggingface.co/datasets/facebook/SACo-VEval) | `external/sam3/data/` per upstream docs | SAM3 evaluation |
| Abd-MRI / Abd-CT / Skin-DS | [FoB-SAM instructions](https://huggingface.co/PrimeBo1/FoB_SAM) | `external/FoB_SAM/data/` | FoB-SAM |
| GBM TEM | Not publicly redistributed in this local snapshot | `external/GBMSeg/data/` | GBMSeg |

## Download order for a full local installation

1. Install code and environments for only the modules you will run.
2. Download the four workbench UI checkpoints you need; at minimum SAM3 and DINOv2-B/14 for OnePrompt-All.
3. Download each dataset only for the corresponding baseline or evaluation.
4. Update the absolute/relative paths in that external project's config or script.
5. Run the verification command in the upstream README before launching a large evaluation.

The resources above have independent licenses, registration conditions, and
usage agreements. Download them only from their official sources and comply
with their terms.
