# External Dependencies and References

**Language:** [English](EXTERNAL_DEPENDENCIES.md) | [中文](EXTERNAL_DEPENDENCIES_zh.md)

This page records provenance for every directory under `external/`. It is a
software-and-data dependency list, not a claim that every external module is
used by the OnePrompt-All inference path.

## Dependency registry

| Local directory | Role in this workspace | Upstream code / data | Citation status |
|---|---|---|---|
| `assets/` | Small COCO quick-test images | [COCO](https://cocodataset.org/) | Dataset citation below |
| `FoB_SAM/` | Few-shot medical baseline | [Code and model](https://huggingface.co/PrimeBo1/FoB_SAM) | FoB-SAM, CVPR 2026 |
| `FSPL/` | Local experiment-artifact snapshot | No upstream source recorded in this copy | Local-only; do not cite as external work |
| `GBMSeg/` | One-shot GBM segmentation baseline | [Project code](https://github.com/SnowRain510/GBMSeg) | Liu et al., MICCAI 2024 |
| `legacy_scripts/` | Local data preparation and nnU-Net helper scripts | No standalone upstream project | Local-only; cite underlying datasets/nnU-Net |
| `Matcher/` | Cross-image one-shot matcher baseline | [Matcher](https://github.com/aim-uofa/Matcher) | Liu et al., ICLR 2024 |
| `Medical-SAM-Adapter/` | Medical SAM adaptation baseline | [Medical-SAM-Adapter](https://github.com/KidsWithTokens/Medical-SAM-Adapter) | Wu et al., 2023 |
| `MedSAM/` | Medical SAM baseline | [MedSAM](https://github.com/bowang-lab/MedSAM) | Ma et al., 2024 |
| `nnUNet/` | Supervised medical segmentation baseline | [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) | Isensee et al., 2021/2024 |
| `ocsam_demo/` | Historical local demo snapshot | Derived from local OCSAM/Matcher workbench | Local-only; cite Matcher/SAM/DINOv2 when applicable |
| `PPO/` | Point-prompt optimizer baseline | [PPO code](https://github.com/SnowRain510/PPO) | Plug-and-Play PPO, CVPR 2025 |
| `research_data/` | Locally materialized research datasets | Dataset-specific sources listed below | Cite each original dataset |
| `SAM-Med2D/` | Medical 2D SAM baseline and SA-Med2D data | [SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D) | Cheng et al., 2023 |
| `sam2/` | SAM2 image/video baseline | [SAM2](https://github.com/facebookresearch/sam2) | Ravi et al., 2024 |
| `sam3/` | SAM3 image/text/visual-prompt backbone | [SAM3](https://github.com/facebookresearch/sam3) | Carion et al., 2025 |
| `Segment-Any-Tissue/` | Local Segment Any Tissue snapshot | No public code release URL recorded in its README | Cite only after official release metadata is available |

## Recommended citation practice

- Cite only the modules, datasets, and pretrained models actually used in a
  result or comparison.
- For OnePrompt-All results, cite **SAM3** and **DINOv2** at minimum.
- Add **SAM**, **SAM2**, **Matcher**, or medical baselines only when those
  methods appear in an experiment, figure, or software comparison.
- This repository is software under active development. Do **not** cite a
  placeholder OnePrompt-All entry as if it were a published paper. Add the
  final project preprint/DOI only after it exists.

## Reference list

```bibtex
@article{sam3_2025,
  title={SAM 3: Segment Anything with Concepts},
  author={Carion, Nicolas and Gustafson, Laura and Hu, Yuan-Ting and others},
  journal={arXiv preprint arXiv:2511.16719}, year={2025}
}
@article{dinov2_2024,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothee and Moutakanni, Theo and others},
  journal={Transactions on Machine Learning Research}, year={2024}
}
@inproceedings{sam_2023,
  title={Segment Anything}, author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and others},
  booktitle={ICCV}, year={2023}
}
@article{sam2_2024,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and others},
  journal={arXiv preprint arXiv:2408.00714}, year={2024}
}
@inproceedings{matcher_2024,
  title={Matcher: Segment Anything with One Shot Using All-Purpose Feature Matching},
  author={Liu, Yang and Zhu, Muzhi and Li, Hengtao and others}, booktitle={ICLR}, year={2024}
}
@article{medsam_2024,
  title={Segment Anything in Medical Images}, author={Ma, Jun and He, Yuting and Li, Feifei and others},
  journal={Nature Communications}, year={2024}
}
@article{medicalsamadapter_2023,
  title={Medical SAM Adapter: Adapting Segment Anything Model for Medical Image Segmentation},
  author={Wu, Junde and Zhang, Yu and others}, journal={arXiv preprint arXiv:2304.12620}, year={2023}
}
@article{nnunet_2021,
  title={nnU-Net: A Self-configuring Method for Deep Learning-based Biomedical Image Segmentation},
  author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon AA and others}, journal={Nature Methods}, year={2021}
}
@article{nnunet_2024,
  title={nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation},
  author={Isensee, Fabian and others}, journal={arXiv preprint arXiv:2404.09556}, year={2024}
}
@inproceedings{gbmseg_2024,
  title={Feature-prompting GBMSeg: One-Shot Reference Guided Training-Free Prompt Engineering for Glomerular Basement Membrane Segmentation},
  author={Liu, Xueyu and Shi, Guangze and Wang, Rui and others}, booktitle={MICCAI}, year={2024}
}
@inproceedings{fobsam_2026,
  title={Focus on the Background: Exploring SAM's Potential in Few-shot Medical Image Segmentation},
  author={Bo, Prime and others}, booktitle={CVPR}, year={2026}
}
@inproceedings{ppo_2025,
  title={Plug-and-Play PPO: An Adaptive Point Prompt Optimizer Making SAM Greater},
  author={PPO Authors}, booktitle={CVPR}, year={2025}
}
@article{sammed2d_2023,
  title={SAM-Med2D}, author={Cheng, Yuxin and others}, journal={arXiv preprint arXiv:2308.16184}, year={2023}
}
@inproceedings{coco_2014,
  title={Microsoft COCO: Common Objects in Context}, author={Lin, Tsung-Yi and others}, booktitle={ECCV}, year={2014}
}
@inproceedings{lvis_2019,
  title={LVIS: A Dataset for Large Vocabulary Instance Segmentation}, author={Gupta, Agrim and Dollar, Piotr and Girshick, Ross},
  booktitle={CVPR}, year={2019}
}
@article{monuseg_2018,
  title={A Multi-organ Nucleus Segmentation Challenge}, author={Kumar, Neeraj and others}, journal={IEEE TMI}, year={2018}
}
@article{idrid_2018,
  title={Indian Diabetic Retinopathy Image Dataset (IDRiD)}, author={Porwal, Prasanna and others}, journal={Medical Image Analysis}, year={2018}
}
@article{glas_2016,
  title={Gland Segmentation in Colon Histology Images: The GlaS Challenge Contest}, author={Sirinukunwattana, Korsuk and others}, journal={Medical Image Analysis}, year={2017}
}
```

## Dataset sources used by external modules

| Dataset | Official source |
|---|---|
| COCO 2017 | [cocodataset.org](https://cocodataset.org/#download) |
| LVIS v1 | [lvisdataset.org](https://www.lvisdataset.org/dataset) |
| FSS-1000 | [official project](https://github.com/HKBU-LAG/FSS-1000) |
| Pascal VOC / Pascal-Part | [VOC](http://host.robots.ox.ac.uk/pascal/VOC/) / [Pascal-Part](http://roozbehm.info/pascal-parts/pascal-parts.html) |
| PACO | [PACO](https://github.com/facebookresearch/paco) |
| MoNuSeg | [challenge site](https://monuseg.grand-challenge.org/) |
| IDRiD | [challenge site](https://idrid.grand-challenge.org/) |
| GlaS | [Warwick archive](https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/) |
| ISIC | [ISIC Archive](https://www.isic-archive.com/) |
| REFUGE | [REFUGE](https://refuge.grand-challenge.org/) |
| BTCV | [Synapse BTCV](https://www.synapse.org/#!Synapse:syn3193805/wiki/217752) |
| FLARE22 | [FLARE22](https://flare22.grand-challenge.org/) |
| SA-Med2D-20M | [Hugging Face](https://huggingface.co/datasets/OpenGVLab/SA-Med2D-20M) |
| SAM2 SA-V | [Meta dataset page](https://ai.meta.com/datasets/segment-anything-video/) |
| SAM3 SA-Co | [SACo Gold](https://huggingface.co/datasets/facebook/SACo-Gold), [Silver](https://huggingface.co/datasets/facebook/SACo-Silver), [VEval](https://huggingface.co/datasets/facebook/SACo-VEval) |
