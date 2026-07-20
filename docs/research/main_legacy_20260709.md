# OCSAM 研究主线进度

日期：2026-07-09

## 2026-07-09 最新状态：官方 MoNuSeg 已补全到本地

这一步已经完成：我从官方 MoNuSeg Grand Challenge Data 页面给出的 Google Drive 链接下载了 training package、testing package、training organ information 和官方 XML-to-mask MATLAB 参考代码，并且把 XML 多边形标注转成了后续实验可直接使用的 binary nuclei mask。

本地位置：

```text
code/datasets/MonuSeg/official/downloads/
code/datasets/MonuSeg/official/raw/
code/datasets/MonuSeg/official/train/images/
code/datasets/MonuSeg/official/train/masks/
code/datasets/MonuSeg/official/test/images/
code/datasets/MonuSeg/official/test/masks/
runs/research_direction/dataset_audit/monuseg_official_manifest.csv
scripts/prepare_official_monuseg.py
```

需要注意一个很重要的数量差异：官方网页文字常说 training 30 images、test 14 images，但当前官方 Drive 训练 zip 实际包含 37 个真实 tif+xml 配对；test zip 包含 14 个真实 tif+xml 配对。因此我现在没有擅自删样本，而是完整保留为：

```text
official Drive package observed locally = 37 train + 14 test = 51 image-level pairs
manifest rows = 51
empty masks = 0
```

如果后续论文要严格写 30/14 challenge protocol，需要再固定一个 30-image training subset；如果是先做 broad screening，建议先使用当前官方包完整的 37/14。

## 当前一句话结论

目前项目已经从“老师问题的零散验证”切到新的主线：

```text
MonuSeg reference mask -> feature matching -> point/box prompt -> SAM-family / MedSAM-family segmenter -> 筛选最有潜力的模型和 backbone -> 再决定是否训练
```

现在还没有进入正式训练阶段。当前正在完成的是：

```text
数据 protocol 补齐 + 模型/backbone zoo 准备 + broad screening 前置准备
```

## 进度清单

| 状态 | 事项 | 结果/位置 |
| --- | --- | --- |
| [x] | 梳理新研究路线 | `docs/track/main_research_pipeline.md` |
| [x] | 确认当前实验实际规模 | 当前已跑：MonuSeg 20 + IDRID 30 = 50 pairs |
| [x] | 审计本地数据规模 | MonuSeg 88 pairs，IDRID 81 pairs |
| [x] | 生成本地数据审计表 | `runs/research_direction/dataset_audit/all_monu_valid_pairs.csv`；`all_idrid_valid_pairs.csv` |
| [x] | 补充官方 MoNuSeg protocol 说明 | `docs/track/official_monuseg_protocol.md` |
| [x] | 集中 7.9 后新路线文档 | 全部放到 `docs/track` |
| [x] | 建立 segmenter zoo 清单 | `docs/track/model_zoo_manifest.csv` |
| [x] | 建立 feature backbone zoo 清单 | `docs/track/feature_backbone_zoo_manifest.csv` |
| [x] | 下载 SAM ViT-L | `assets/checkpoints/sam_vit_l_0b3195.pth` |
| [x] | 下载 SAM2.1 Hiera Small/Base+/Large | `assets/checkpoints/sam2.1_hiera_*.pt` |
| [x] | 下载 DINOv2 ViT-B | `assets/checkpoints/dinov2_vitb14_pretrain.pth` |
| [x] | 下载 BioMedCLIP | `assets/checkpoints/biomedclip/open_clip_pytorch_model.bin` |
| [x] | 下载 SAM-Med2D | `assets/checkpoints/sam_med2d_b.pth` |
| [x] | 确认 SAM3 official HF 状态 | HF gated，需要 approval/token |
| [x] | 确认 UNI / CONCH 状态 | HF gated，需要 approval/token |
| [ ] | 下载/补齐 official full MoNuSeg 30/14 文件 | 待做 |
| [ ] | 生成 official 44-image manifest | 待做 |
| [ ] | 接入 SAM2.1 Large/Base+/Small 到 screening runner | 待做 |
| [ ] | 接入 SAM-Med2D 到 screening runner | 待做 |
| [ ] | 接入 BioMedCLIP feature extraction | 待做 |
| [ ] | 跑 MonuSeg segmenter zoo broad screening | 待做 |
| [ ] | 跑 feature backbone zoo screening | 待做 |
| [ ] | 根据 screening 收缩到 2-3 个模型组合 | 待做 |
| [ ] | 决定是否进入 adapter / selector / prompt policy 训练 | 待做 |

## 现在应该先看什么

1. `docs/track/main.md`  
   先看这里，确认项目当前状态。

2. `docs/track/official_monuseg_protocol.md`  
   再看这里，确认为什么当前数据还不能叫 official full MoNuSeg，以及后续怎么补齐。

3. `docs/track/main_research_pipeline.md`  
   看完整路线，包括 model zoo、feature zoo、prompt policy、baseline/消融实验设计。

4. `docs/track/model_zoo_manifest.csv`  
   看哪些 SAM / MedSAM 模型已经有权重，哪些还 gated。

5. `docs/track/feature_backbone_zoo_manifest.csv`  
   看 DINOv2、BioMedCLIP、UNI、CONCH 的状态。

## 当前最重要的下一步

下一步不是马上训练，而是先补齐 official MoNuSeg protocol：

```text
下载/核对 official 30 train + 14 test
-> 生成 official 44-image manifest
-> 再基于 image-level split 切 patch 扩大训练规模
```

完成这个之后，再进入：

```text
SAM2.1 / SAM-Med2D / SAM3 / SAM-BLH broad screening
BioMedCLIP / DINOv2 / UNI / CONCH feature screening
```

这样后续训练和论文主表才不会被“数据规模太小”卡住。
