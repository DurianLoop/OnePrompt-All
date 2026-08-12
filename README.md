# OnePrompt-All

**One rough prompt, all same-class instances in one image.**
**一次粗提示，分割单张图中的全部同类实例。**

[English](#english) | [中文](#中文) | [Poster](#poster--editable-source) | [References](#references)

---

<a id="english"></a>
## English

### Overview

OnePrompt-All is a training-free framework and interactive workbench for
**single-image, single-exemplar, multi-instance segmentation**. A user clicks
one object, draws a loose box, or circles an approximate region. The system
infers the intended visual concept within that same image and returns a
separate mask for every matching instance—without a class name, external
reference image, reference mask, or object count.

![SAM Prompt Benchmark workbench](docs/assets/fig3_workbench.jpg)

The unified workbench retains three modes: **SAM Series**, **Matcher Series**,
and **OnePrompt-All**. This makes prompted segmentation, cross-image reference
matching, and same-image multi-instance segmentation directly comparable in a
single browser interface.

### Workbench usage

Run `surf/demos/medical_sam_click_app.py`, select an image, switch to
**OnePrompt-All**, and choose one prompt type:

1. **Click**: click anywhere inside one target instance.
2. **Box**: draw a loose rectangle; it may include background.
3. **Circle**: drag a rough circular region; it may include background.

One gesture triggers one complete inference. The result panel displays each
accepted instance as an independent colored mask; no iterative user
positive/negative clicks are required.

### Environment setup

The tested configuration is Windows, Python 3.12, CUDA 12.6+ and a
CUDA-capable GPU. Follow the official [SAM3 installation guide](https://github.com/facebookresearch/sam3#installation)
when selecting your PyTorch/CUDA build.

```powershell
# 1. Create an environment
conda create -n oneprompt-all python=3.12 -y
conda activate oneprompt-all

# 2. Install a CUDA build of PyTorch (example; match it to your system)
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. Install web/workbench dependencies
pip install -r requirements-demo.txt

# 4. Install SAM3 alongside this repository
git clone https://github.com/facebookresearch/sam3.git external/sam3
pip install -e external/sam3
```

`requirements-demo.txt` contains the web/UI dependencies. Keep the PyTorch
version required by SAM3 if a conflict occurs; `surf/requirements.txt` contains
legacy dependencies for the original Matcher code and should not override it.

### Checkpoints

Weights are not distributed in this repository. Create this local layout:

```text
assets/
└── checkpoints/
    ├── sam3_modelscope/
    │   └── sam3.pt
    └── dinov2_vitb14_pretrain.pth
```

| Component | Expected local file | Download source |
|---|---|---|
| SAM3 instance generator | `assets/checkpoints/sam3_modelscope/sam3.pt` | Request access at the official [SAM3 Hugging Face repository](https://huggingface.co/facebook/sam3), authenticate with `hf auth login`, and download a compatible image checkpoint. Rename it to `sam3.pt`, or supply `--sam3-checkpoint <path>`. |
| DINOv2-B/14 feature encoder | `assets/checkpoints/dinov2_vitb14_pretrain.pth` | Download the ViT-B/14 backbone from official [DINOv2 pretrained models](https://github.com/facebookresearch/dinov2#pretrained-models) and save it with this filename. |

Other SAM, SAM2, MedSAM, and Matcher checkpoints are optional and only enable
their corresponding workbench modes.

### Launch

From the repository root:

```powershell
python .\surf\demos\medical_sam_click_app.py --host 127.0.0.1 --port 7860
```

Open <http://127.0.0.1:7860>. If a server is already listening on port 7860,
reuse it instead of launching a second instance.

For a non-default SAM3 location:

```powershell
python .\surf\demos\medical_sam_click_app.py `
  --host 127.0.0.1 --port 7860 `
  --sam3-checkpoint D:\models\sam3.pt
```

`surf/datasets/oneprompt_all/natural20/` contains 20 small natural-image
examples for immediate browsing. Full datasets, checkpoints, and generated
outputs remain local and are excluded from Git.

### Method

![OnePrompt-All architecture](docs/assets/fig1_architecture2.jpg)

Given a coarse prompt \(p\) in image \(I\), OnePrompt-All constructs an online
exemplar rather than using an external reference:

1. **Prompt parsing and seed resolution**: SAM3 converts the point, loose box,
   or circle into a seed mask.
2. **Feature purification**: DINOv2-B/14 tokens inside the seed form an
   image-specific exemplar representation.
3. **Same-image candidate discovery**: SAM3 produces image-wide instance
   candidates; DINO similarity is used as soft evidence rather than a hard
   veto.
4. **Candidate-set selection**: SAM confidence, soft feature consistency,
   minimum-area filtering, and mask NMS suppress duplicates and fragments.
5. **Batch instance output**: all retained masks are returned separately for
   segmentation and counting.

Dense scenes use no fixed automatic background negatives by default, since a
seemingly background patch may be another true target instance.

### Evaluation snapshot

On an integrated benchmark of 2,220 same-image multi-instance images from 232
categories, with five random interior clicks per image:

| Method | Input at inference | F1 | Precision | Recall |
|---|---|---:|---:|---:|
| SAM3 Text Oracle | Correct class name | 0.691 | 0.737 | 0.693 |
| OnePrompt-All | One positive point, no class name | 0.667 | 0.687 | 0.712 |

The SAM3 Text Oracle receives the ground-truth class name; OnePrompt-All sees
only a point, while obtaining comparable F1 and higher recall. See the
[method note](surf/research/ONEPROMPT_ALL_METHOD.md) and
[evaluation scripts](surf/research/) for the protocol.

---

<a id="中文"></a>
## 中文

### 项目简介

OnePrompt-All 是一个面向**单图、单示例、多实例分割**的免训练方法与可交互工作台。用户只需在当前图像中点击一个目标、画一个粗框，或圈出大致区域；系统会在同一张图中理解目标的视觉概念，并输出全部同类实例的独立 mask。推理时不需要类别名称、外部参考图、精确参考 mask 或目标数量。

![SAM Prompt Benchmark 工作台](docs/assets/fig3_workbench.jpg)

工作台统一提供三类模式：**SAM Series**、**Matcher Series** 和
**OnePrompt-All**，可在浏览器中直接比较常规提示分割、跨图参考匹配与同图多实例分割。

### 工作台使用方式

启动 `surf/demos/medical_sam_click_app.py` 后，选择图像并切换到
**OnePrompt-All**。右上角提供三种一次性粗提示：

1. **Click**：在任意一个目标实例内部点击一点；
2. **Box**：画一个粗略矩形框，允许包含背景；
3. **Circle**：拖动圈出大致圆形区域，允许包含背景。

一次用户操作只触发一次完整推理。结果面板会以不同颜色显示每个保留实例的独立 mask，不要求用户连续添加正点或负点。

### 环境与模型配置

推荐环境为 Windows、Python 3.12、CUDA 12.6+ 与 NVIDIA GPU。请以官方 [SAM3 安装说明](https://github.com/facebookresearch/sam3#installation) 为准选择兼容的 PyTorch/CUDA 版本。

```powershell
conda create -n oneprompt-all python=3.12 -y
conda activate oneprompt-all
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-demo.txt
git clone https://github.com/facebookresearch/sam3.git external/sam3
pip install -e external/sam3
```

模型权重不随仓库发布，请放置为：

```text
assets/checkpoints/sam3_modelscope/sam3.pt
assets/checkpoints/dinov2_vitb14_pretrain.pth
```

- `sam3.pt`：从官方 [SAM3 Hugging Face 页面](https://huggingface.co/facebook/sam3) 申请访问权限、`hf auth login` 后下载兼容的图像 checkpoint；也可用 `--sam3-checkpoint` 指定其他位置。
- `dinov2_vitb14_pretrain.pth`：从官方 [DINOv2 预训练模型页面](https://github.com/facebookresearch/dinov2#pretrained-models) 下载 ViT-B/14 backbone 后按此文件名保存。

运行网页：

```powershell
python .\surf\demos\medical_sam_click_app.py --host 127.0.0.1 --port 7860
```

浏览器访问 <http://127.0.0.1:7860>。如果 7860 端口已有服务，请直接复用，不要重复启动。

仓库内含 `surf/datasets/oneprompt_all/natural20/` 的 20 张小型自然图示例；完整数据集、权重和运行结果保持本地且不纳入 Git。

### 方法与结果

![OnePrompt-All 架构](docs/assets/fig1_architecture2.jpg)

对于图像 \(I\) 中的粗提示 \(p\)，方法先由 SAM3 得到 seed mask，再用 seed 内的 DINOv2-B/14 特征构造在线 exemplar；随后从同图 SAM3 实例候选中进行发现与选择。DINO 特征只作为软证据，结合 SAM 置信度、最小面积过滤与 mask NMS，以兼顾外观变化实例的召回和错误类别/碎片/重复候选的抑制。最终一次性返回全部独立实例 mask，可同时服务于分割与计数。

在整合的 2,220 张、232 类同图多实例 benchmark 上，每图随机采样 5 个实例内部点击：

| 方法 | 推理时输入 | F1 | Precision | Recall |
|---|---|---:|---:|---:|
| SAM3 Text Oracle | 正确类别名称 | 0.691 | 0.737 | 0.693 |
| OnePrompt-All | 单个正点，无类别名称 | 0.667 | 0.687 | 0.712 |

Text Oracle 预先获得了 GT 类别名；OnePrompt-All 只有一个点提示，但得到相近的 F1 和更高的 Recall。具体协议见[方法说明](surf/research/ONEPROMPT_ALL_METHOD.md)和[评测脚本](surf/research/)。

## Poster & editable source

- [Download poster PDF](docs/assets/surf_poster.pdf)
- [Editable LaTeX source](poster/main.tex)

The `poster/` directory includes the updated PDF, LaTeX source, and its figure
dependencies. It can be uploaded to a blank Overleaf project and compiled with
pdfLaTeX.
`poster/` 目录包含更新后的 PDF、LaTeX 源文件及依赖图片；上传到空白 Overleaf
项目后使用 pdfLaTeX 即可编辑和编译。

## Repository layout

```text
surf/
├── demos/       # FastAPI workbench entrypoint
├── workbench/   # OnePrompt-All and dataset browser
├── research/    # benchmark construction and evaluation
├── adapters/    # SAM/SAM2/SAM3/Matcher adapters
└── datasets/oneprompt_all/natural20/  # tracked examples
poster/          # editable poster source and PDF
docs/assets/     # README figures and poster PDF
assets/checkpoints/  # local-only weights
external/sam3/   # local SAM3 clone
```

## References

If you use this repository, please cite OnePrompt-All and the foundation
models it builds on. Replace the author/year placeholders in the first entry
with the final bibliographic information before publication.

```bibtex
@misc{onepromptall2026,
  title        = {OnePrompt-All: Single-Image Single-Exemplar Multi-Instance Segmentation},
  author       = {OnePrompt-All Authors},
  year         = {2026},
  howpublished = {\url{https://github.com/DurianLoop/OnePrompt-All}}
}

@article{sam3,
  title   = {SAM 3: Segment Anything with Concepts},
  author  = {Carion, Nicolas and Gustafson, Laura and Hu, Yuan-Ting and others},
  journal = {arXiv preprint},
  year    = {2025}
}

@article{dinov2,
  title   = {DINOv2: Learning Robust Visual Features without Supervision},
  author  = {Oquab, Maxime and Darcet, Timothee and Moutakanni, Theo and others},
  journal = {Transactions on Machine Learning Research},
  year    = {2024}
}

@inproceedings{sam,
  title     = {Segment Anything},
  author    = {Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and others},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year      = {2023}
}

@inproceedings{matcher,
  title     = {Matcher: Segment Anything with One Shot Using All-Purpose Feature Matching},
  author    = {Liu, Yang and Zhu, Muzhi and Li, Hengtao and others},
  booktitle = {International Conference on Learning Representations},
  year      = {2024}
}
```

## Acknowledgements

OnePrompt-All builds on [SAM3](https://github.com/facebookresearch/sam3),
[DINOv2](https://github.com/facebookresearch/dinov2),
[Segment Anything](https://github.com/facebookresearch/segment-anything), and
[Matcher](https://github.com/aim-uofa/Matcher). Please follow their respective
licenses and checkpoint-access terms.
