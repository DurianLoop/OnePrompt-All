# 模型权重

**语言：** [English](MODEL_WEIGHTS.md) | [中文](MODEL_WEIGHTS_zh.md)

权重不随本仓库发布，也不应提交到 Git。

工作台和全部 `external/` 模块的代码、权重、数据集完整矩阵见[完整资源安装指南](ALL_RESOURCES_zh.md)。

## 必需目录结构

```text
assets/
└── checkpoints/
    ├── sam3_modelscope/
    │   └── sam3.pt
    └── dinov2_vitb14_pretrain.pth
```

```powershell
New-Item -ItemType Directory -Force .\assets\checkpoints\sam3_modelscope
```

## SAM3

1. 在官方 [SAM3 Hugging Face 仓库](https://huggingface.co/facebook/sam3)申请访问权限。
2. 本地认证：

   ```powershell
   pip install -U huggingface_hub
   hf auth login
   ```

3. 下载与已安装 `facebookresearch/sam3` 源码兼容的 SAM3 **图像 checkpoint**，详情见[官方说明](https://github.com/facebookresearch/sam3#getting-started)。
4. 放置或重命名为 `assets/checkpoints/sam3_modelscope/sam3.pt`。

若使用其他位置：

```powershell
python .\surf\demos\medical_sam_click_app.py --host 127.0.0.1 --port 7860 --sam3-checkpoint D:\models\sam3.pt
```

请勿混用不兼容的 SAM3 / SAM3.1 代码与权重。

## DINOv2-B/14

1. 访问官方 [DINOv2 预训练模型](https://github.com/facebookresearch/dinov2#pretrained-models)。
2. 下载 **ViT-B/14 backbone** checkpoint，而非分类头。
3. 保存为 `assets/checkpoints/dinov2_vitb14_pretrain.pth`。

## 可选工作台权重

| 模式 | 常用本地文件名 | 来源 |
|---|---|---|
| SAM | `assets/checkpoints/sam_vit_b.pth` | [Segment Anything](https://github.com/facebookresearch/segment-anything#model-checkpoints) |
| SAM2 | `assets/checkpoints/sam2.1_hiera_tiny.pt` | [SAM2](https://github.com/facebookresearch/sam2) |
| MedSAM | `assets/checkpoints/medsam_vit_b.pth` | [MedSAM](https://github.com/bowang-lab/MedSAM) |
| Matcher | `assets/checkpoints/swint_only_sam_many2many.pth` | [Matcher](https://github.com/aim-uofa/Matcher) |

## 验证

```powershell
Get-Item .\assets\checkpoints\sam3_modelscope\sam3.pt
Get-Item .\assets\checkpoints\dinov2_vitb14_pretrain.pth
```
