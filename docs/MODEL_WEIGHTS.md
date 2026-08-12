# Model Weights

**Language:** [English](MODEL_WEIGHTS.md) | [中文](MODEL_WEIGHTS_zh.md)

Weights are not included in this repository and must not be committed.

## Required layout

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

1. Request access at the official [SAM3 Hugging Face repository](https://huggingface.co/facebook/sam3).
2. Authenticate locally:

   ```powershell
   pip install -U huggingface_hub
   hf auth login
   ```

3. Download a SAM3 **image** checkpoint compatible with the installed `facebookresearch/sam3` code; see the [official instructions](https://github.com/facebookresearch/sam3#getting-started).
4. Place or rename the checkpoint to `assets/checkpoints/sam3_modelscope/sam3.pt`.

For a different location, use:

```powershell
python .\surf\demos\medical_sam_click_app.py --host 127.0.0.1 --port 7860 --sam3-checkpoint D:\models\sam3.pt
```

Do not mix incompatible SAM3 and SAM3.1 code/checkpoints.

## DINOv2-B/14

1. Visit [DINOv2 pretrained models](https://github.com/facebookresearch/dinov2#pretrained-models).
2. Download the **ViT-B/14 backbone** checkpoint, not a classifier head.
3. Save it as `assets/checkpoints/dinov2_vitb14_pretrain.pth`.

## Optional workbench weights

| Mode | Typical local filename | Source |
|---|---|---|
| SAM | `assets/checkpoints/sam_vit_b.pth` | [Segment Anything](https://github.com/facebookresearch/segment-anything#model-checkpoints) |
| SAM2 | `assets/checkpoints/sam2.1_hiera_tiny.pt` | [SAM2](https://github.com/facebookresearch/sam2) |
| MedSAM | `assets/checkpoints/medsam_vit_b.pth` | [MedSAM](https://github.com/bowang-lab/MedSAM) |
| Matcher | `assets/checkpoints/swint_only_sam_many2many.pth` | [Matcher](https://github.com/aim-uofa/Matcher) |

## Verify

```powershell
Get-Item .\assets\checkpoints\sam3_modelscope\sam3.pt
Get-Item .\assets\checkpoints\dinov2_vitb14_pretrain.pth
```
