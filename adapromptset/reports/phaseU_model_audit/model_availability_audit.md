# Phase U WP0 — Model Availability & Compatibility Audit

**Date**: 2026-07-24
**Status**: COMPLETE

## 1. Segmenter Models

| Model | Family | Checkpoint | Architecture | Status | VRAM Est. | Notes |
|-------|--------|-----------|-------------|--------|-----------|-------|
| SAM-B | SAM1 | sam_vit_b.pth (358 MB) | vit_b (94M) | AVAILABLE | ~2 GB | Phase T baseline |
| SAM-L | SAM1 | sam_vit_l_0b3195.pth (1192 MB) | vit_l (312M) | AVAILABLE | ~4 GB | Scale comparison |
| SAM-H | SAM1 | sam_vit_h_4b8939.pth (2446 MB) | vit_h (641M) | AVAILABLE | ~8 GB | Largest SAM1 |
| MedSAM-B | SAM1 | medsam_vit_b.pth (358 MB) | vit_b (94M) | AVAILABLE | ~2 GB | Medical fine-tuned |
| SAM-Med2D-B | SAM1 | sam_med2d_b.pth (2443 MB) | vit_b (94M) | BLOCKED | ~2 GB | Checkpoint contains optimizer state; needs weights_only=False |
| SAM2.1-Tiny | SAM2 | sam2.1_hiera_tiny.pt (149 MB) | Hiera-T (39M) | AVAILABLE | ~3 GB | Requires Hydra compose + custom YAML configs |
| SAM2.1-Small | SAM2 | sam2.1_hiera_small.pt (176 MB) | Hiera-S (46M) | AVAILABLE | ~3 GB | Same config structure as Tiny |
| SAM2.1-Base+ | SAM2 | sam2.1_hiera_base_plus.pt (309 MB) | Hiera-B+ (81M) | AVAILABLE | ~4 GB | |
| SAM2.1-Large | SAM2 | sam2.1_hiera_large.pt (857 MB) | Hiera-L (224M) | AVAILABLE | ~8 GB | |
| SAM3 | SAM3 | sam3_modelscope/sam3.pt | - | NOT AVAILABLE | - | torch.jit.script requires .py source files; only .pyc distributed |

## 2. Matcher Models

| Model | Family | Checkpoint | Status | Notes |
|-------|--------|-----------|--------|-------|
| DINOv2-L | DINOv2 | dinov2_vitl14_pretrain.pth (1161 MB) | AVAILABLE | Phase T baseline matcher |
| DINOv2-B | DINOv2 | dinov2_vitb14_pretrain.pth (330 MB) | AVAILABLE | Lighter matcher for WP2 |
| DINOv3 | DINOv3 | - | NOT AVAILABLE | Not downloaded; requires user approval per §5.3 |

## 3. Prompt Interface Support

| Model | Point | Box | Mask | Exemplar | Text | Notes |
|-------|-------|-----|------|----------|------|-------|
| SAM1 (all) | YES | YES | YES | NO | NO | segment_anything SamPredictor |
| SAM2 (all) | YES | YES | YES | NO | NO | SAM2ImagePredictor |
| SAM3 | - | - | - | - | - | NOT AVAILABLE |

## 4. Minimum Viable Matrix (Recommended)

Based on availability audit, the recommended minimum matrix is:

**Segmenter axis (WP1)**:
- S1 = SAM-B (baseline)
- S2 = SAM-L (scale comparison)
- S3 = SAM2.1-Tiny (new family)
- Optional: SAM-H, SAM2.1-Small, MedSAM-B

**Matcher axis (WP2)**:
- M1 = DINOv2-L (baseline)
- M2 = DINOv2-B (lighter comparison)

**Blocked**:
- SAM3: Not available (torch.jit.script source requirement)
- DINOv3: Not downloaded
- SAM-Med2D-B: Checkpoint format issue

## 5. Environment Requirements

- SAM1 models: Python 3.11+ (sam_gpu env), segment_anything, torch
- SAM2 models: Python 3.11 (sam_gpu env), hydra, omegaconf, torch
- GPU: RTX 5090 D v2 (32 GB VRAM)
- All models tested successfully in sam_gpu environment

## 6. Scientific Role Assignment

| Model | Scientific Role |
|-------|----------------|
| SAM-B | Baseline segmenter (Phase T anchor) |
| SAM-L | Scale comparison (larger SAM1) |
| SAM2.1-Tiny | New family comparison (SAM2 vs SAM1) |
| SAM2.1-Small | SAM2 scale comparison |
| SAM-H | Maximum SAM1 scale |
| MedSAM-B | Domain-specific (medical) comparison |
| DINOv2-L | Baseline matcher |
| DINOv2-B | Matcher scale comparison |