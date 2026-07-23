# MoNuSeg Dataset Availability Audit

Date: 2026-07-23
Branch: `phaseT-publication-completion`

## Status: AVAILABLE (KMoNuSeg variant)

## Data Location

`D:\SAM\code\datasets\MonuSeg\kmms\`

## Structure

```
kmms/
├── images/   (82 PNG files, 256x256, RGBA)
└── masks/    (82 PNG files, 256x256, L mode grayscale 0-255)
```

File naming convention: `01_1.png`, `01_2.png`, ..., `28_3.png` (case_patch format).

## Format Details

- **Images**: 256x256 pixels, RGBA mode. Must convert to RGB and resize to
  518x518 with BICUBIC for DINOv2-L compatibility.
- **Masks**: 256x256 pixels, L mode (grayscale, 0-255). Binarize with
  threshold >127 (foreground = white/near-white). Resize to 518x518 with
  NEAREST interpolation.

## Split Plan

82 total cases. Patient IDs are unreliable; case_id is used as patient_id
proxy (directive §6.3, dataset registry). Split:
- Reference pool: ~52 cases (indices 0-51 sorted alphabetically)
- Validation queries: ~30 cases (indices 52-81 sorted alphabetically)

This split is frozen before WP2 inference and recorded in
`dataset_role_decision.md`.

## Phase T Exposure Status

- Reference pool: EXPOSED (reference pool only)
- Validation queries: UNEXPOSED until WP2 run
- Test (official MoNuSeg test 14 cases): LOCKED_UNUSED

## Historical Exposure

KMoNuSeg was not used in prior phases with the reference-resampling protocol.
The official MoNuSeg train (37 cases) was used in Phase 2D as a secondary
dataset, but the KMoNuSeg variant (82 cases) is a superset with different
case IDs.
