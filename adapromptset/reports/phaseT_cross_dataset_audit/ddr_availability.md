# DDR (IDRID) Dataset Availability Audit

Date: 2026-07-23
Branch: `phaseT-publication-completion`

## Status: AVAILABLE (IDRID as DDR proxy)

The protocol references "DDR" as the tertiary cross-domain retinal dataset.
On disk, the available retinal dataset is IDRID (Indian Diabetic Retinopathy
Image Dataset), which serves the same role. The protocol's dataset sizes
(383 train / 149 valid / 225 test) do not match the on-disk IDRID data
(55 train / 27 test); the Phase T WP2 uses the available IDRID training set.

## Data Location

`D:\SAM\code\datasets\IDRID\DR_Training_Set\`

## Structure

```
DR_Training_Set/
├── Fundus Images/     (55 JPG files, ~4288x2848, RGB)
└── Combined Masks/    (54 PNG files, ~4288x2848, RGB color-coded)
```

Note: 55 images but only 54 masks — one image (IDRiD_55.jpg) has no mask
and is excluded.

## Format Details

- **Images**: JPG, ~4288x2848 pixels, RGB mode. Resize to 518x518 with
  BICUBIC for DINOv2-L compatibility.
- **Masks**: PNG, RGB color-coded with 5 colors:
  - `[0, 0, 0]` = background (black)
  - `[0, 0, 255]` = SE (Soft Exudates, blue)
  - `[0, 255, 0]` = MA (Microaneurysms, green)
  - `[255, 0, 0]` = HE (Hemorrhages, red)
  - `[255, 0, 255]` = EX (Exudates, magenta)

## Frozen Lesion Intent

**EX (Exudates)** — selected as the frozen lesion intent for Phase T WP2.

Rationale:
- EX is present in all 54 masks (100% coverage).
- EX (exudates) is the most clinically prominent DR lesion type.
- The intent is frozen before WP2 inference per directive §6.3.

Extraction rule: `binary_mask = (R > 200) & (B > 200) & (G < 50)` (magenta).

## Split Plan

54 total cases (with non-empty EX masks). Split:
- Reference pool: ~24 cases (indices 0-23 sorted by case_id)
- Validation queries: ~30 cases (indices 24-53 sorted by case_id)

This split is frozen before WP2 inference and recorded in
`dataset_role_decision.md`.

## Phase T Exposure Status

- Reference pool: EXPOSED (reference pool only)
- Validation queries: UNEXPOSED until WP2 run
- Testing set (27 cases): LOCKED_UNUSED

## Historical Exposure

IDRID train/valid were used for Route B matcher smoke and failure-taxonomy
analysis in Phase 4+. Phase T WP2 uses a new reference-resampling protocol,
so the validation queries remain eligible for directional-support statistics
under the unified Phase T protocol.
