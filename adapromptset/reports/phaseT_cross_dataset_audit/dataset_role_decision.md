# Cross-Dataset Role Decision

Date: 2026-07-23
Branch: `phaseT-publication-completion`
Directive: §6 (WP2), §6.3 (unified protocol)

## Dataset Selection

| Dataset | Role | Available | Cases | Query Split | Reference Pool |
|---|---|---|---|---|---|
| GlaS | development replication | NO | 0 | N/A | N/A |
| MoNuSeg (KMoNuSeg) | secondary diagnostic | YES | 82 | 30 validation | 52 reference |
| IDRID | development replication | YES | 54 | 30 validation | 24 reference |

GlaS data is not available (empty directory). MoNuSeg and IDRID serve as the
two additional datasets for the minimum benchmark submission requirement
(§6.4: "MoNuSAC holdout confirmed + at least 2 additional datasets
directional support").

## MoNuSeg (KMoNuSeg) Configuration

- **Source**: `D:\SAM\code\datasets\MonuSeg\kmms\`
- **Image format**: PNG 256x256 RGBA → RGB, resize 518x518 BICUBIC
- **Mask format**: PNG 256x256 L mode → binary (>127), resize 518x518 NEAREST
- **Split method**: Alphabetical sort by filename, first 52 = reference pool,
  last 30 = validation queries
- **Patient ID**: case_id used as patient_id proxy (unreliable patient IDs)
- **Reference-query constraint**: `reference_case_id != query_case_id`
- **Prompt config**: Universal (P20-N1, P30-N0, P50-N0) — same as MoNuSAC
- **Dataset-specific setting**: None (universal settings only)

## IDRID Configuration

- **Source**: `D:\SAM\code\datasets\IDRID\DR_Training_Set\`
- **Image format**: JPG ~4288x2848 RGB → resize 518x518 BICUBIC
- **Mask format**: PNG RGB color-coded → extract EX (magenta: R>200 & B>200 & G<50),
  resize 518x518 NEAREST
- **Frozen lesion intent**: EX (Exudates) — present in all 54 masks
- **Split method**: Alphabetical sort by case_id, first 24 = reference pool,
  last 30 = validation queries
- **Patient ID**: case_id used as patient_id proxy
- **Reference-query constraint**: `reference_case_id != query_case_id`
- **Prompt config**: Universal (P20-N1, P30-N0, P50-N0) — same as MoNuSAC
- **Dataset-specific setting**: Lesion intent = EX (frozen before WP2 inference)

## Universal Settings (shared across all datasets)

| Component | Value |
|---|---|
| Matcher | DINOv2-L (frozen) |
| Segmenter | SAM-B (frozen) |
| SAM calls | 1 per consensus aggregation |
| Reference counts | k ∈ {1, 3, 6} |
| Aggregation | single / mean |
| Metrics | V_ref / MAD / J_pred / CVaR / CFR / Dice |
| Statistical unit | query / case |
| Bootstrap | 10,000 iterations, 95% CI |
| Seeds | 5 resampling seeds |
| References per query | 6 (patient-disjoint) |
| Prompt configs | P20-N1, P30-N0, P50-N0 |

## Prohibited Operations

1. Accessing testA / testB / MoNuSeg official test / IDRID test.
2. Sharing references across datasets.
3. Claiming "universal across datasets" (prohibited claim #2).
4. Pooling queries across datasets for aggregate significance tests.
