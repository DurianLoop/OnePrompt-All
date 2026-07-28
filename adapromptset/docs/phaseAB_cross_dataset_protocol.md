# Phase AB Cross-Dataset Protocol

## Datasets

1. MoNuSeg (30 queries) — external nuclei segmentation
2. IDRID (30 queries) — external lesion segmentation

## Protocol

For each dataset:
- Fixed reference count (K=6)
- Reference seeds >= 5
- Query-level aggregation
- Full risk-coverage curves
- Separate reporting of:
  1. Segmentation quality (mean Dice, CFR)
  2. Risk ranking quality (AUROC, AURC)
  3. Selective prediction benefit (CFR reduction, Dice preservation)

## Classification

### Supportive
- AUROC >= 0.70
- AURC better than Random
- CFR@80 relative reduction >= 20%

### Boundary
- Risk ranking transfers but segmentation quality very low
- RISK_SIGNAL_TRANSFERS_SEGMENTATION_FAILS

### Failure
- No risk transfer
- NO_RISK_TRANSFER

## Minimum Requirements for Paper

- At least one external supportive dataset
- At least one clearly identified boundary dataset

## Current Status

MoNuSeg: CFR=0.0 (no catastrophic failures to detect) — cannot evaluate risk signal
IDRID: AUROC=0.024, AURC better than random, CFR@80 reduction=14.5% — boundary case
Validation41: AUROC=0.222, CFR@80 reduction=23.1% — partial support