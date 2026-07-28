# Phase AB Scientific Scope

## Core Claim

> Cross-reference prediction consistency provides a simple, GT-free signal for ranking segmentation risk and reducing catastrophic failures through selective prediction in multi-reference visual prompting.

## What We Study

- Prediction consistency as a risk signal
- Coverage-based selective prediction
- Risk-coverage trade-off curves
- Cross-pipeline behavior (5 pipelines)
- Cross-dataset boundaries (MoNuSAC, MoNuSeg, IDRID)

## What We Do NOT Study

- New segmentation models
- Prompt adaptation or optimization
- Learned risk models (RRN → Appendix)
- Universal anti-correlation properties
- Dynamic reference selection
- Training-free claims (matcher is trained)
- Identical conditions comparisons

## Pipeline Scope

Primary: DINOv2-L + SAM-B
Secondary: DINOv2-B + SAM-B, DINOv2-L + SAM-L, DINOv2-L + SAM2-Tiny, DINOv2-L + SAM2-Small

## Dataset Scope

Development: MoNuSAC (80 queries)
Validation: MoNuSAC (41 queries)
External: MoNuSeg (30 queries), IDRID (30 queries)

## Limitations (Honest)

- No true holdout/confirmatory cohort
- MoNuSeg has CFR=0 (no catastrophic failures to detect)
- IDRID has severe segmentation quality issues (mean Dice ~0.04)
- SAM2 pipelines show no benefit from Prediction Consistency
- Only fold-0 for Pascal/COCO
- SAM-Med2D results unverifiable