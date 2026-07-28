# WP3b: Cross-Dataset Multi-Seed Protocol Report

**Date**: 2026-07-28
**Status**: PHASEAB_CROSS_DATASET_ADQUATE

## Summary
- Supportive datasets: 1
- Boundary datasets: 1
- Failure datasets: 0

## Dataset Results

| Dataset | Verdict | N Queries | N Seeds | Mean Dice | CFR | Risk Rho | Seg Quality |
|---|---:|---:|---:|---:|---:|---:|
| MoNuSeg | supportive | 30 | 5 | 0.4948 | 0.0000 | 0.827 | moderate |
| IDRID | boundary | 30 | 5 | 0.1022 | 0.8956 | 0.952 | poor |
| Validation41 | partial_pass | 41 | 1 | 0.4141 | 0.1220 | N/A (single seed) | moderate |

## Three Questions per Dataset

### MoNuSeg
1. Segmentation quality: moderate (Dice=0.4948)
2. Prediction consistency risk ranking: ¦Ñ=0.827 (feature: prediction_jaccard_k1_within)
3. Selective prediction benefit: verdict=supportive

### IDRID
1. Segmentation quality: poor (Dice=0.1022)
2. Prediction consistency risk ranking: ¦Ñ=0.952 (feature: prediction_jaccard_k1_within)
3. Selective prediction benefit: verdict=boundary

## Conclusion
Cross-dataset analysis: 1 supportive, 1 boundary, 0 failure.

At least one external dataset shows supportive evidence for prediction consistency as a risk signal.

At least one dataset shows boundary behavior where risk signal transfers but segmentation quality is limited.