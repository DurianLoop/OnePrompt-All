# Phase AA Baseline Protocol

**Date**: 2026-07-27

## Learning-Free Baselines

| ID | Name | Description |
|----|------|-------------|
| B0 | Random | Random ranking |
| B1 | Matching Entropy | Matching-map entropy |
| B2 | Matching Disagreement | Matching-map disagreement |
| B3 | Candidate Jaccard | Candidate mask Jaccard similarity |
| B4 | Prompt Overlap | Prompt-coordinate overlap |
| B5 | Prediction Consistency | Cross-reference prediction consistency (OUR METHOD) |
| B6 | Prediction Area Variance | Variance of prediction mask areas |
| B7 | Mean SAM Confidence | Mean SAM confidence/uncertainty |

## Learned Baselines

| ID | Name | Description |
|----|------|-------------|
| B8 | M2-Instability | Gradient-boosted instability predictor |
| B9 | Dual-head M2 | Dual-head M2 model |

## Diagnostic Upper Baseline

| ID | Name | Description |
|----|------|-------------|
| B10 | M2-Catastrophic | Supervised catastrophic label predictor (NOT comparable to B5) |

## External Segmentation Baselines (Ancillary)

### Table A: Shared Backbone (SAM-B)
- Top-K + SAM-B
- CycleSAM + SAM-B
- SAM-Med2D + SAM-B (only if reproducible)

### Table B: Official Default
- GF-SAM + SAM-H
- MAUP + SAM-H
- H-SAM + SAM-L

NOTE: Tables A and B must be clearly separated with backbone differences explicitly annotated.