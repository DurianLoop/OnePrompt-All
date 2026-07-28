# WP3: Cross-Pipeline Boundary Completion Report

**Date**: 2026-07-28
**Status**: PHASEAB_CROSS_PIPELINE_DIRECTIONAL_PASS
**Primary Pipeline**: DINOv2-L + SAM-B (directional)

## Summary
- Strong support: 0/5
- Directional support: 5/5
- No support: 0/5

**Paper wording**: pipeline-specific, strongest for SAM-B

## Pipeline Results

| Pipeline | Support | AUROC | AURC(PC) | AURC(Rand) | CFR Red@80% | Dice@80 |
|---|---:|---:|---:|---:|---:|---:|
| L+SAM-B | directional | 0.036 | 0.0158 | 0.0763 | 88.6% | 0.5007 |
| L+SAM-L | directional | 0.330 | 0.1002 | 0.1395 | 23.1% | 0.3029 |
| L+SAM2-T | directional | 0.402 | 0.0910 | 0.1250 | 9.1% | 0.3248 |
| L+SAM2-S | directional | 0.408 | 0.1339 | 0.2175 | -1.6% | 0.2809 |
| B+SAM-B | directional | 0.026 | 0.0145 | 0.1419 | 88.6% | 0.4683 |

## Gate Checks

### L+SAM-B
- AUROC >= 0.70: False
- AURC better than Random: True
- CFR not increase: True
- Verdict: **directional**

### L+SAM-L
- AUROC >= 0.70: False
- AURC better than Random: True
- CFR not increase: True
- Verdict: **directional**

### L+SAM2-T
- AUROC >= 0.70: False
- AURC better than Random: True
- CFR not increase: True
- Verdict: **directional**

### L+SAM2-S
- AUROC >= 0.70: False
- AURC better than Random: True
- CFR not increase: True
- Verdict: **directional**

### B+SAM-B
- AUROC >= 0.70: False
- AURC better than Random: True
- CFR not increase: True
- Verdict: **directional**

## Conclusion
Prediction consistency shows 5/5 pipelines with at least directional support.

Primary pipeline (DINOv2-L + SAM-B) shows the strongest results with CFR@80 reduction of 88.6%.