# Phase AA Prediction Consistency Definition

**Date**: 2026-07-27
**Status**: TO BE FROZEN (after AA1)

## Input

For a query q with m references:
- R_q = {R_1, ..., R_m}
- Each reference produces a prediction mask: M̂_{q,r}

## Pairwise Prediction Consistency

J_{ij} = IoU(M̂_{q,i}, M̂_{q,j})

C_pred(q) = (2 / (m(m-1))) * Σ_{i<j} J_{ij}

Risk_PC(q) = 1 - C_pred(q)

Higher risk = more disagreement between references.

## Frozen Details

To be finalized:
- Empty mask vs empty mask IoU: 0 (by convention)
- Empty mask vs non-empty mask IoU: 0
- Computation scale: original image scale
- Binary threshold: 0.5
- Connected-component postprocess: none
- Reference count: m = 8 (default)
- Reference sampling: random (fixed seed)
- Mask fusion: not part of score
- Tie handling: random
- NaN handling: treat as 0

## Final Consensus Prediction

Mean-consensus prediction (element-wise mean of all m masks, thresholded at 0.5).