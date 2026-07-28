# Corrected Anti-Correlation Claim

**Date**: 2026-07-27
**Based on**: Phase AA Integrity Audit (AA0)

## Final Disposition

### REMOVE FROM MAIN TEXT

The anti-correlation between reference instability and catastrophic failure rate is NOT a structural property of multi-reference visual prompting. Evidence:

1. **Null model bug**: Phase W computed r_null = 0.0 (bug). The correct Null 1 (random beta, mean-Dice preserving) produces r_null = -0.583, which reproduces 94% of the observed correlation.

2. **Corrected residual**: Δr = r_observed - r_null = +0.071 (L+SAM-B) and +0.034 (B+SAM-B). The residual is POSITIVE, meaning the anti-correlation is weaker than the null predicts.

3. **Pipeline specificity**: The anti-correlation is only significant for SAM-B pipelines (p<0.001). For SAM-L (p=0.049, not Holm-significant), SAM2-Tiny (p=0.325), and SAM2-Small (p=0.113), there is no significant association.

4. **Mechanical artefact**: The observed anti-correlation is entirely explained by the mechanical relationship between mean Dice level and CFR. When mean Dice is controlled, the partial correlation drops to -0.064 (p=0.575, ns) for SAM-L.

## What Can Be Claimed

### SUPPORTED
- Reference instability is real and measurable (mean best-worst gap = 0.187, 88.75% queries with gap ≥ 0.05)
- Prediction consistency is a GT-free risk signal (B4 strongest in 3/5 pipelines)

### PROHIBITED
- Anti-correlation between instability and CFR is a structural property
- Anti-correlation is universal or model-agnostic
- Anti-correlation has a "structural residual" beyond mechanical effects

### ALLOWED (with qualification)
- "We observe an empirical association between reference instability and failure severity in SAM-B pipelines, but this association is pipeline-specific and does not generalize to SAM2 variants."

## Paper Impact

The paper can be complete without anti-correlation claims. The core contributions are:
1. RRE problem definition
2. Prediction Consistency as a simple risk signal
3. Coverage-based selective prediction

Anti-correlation was never the core method; it was a supporting observation. Removing it simplifies the paper and removes a vulnerability.