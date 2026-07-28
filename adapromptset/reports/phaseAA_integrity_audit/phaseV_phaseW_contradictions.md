# Phase V/W Contradiction Audit

**Date**: 2026-07-27
**Gate**: AA0

## Contradiction 1: Null Model

### Phase V Claim
"Null 1 (random beta) reproduces 94% of observed correlation (r=-0.583 vs observed -0.619)."

### Phase W Claim
"r_null_mean = 0.0, null_explains_pct = 0%"

### Root Cause
**Phase W bug**: The `compute_null_correlation` function in `run_phaseW_wp7_anticorrelation.py` was called but returned 0.0 for all pipelines. The Phase V null model (Null 1: random beta with same mean Dice) was a proper computation showing r_null = -0.583 for SAM-B.

### Resolution
- Phase V null model is CORRECT
- Phase W null model is BUGGY
- The corrected Δr for SAM-B is POSITIVE (+0.071 for L+SAM-B, +0.034 for B+SAM-B), meaning the anti-correlation is entirely explained by mechanical factors
- The "structural residual" claim in Phase W is UNSUPPORTED

---

## Contradiction 2: SAM-L Significance

### Phase V Claim
"L+SAM-L r=-0.221 (p=0.049, not Holm-significant), partial r=-0.064 (p=0.575, ns)"

### Phase W Claim
"L+SAM-L significant_residual=true, p_permutation=0.0382"

### Root Cause
Phase W did not apply Holm correction. With 4 tests (5 pipelines, but 1 is primary), the Holm threshold for L+SAM-L (rank 4) would be 0.05/1 = 0.05, but with 5 tests it's 0.05/2 = 0.025 for rank 2. In either case, p=0.049 is marginal and fails Holm correction.

### Resolution
- Phase V label is CORRECT: L+SAM-L is NOT significant after Holm correction
- Phase W label is INCORRECT: "significant_residual=true" overstates the evidence
- The correct label is "NO" for Holm significance

---

## Contradiction 3: Anti-Correlation Residual

### Phase W Claim
"Δr = r_observed - 0.0 = r_observed" (because r_null_mean = 0.0)

### Corrected Computation
"Δr = r_observed - r_null(Null 1)"

For SAM-B pipelines:
- L+SAM-B: Δr = -0.512 - (-0.583) = **+0.071** (positive!)
- B+SAM-B: Δr = -0.549 - (-0.583) = **+0.034** (positive!)

### Resolution
- Phase W Δr is INCORRECT (based on buggy null)
- The corrected Δr is POSITIVE, meaning the anti-correlation is WEAKER than the null model predicts
- This completely invalidates the "structural residual" narrative
- The anti-correlation is a MECHANICAL artefact, not a structural property

---

## Gate AA0 Decision

### PHASEAA_INTEGRITY_AUDIT_FAIL

**Reasons**:
1. Contradiction 1 (null model): Phase W null model is buggy — returns 0.0 instead of proper Null 1 value
2. Contradiction 2 (SAM-L significance): Phase W incorrectly marks L+SAM-L as significant without Holm correction
3. Contradiction 3 (anti-correlation residual): Phase W Δr = r_observed (bug), corrected Δr is POSITIVE

**Consequence per todo.md §4.3**:
> "若无法解释: PHASEAA_INTEGRITY_AUDIT_FAIL → anti-correlation 从主文删除，只保留 reference instability 与 selective prediction"

### Recommended Action
1. Remove anti-correlation from main text entirely
2. Retain only: reference instability + prediction consistency + selective prediction
3. The paper can stand without anti-correlation claims