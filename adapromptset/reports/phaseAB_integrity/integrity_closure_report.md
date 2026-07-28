# Phase AB Integrity Closure Report

**Date**: 2026-07-27
**Gate**: WP0
**Status**: PHASEAB_INTEGRITY_CLOSURE_PASS

## 1. Null Model Conflict Resolution

**Finding**: Phase W null model was buggy. The `compute_null_correlation` function returned 0.0 instead of the proper Null 1 (mean-preserving random beta) value.

**Resolution**:
- Phase V Null 1 (r_null = -0.583 for L+SAM-B) is CORRECT
- Phase W null model is BUGGY
- Corrected Δr is POSITIVE for all SAM-B pipelines
- Anti-correlation is a MECHANICAL artefact, not structural

**Status**: RESOLVED. Anti-correlation removed from main text.

## 2. SAM-L Significance

**Finding**: Phase W incorrectly marked L+SAM-L as significant without Holm correction.

**Resolution**:
- L+SAM-L: p=0.049 → NOT Holm-significant
- Phase V label is CORRECT
- Phase W label is INCORRECT

**Status**: RESOLVED. All significance labels unified.

## 3. Pipeline Count

**Frozen**: 5 pipelines (P1-P5)
- P1: DINOv2-L + SAM-B (primary)
- P2: DINOv2-B + SAM-B
- P3: DINOv2-L + SAM-L
- P4: DINOv2-L + SAM2-Tiny
- P5: DINOv2-L + SAM2-Small

**Status**: CONSISTENT across all phases and reports.

## 4. Terminology

**Approved terms**: C_pred(q), R_PC(q), CFR, AURC, Coverage, Selective Prediction

**Retired terms**: training-free, calibration-free, holdout-confirmed, universal, identical conditions, Full RAMP (main), RRN (main)

**Status**: FROZEN. No ambiguity in new draft.

## 5. Retired Claims

15 claims permanently retired (see retired_claims.md). All prohibited claims removed from active claim registry.

**Status**: COMPLETE.

## Gate Decision

### PHASEAB_INTEGRITY_CLOSURE_PASS

All 6 conditions met:
1. No conflicting null results ✓
2. All p-values and Holm conclusions consistent ✓
3. Model and pipeline counts consistent ✓
4. New draft does not depend on anti-correlation ✓
5. All prohibited claims removed ✓
6. Main method numbers traceable to original artifacts ✓