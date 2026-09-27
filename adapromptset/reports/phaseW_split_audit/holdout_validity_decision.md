# WP0 — Holdout Validity Decision

**Date**: 2026-07-27
**Audit**: Phase W Split Provenance

## Executive Summary

**The 80 MoNuSAC queries are NOT a valid holdout.** They are a Development Cohort.

## Evidence

1. **100% overlap with M2 training**: All 80 queries used in the v5 paper results were used for M2 (gradient boosting) model training in Phase T.
2. **100% overlap with RAMP risk predictor training**: All 80 queries were used for training the gradient-boosted tree risk predictor.
3. **100% overlap with RRN CV**: All 80 queries were used for RRN grouped 5-fold cross-validation.
4. **100% overlap with component ablation**: All 80 queries were used for component ablation analysis.
5. **100% overlap with paper v5**: All results in the v5 paper are based on these 80 queries.
6. **Query naming convention**: All queries are named `train_XXX`, confirming they originate from the MoNuSAC training set.

## M2 Validation Split Discovery

M2 had a proper train/validation split:
- **M2 Training**: 80 queries (these are the 80 queries in the paper)
- **M2 Validation**: 41 queries (these are NOT in the paper results)

The 41 M2 validation queries were held out from M2 training but were never used in the paper. They represent a potential validation cohort.

## Data Role Reassignment

| Original Label | Correct Label | N Queries | Status |
|---------------|---------------|-----------|--------|
| "holdout" (80) | Development Cohort | 80 | Used for training, feature design, ablation, paper decisions |
| M2 Validation (41) | Validation Cohort | 41 | Held out from M2 training, not observed in paper results |
| — | Internal Holdout | 0 | Does not exist |
| — | Formal Test | 0 | Does not exist |

## Gate Decision

**PHASEW_SPLIT_RENAMED_TO_DEVELOPMENT_COHORT**

The 80 queries must be renamed from "holdout" to "Development Cohort" in all paper text.

## Implications

1. All "holdout-confirmed" claims must be removed
2. The 41 M2 validation queries could be used as a validation cohort for future evaluation
3. A true holdout or formal test set would need to be established from scratch
4. The paper's current results are development-set results, not holdout results
5. This does not invalidate the findings, but it does limit the strength of conclusions

## Recommended Actions

1. Rename 80 queries to "Development Cohort" in all paper text
2. Remove all "holdout" and "holdout-confirmed" language
3. Consider evaluating on the 41 M2 validation queries as a validation check
4. If a true holdout is needed, it must be a completely separate dataset split