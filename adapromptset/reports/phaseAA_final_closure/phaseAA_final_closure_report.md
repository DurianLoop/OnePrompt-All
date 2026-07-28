# Phase AA Final Closure Report

**Date**: 2026-07-27
**Status**: PHASEAA_PAPER_NEAR_READY
**Route**: Route B - Simplified Selective Prediction

## Gate Summary

| Gate | Status | Description |
|------|--------|-------------|
| AA0 | PHASEAA_INTEGRITY_AUDIT_FAIL | Phase V/W contradiction audit |
| AA1 | PHASEAA_VALIDATION41_NOT_HOLDOUT | 41 validation query audit |
| AA2 | PHASEAA_PC_DEVELOPMENT_PARTIAL_PASS | Development Cohort 80-query CV |
| AA3 | PHASEAA_SKIPPED_NO_CONFIRMATORY_COHORT | Confirmatory experiment (skipped) |
| AA4 | PHASEAA_CROSS_PIPELINE_PASS | Cross-pipeline generality |
| AA5 | PHASEAA_EXTERNAL_DIRECTIONAL_SUPPORT | Cross-dataset validation |
| AA6 | PHASEAA_BASELINE_PROVENANCE_PARTIAL | Baseline provenance |
| AA7 | PHASEAA_ANTICORRELATION_REMOVED_FROM_MAIN | Anti-correlation disposition |

## Key Findings

- **aa0**: Null model contradiction confirmed — Phase W buggy. Anti-correlation removed from main.
- **aa1**: 41 validation queries = Validation Cohort, NOT holdout. No confirmatory cohort exists.
- **aa2**: B4 Prediction Consistency: 75% CFR@80 reduction on L+SAM-B, partial pass on SAM2.
- **aa3**: Skipped — no valid confirmatory cohort.
- **aa4**: Prediction consistency shows directional support in 5/5 pipelines, strongest for SAM-B.
- **aa5**: 1/3 datasets show directional support
- **aa6**: SAM-Med2D: REMOVE_FROM_MAIN_TABLE. External baselines: RESULTS_FOUND.
- **aa7**: Anti-correlation removed from main text. Mechanical artefact, not structural.

## Publication Readiness

- Status: **PHASEAA_PAPER_NEAR_READY**
- Checks passed: 12/14
- Blockers: confirmatory_cohort_exists, full_draft_exists

## Final Contributions

1. RRE problem definition with reference-resampling evaluation
2. Prediction consistency as a simple, GT-free query-level risk signal
3. Coverage-risk trade-off evaluation across multiple pipelines and datasets

## Removed from Main Text

- Full RAMP
- RRN
- weighting
- fusion
- anti-correlation
- theorem/lemma
- SAM-Med2D (unverifiable)

## Unresolved Blockers

- No true holdout/confirmatory cohort
- SAM-Med2D checkpoint unverifiable
- Pascal-5i/COCO-20i only fold-0
- SAM2 pipelines show no benefit from B4

## Recommended Title

> Prediction Consistency as a Simple Risk Signal for Multi-Reference Visual Prompting

## Next Steps

- Write full venue-neutral draft (paper/phaseAA/)
- Generate final figures (risk-coverage curves, cross-pipeline summary)
- Run on MoNuSeg/IDRID with full multi-seed protocol if possible
- Consider DINOv3 + SAM3 for modern pipeline audit (optional)
