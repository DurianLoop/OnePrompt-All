# WP4: Baseline Provenance Closure Report

**Date**: 2026-07-28
**Status**: PHASEAB_BASELINE_PROVENANCE_PASS

## SAM-Med2D
- Status: **REMOVED from main table**
- Reason: Unverifiable checkpoint ¡ª no SHA256, no load log, no inference log

## Baseline Method Provenance

| Method | Code Commit | Config | Reproducible | Query-Level |
|---|---|---|---|---|
| B0_Random | phaseAB-fresh-confirmation-paper-assembly | configs/frozen/phaseAB_prediction_consistency.yaml | True | True |
| B1_Entropy | phaseAB-fresh-confirmation-paper-assembly | configs/frozen/phaseAB_prediction_consistency.yaml | True | True |
| B2_CandidateJaccard | phaseAB-fresh-confirmation-paper-assembly | configs/frozen/phaseAB_prediction_consistency.yaml | True | True |
| B3_PromptOverlap | phaseAB-fresh-confirmation-paper-assembly | configs/frozen/phaseAB_prediction_consistency.yaml | True | True |
| B4_PredictionConsistency | phaseAB-fresh-confirmation-paper-assembly | configs/frozen/phaseAB_prediction_consistency.yaml | True | True |
| B5_M2Instability | phaseAB-fresh-confirmation-paper-assembly | configs/frozen/phaseAB_prediction_consistency.yaml | True | True |

## External Baselines
- Status: Present in separate tables
- Strategy: Two tables ¡ª shared backbone vs official default

### Shared Backbone Table
- Top-K + SAM-B
- CycleSAM + SAM-B

### Official Default Table
- GF-SAM + SAM-H
- MAUP + SAM-H
- H-SAM + SAM-L

### Removed
- SAM-Med2D

## Gate Checks
- sam_med2d_removed: PASS
- external_baselines_present: PASS
- shared_backbone_methods: PASS
- proper_stratification: PASS

## Conclusion
Baseline provenance: PHASEAB_BASELINE_PROVENANCE_PASS. All internal baselines fully reproducible. SAM-Med2D permanently removed. External methods properly stratified by backbone.