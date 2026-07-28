# Terminology Registry

**Date**: 2026-07-27
**Phase**: AB Integrity Closure

## Approved Terminology

| Term | Symbol | Definition |
|------|--------|------------|
| Prediction Consistency | C_pred(q) | Mean pairwise IoU of masks from different references |
| Prediction Consistency Risk | R_PC(q) | 1 - C_pred(q), higher = riskier |
| Catastrophic Failure | - | Dice < 0.1 |
| Catastrophic Failure Rate | CFR | Fraction of queries with Dice < 0.1 |
| Coverage | C | Fraction of queries accepted (lowest risk) |
| Selective Prediction | - | Accept low-risk queries, abstain on high-risk |
| Risk-Coverage Curve | - | CFR vs coverage curve |
| AURC | - | Area Under Risk-Coverage curve |
| Development Cohort | - | 80 MoNuSAC queries used for exploration |
| Validation Cohort | - | 41 MoNuSAC queries used for model selection |
| Reference Instability | - | Variance in outcomes across different references |

## Retired Terminology

| Old Term | Reason for Retirement |
|----------|----------------------|
| training-free | Matcher (DINOv2) is trained |
| calibration-free | Not applicable (no probability output) |
| holdout-confirmed | No true holdout exists |
| identical conditions | External baselines use different backbones |
| universal property | Not supported by evidence |
| structural theorem | Qualitative only, no quantitative validation |
| Full RAMP | Removed from main text |
| RRN | Appendix only (AUROC=0.573) |
| anti-correlation | Mechanical artefact, not structural |