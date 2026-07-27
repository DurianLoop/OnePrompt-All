# Phase AA Statistics Protocol

**Date**: 2026-07-27

## Query-Level Statistics

All statistics computed at query level, not episode level.

## Confidence Intervals

- Bootstrap CI (10,000 iterations) for all metrics
- 95% CI reported as [lower, upper]

## Multiple Comparison Correction

- Holm correction applied to all p-values across baselines
- Raw p and Holm-adjusted p both reported

## Significance Testing

- Paired bootstrap test for AURC comparison
- Permutation test for correlation residuals
- Holm-adjusted significance labels

## Metrics

- AUROC: Area Under ROC Curve (catastrophic failure prediction)
- AUPRC: Area Under Precision-Recall Curve
- AURC: Area Under Risk-Coverage curve
- Excess AURC: AURC - AURC_oracle
- CFR@c: Catastrophic Failure Rate at coverage c
- Dice@c: Mean Dice at coverage c
- Selective Dice: Dice on accepted queries only

## Calibration

- ECE (Expected Calibration Error) only for learned models
- Not applicable to learning-free baselines