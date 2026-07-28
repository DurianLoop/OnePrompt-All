# Phase S Track B: Learned Reference Reliability — Gate B Summary

Date: 2026-07-22T17:03:12
Validation queries: 41
Best config: P30-N0
Model: M4_gbm (T=0.25, eps=0.0, lam=0.0)

## Weighted vs Uniform (B2_mean)

| Metric | Weighted | Uniform |
|---|---:|---:|
| Mean Dice | 0.4732 | 0.4756 |
| CVaR_0.1 | 0.0768 | 0.0738 |
| CFR (Dice<0.1) | 0.0732 | 0.0732 |
| Mean ref std | 0.0868 | 0.0893 |
| Mean pairwise dev | 0.0455 | 0.1139 |

Delta Dice: -0.0024
Std reduction: 0.0283
Pairwise deviation reduction: 0.6009
Utility prediction Spearman: 0.4781
Bootstrap mean delta: -0.0024 (CI [-0.0212, 0.0133])
Max weight observed: 0.4047
Consistent config count (of 3): 2

## Gate B Conditions (§7.6)

| # | Condition | Pass |
|---|---|---|
| 1 | Mean Dice +0.005 | NO |
| 2 | Within-query ref std -10% | NO |
| 3 | Pairwise ref dev -10% | YES |
| 4 | CVaR_0.1 +0.01 | NO |
| 5 | CFR relative -15% | NO |
| 6 | Prediction consistency IoU +0.02 | NO |

## Requirements

| Requirement | Pass |
|---|---|
| Utility prediction Spearman >= 0.30 | YES |
| Direction consistent (>= 2/3 configs) | YES |
| No query GT used | YES |
| Single SAM call per query | YES |
| No weight collapse (max<0.95, min>=eps) | YES |
| Bootstrap point estimate > 0 | NO |

Conditions passed: 1/6
All requirements pass: False
Gate pass: False
## Decision: **PHASES_TRACK_B_NO_GAIN**

