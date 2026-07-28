# Table 12 — Weighted Aggregation (Track B)

**Status:** Track B NO_GAIN. Compare M4_gbm weighted aggregation vs uniform (B2_mean) baseline.

## 1. Headline Comparison (P30-N0, k=6)

| Metric | Uniform (B2_mean) | Weighted (M4_gbm) | Delta | Direction | Threshold | Status |
|---|---|---|---|---|---|---|
| Mean Dice | 0.4756 | 0.4732 | -0.0024 | worse | +0.005 | **FAIL (cond1)** |
| CVaR_0.1 | 0.0738 | 0.0768 | +0.003 | worse | ≥0 | FAIL (cond4) |
| CFR (Dice<0.1) | 0.0732 | 0.0732 | 0.0000 | tie | ≥0 | FAIL (cond6) |
| Mean ref std | 0.0893 | 0.0868 | -0.0025 (-2.83%) | better | -10% | **FAIL (cond2)** |
| Mean pairwise dev | 0.1139 | 0.0455 | -0.0684 (-60.09%) | better | -10% | **PASS (cond3)** |

## 2. Bootstrap Confidence Interval (Mean Dice Delta)

| Statistic | Value |
|---|---|
| Mean delta (weighted - uniform) | -0.0024 |
| Bootstrap 95% CI lower | -0.0212 |
| Bootstrap 95% CI upper | +0.0133 |
| # bootstrap samples | 1000 |

**CI includes 0** → not statistically significant. Cannot claim Dice improvement (or harm) at α=0.05.

## 3. Per-Config Direction Consistency (cond5)

| Config | Uniform Dice | Weighted Dice | Delta | Direction |
|---|---|---|---|---|
| P20-N1 | 0.4521 | 0.4498 | -0.0023 | worse |
| P30-N0 | 0.4756 | 0.4732 | -0.0024 | worse |
| P50-N0 | 0.4891 | 0.4896 | +0.0005 | better (marginal) |

**Direction consistent: 2/3 configs** (PASSED requirement, but does not reverse the NO_GAIN decision).

## 4. Weight Distribution

| Statistic | Value |
|---|---|
| Max weight observed | 0.4047 |
| Min weight | 0.0312 |
| Mean weight | 0.1667 (= 1/6, sanity check) |
| Std weight | 0.0812 |
| Effective k | 5.21 (no collapse) |

Weighting is **non-degenerate** (max weight 0.40, effective k 5.21) — the model is genuinely differentiating references.

## 5. Why NO_GAIN Despite Predictability

The puzzle: utility is predictable (Spearman 0.478) AND weights are non-degenerate, but Dice does not improve.

**Decomposition:**
- Predictability is **moderate** (0.478) — not perfect.
- Mean Dice differences across references are small (mean pairwise dev 0.11) — most references are similar quality.
- The best references are not **dramatically** better than the average; weighting them more just shifts the mean slightly.
- The weighting makes predictions more similar (pairwise dev ↓60%) — confirming the utility signal is real — but the centroid of those similar predictions is not a better segmentation.

**This is the Track B Utility-Only claim:** utility is a real, predictable signal, but it does not translate into a deployable Dice gain. Scientific finding, not a method contribution.

## 6. Gate B Decision

| Required for PASS | cond1 (Dice+0.005) | cond2 (std -10%) |
|---|---|---|
| Status | FAIL (-0.0024) | FAIL (-2.83%) |

**Decision: NO_GAIN** (both required conditions failed).

Only cond3 (pairwise dev -10%) PASSED — and pairwise dev reduction is a "compression" effect, not a deployable accuracy gain.

## 7. Compute

| Resource | Used |
|---|---|
| GPU-hours | 0.00 |
| SAM calls (Phase 2 only) | 123 |
| Best model | M4_gbm |

## 8. Reference

- `reports/phaseS_trackB/gateB_summary.json`
- `reports/phaseS_trackB/val_weighted_consensus_metrics.csv`
- `reports/phaseS_trackB/val_comparison.csv`
- `reports/phaseS_trackB/val_weights.csv`
