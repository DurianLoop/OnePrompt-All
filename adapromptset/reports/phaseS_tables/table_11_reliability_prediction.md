# Table 11 — Reliability Prediction (Track B)

**Status:** Track B — `PHASES_TRACK_B_NO_GAIN` — 1/6 Gate B conditions PASSED, 5/6 requirements passed.

## 1. Model Ladder (M0–M5)

| Model | Description | Inner CV Spearman (train) | Notes |
|---|---|---|---|
| M0 | Constant baseline | 0.000 | No prediction |
| M1 | Single feature (prediction_jaccard_mean) | 0.7412 | Strongest single feature |
| M2 | Top-3 features (linear) | 0.8233 | Linear regression |
| M3 | Random forest (top features) | 0.8718 | Tree ensemble |
| M4_gbm | **Gradient Boosting** | **0.9163** | **Best model** |
| M5 | Stacked ensemble (RF + GBM) | 0.9088 | Marginal overfitting observed |

**Best model: M4_gbm (Gradient Boosting)** — selected by inner CV.

## 2. Train vs Validation Spearman

| Split | Spearman ρ | Config |
|---|---|---|
| Train (inner CV) | **0.9163** | P30-N0 |
| Validation | **0.4781** | P30-N0 |

**Generalisation gap:** 0.9163 → 0.4781 (drop of 0.438). The model is **predictive on train** (very strong) and **moderately predictive on val** (0.478, above the 0.4 threshold).

## 3. Per-Feature Correlations (Validation Set, P30-N0)

| Feature | Description | Spearman with reference utility (val) |
|---|---|---|
| `prediction_jaccard_mean` | Mean pairwise pixel-Jaccard of single-ref predictions | **0.4781** (drives the M4 model) |
| `mean_ref_std` | Mean within-query ref std (Dice) | -0.4233 |
| `mean_pairwise_dev` | Mean pairwise Dice deviation | -0.4102 |
| `best_worst_gap` | Best-worst reference Dice gap | -0.3812 |
| `prediction_iou_min` | Min pairwise pixel-Jaccard | -0.3477 |
| `ref_count` | Number of references (constant k=6) | 0.0000 (constant) |

All test-time-safe features (computed from SAM outputs only, no ground truth).

## 4. Track B Configuration

| Hyperparameter | Value | Meaning |
|---|---|---|
| T (temperature) | 0.25 | Softmax temperature on predicted utility → weights |
| ε (weight floor) | 0.0 | Minimum weight per reference (no floor) |
| λ (risk penalty) | 0.0 | CVaR risk penalty in weighting (disabled) |
| Best config | P30-N0 | Primary operating point |
| Best model | M4_gbm | Gradient Boosting |

## 5. Weight Distribution Diagnostics

| Diagnostic | Value | Comment |
|---|---|---|
| Max weight observed | 0.4047 | No collapse (well below 1.0) |
| Weight std (mean) | 0.0812 | Substantial variation |
| Effective k | 5.21 | Close to k=6 (no degenerate weighting) |

## 6. Gate B Conditions (1/6 PASSED)

| # | Condition | Threshold | Result | Status |
|---|---|---|---|---|
| B1 | Mean Dice weighted > uniform + 0.005 | +0.005 | -0.0024 | **FAIL** |
| B2 | Mean ref std reduction ≥10% | 10% | 2.83% | **FAIL** |
| B3 | Mean pairwise dev reduction ≥10% | 10% | 60.09% | **PASS** |
| B4 | CVaR improvement ≥0 | ≥0 | +0.003 (worse) | FAIL |
| B5 | Direction consistent across ≥2/3 configs | 2/3 | 2/3 | PASS (requirement) |
| B6 | CFR reduction ≥0 | ≥0 | 0.0000 (no change) | FAIL |

**Decision rule (preregistered):** cond1 AND cond2 must both PASS for Track B PASS.
- cond1 FAIL, cond2 FAIL → **NO_GAIN**.

## 7. Key Finding (Track B Utility-Only Claim)

> "Reference utility is predictable from test-time-safe features, but weighted consensus does not reduce variance; the utility signal is a scientific finding without a deployable stability gain."

- Predictability: train Spearman 0.9163, val Spearman 0.4781 — **predictable**.
- Weighted Dice: 0.4732 vs uniform 0.4756 (delta = -0.0024, bootstrap CI [-0.0212, 0.0133]).
- Weighted ref std: 0.0868 vs uniform 0.0893 (reduction 2.83%, FAILED ≥10%).
- Weighted pairwise dev: 0.0455 vs uniform 0.1139 (reduction 60.09%, PASSED ≥10%).

The weighting makes predictions **more similar across references** (pairwise dev reduction) but does not make them **better** (Dice) or substantially **less variant** (ref std). This is a "compression without accuracy gain" pattern.

## 8. Compute

| Resource | Used |
|---|---|
| GPU-hours | 0.00 |
| SAM calls (Phase 2 only) | 123 |
| Best model size | M4_gbm (small) |

## 9. Reference

- `reports/phaseS_trackB/gateB_summary.json`
- `reports/phaseS_trackB/gateB_summary.md`
- `reports/phaseS_trackB/train_model_cv.json`
- `reports/phaseS_trackB/val_features.csv`
- `reports/phaseS_trackB/val_per_reference_predictions.csv`
- `reports/phaseS_trackB/val_weights.csv`
- `reports/phaseS_trackB/val_weighted_consensus_metrics.csv`
