# Table 08 — Prediction Consistency (J_pred)

**Goal:** Quantify whether the SAM-refined prediction **mask** (not just Dice) is stable across references. J_pred is the mean pairwise pixel-Jaccard between predictions C_{q,r} across references.

## 1. Definitions

| Metric | Definition |
|---|---|
| `J_pred` (single) | Mean pairwise pixel-Jaccard between single-reference predictions C_{q,r} within a query (the "A set") |
| `J_pred` (consensus) | Mean pairwise pixel-Jaccard between consensus predictions across reference subsets (the "B set") |
| `prediction_jaccard_mean` | Mean of J_pred over reference pairs (the strongest test-time-safe predictor of reference instability) |

## 2. J_pred Single vs Consensus (P30-N0, primary config)

| k | J_pred single (A set) | J_pred consensus (B set) | A vs B |
|---|---|---|---|
| 1 | 0.4218 | — | — |
| 3 | 0.4396 | 0.6128 | 0.5102 |
| 6 | 0.4511 | 0.6841 | 0.5618 |

(Per-query values in `reports/phaseS_trackA_benchmark/prediction_consistency.csv`.)

## 3. Per-Config Summary (k=6)

| Config | J_pred single | J_pred consensus | A vs B |
|---|---|---|---|
| P20-N1 | 0.4012 | 0.6518 | 0.5288 |
| P30-N0 | 0.4511 | 0.6841 | 0.5618 |
| P50-N0 | 0.4882 | 0.7212 | 0.6044 |

## 4. Key Findings

- **Single-reference J_pred is ~0.42–0.49** — meaning that on average, swapping references changes roughly **half** of the predicted foreground pixels.
- **Consensus J_pred is ~0.65–0.72** — averaging k references substantially stabilises the *mask* (more than it stabilises Dice, because mask-Jaccard is more sensitive to small shifts).
- **A vs B J_pred ~0.51–0.60** — even with consensus, single-reference predictions differ from consensus predictions on ~40–50% of pixels.
- The mask-level instability is even more pronounced than the Dice-level instability (Table 04).

## 5. Test-Time-Safe Predictability

| Train feature | Target | Config | Spearman |ρ| (train) |
|---|---|---|---|
| `prediction_jaccard_mean` | std_dice | P50-N0 | **0.8832** |
| `prediction_jaccard_mean` | V_ref | P50-N0 | 0.8471 |
| `prediction_jaccard_mean` | worst_dice | P30-N0 | 0.7912 |

`prediction_jaccard_mean` is the **strongest test-time-safe predictor** of reference instability:
- It is computed from SAM outputs only (no ground truth).
- It correlates strongly (|ρ|=0.88) with reference-induced Dice variance.
- This validates Gate A6 (test-time-safe predictability).

## 6. Why This Matters

The predictability of reference instability from test-time-safe features is the **scientific finding** that supports the Track B Utility-Only claim:
- Reference utility is predictable from test-time-safe features (Spearman = 0.478 on val for Track B).
- But weighted consensus does NOT reduce variance or improve Dice (Track B NO_GAIN).
- The utility signal is a **scientific finding without a deployable stability gain**.

## 7. Reference

- `reports/phaseS_trackA_benchmark/prediction_consistency.csv`
- `reports/phaseS_trackA_metric_audit/prediction_consistency.csv`
- `reports/phaseS_trackB/val_per_reference_predictions.csv`
- `reports/phaseS_trackB/val_features.csv`
