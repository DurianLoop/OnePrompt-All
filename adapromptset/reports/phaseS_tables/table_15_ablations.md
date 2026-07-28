# Table 15 — Ablations

## 1. Track C α/β/γ Ablation (Mean Train Dice at Each Level)

Single-variable ablation (averaging over the other two parameters), 80 train queries × 3 configs (P20-N1, P30-N0, P50-N0), 27-combo grid.

### 1.1 α Ablation (C3 Pixel-wise Agreement Gate)

| α | Mean train Dice | Δ vs α=0 | Status |
|---|---|---|---|
| 0 | 0.5003 | — (baseline) | best |
| 1 | 0.4901 | -0.0102 | worse |
| 5 | 0.4795 | -0.0208 | worst |

**C3 (agreement gating) HURTS Dice.** All non-zero α values decrease Dice monotonically. Best α = 0 (no agreement gating).

### 1.2 β Ablation (C4 Stable Candidate Scoring — contrast-mean)

| β | Mean train Dice | Δ vs β=0 | Status |
|---|---|---|---|
| 0 | 0.4951 | — (baseline) | best |
| 0.5 | 0.4922 | -0.0029 | worse |
| 1 | 0.4905 | -0.0046 | worst |

**C4 contrast-mean component HURTS Dice.** Smaller effect than α. Best β = 0.

### 1.3 γ Ablation (C4 Stable Candidate Scoring — agreement fraction)

| γ | Mean train Dice | Δ vs γ=0 | Status |
|---|---|---|---|
| 0 | 0.4940 | — (baseline) | best |
| 0.5 | 0.4922 | -0.0018 | worse |
| 1 | 0.4917 | -0.0023 | worst |

**C4 agreement-fraction component HURTS Dice.** Smallest effect. Best γ = 0.

## 2. Track C Combined Ablation Summary

| Setting | Mean train Dice |
|---|---|
| Best (α=0, β=0, γ=0) = B2_mean | 0.503761 |
| Worst (α=5, β=1, γ=1) | 0.4789 |
| Range | 0.0249 |

**Conclusion:** Every C3/C4 method parameter moves Dice AWAY from the optimum. The Track C contribution is **negative** across the board.

## 3. Track B T/ε/λ Ablation Note

Track B's M4_gbm weighting uses three hyperparameters:
- T (temperature) = 0.25
- ε (weight floor) = 0.0
- λ (risk penalty) = 0.0

These were tuned on inner CV (train) and the selected values are T=0.25, ε=0.0, λ=0.0. Ablation across T ∈ {0.1, 0.25, 0.5, 1.0} showed minimal effect on val Dice (range <0.003). ε > 0 and λ > 0 both slightly worsened val Dice by flattening the weighting.

The T/ε/λ ablation is summarised in `reports/phaseS_trackB/gateB_summary.md` — the main finding is that Track B is **insensitive** to T/ε/λ in the sense that no (T, ε, λ) combination closes the Dice gap to B2_mean.

## 4. Track A: Test-Time-Safe Feature Ablation

Best single feature: `prediction_jaccard_mean` (Spearman 0.8832 on train, P50-N0, std_dice target).

| Feature | Target | Train Spearman |ρ| |
|---|---|---|
| prediction_jaccard_mean | std_dice | 0.8832 |
| prediction_jaccard_mean | V_ref | 0.8471 |
| mean_ref_std | V_ref | 0.7812 |
| best_worst_gap | worst_dice | 0.7302 |
| mean_pairwise_dev | CVaR_0.1 | 0.7102 |

The strongest test-time-safe predictor is `prediction_jaccard_mean` — a feature computed from SAM outputs only, no ground-truth access.

## 5. Reference-Count Ablation (k)

| k | Mean Dice (P30-N0) | V_ref | CVaR_0.1 | CFR |
|---|---|---|---|---|
| 1 | 0.4188 | 0.0080 | 0.0583 | 0.1463 |
| 3 | 0.4521 | 0.0068 | 0.0661 | 0.1122 |
| 6 | 0.4756 | 0.0055 | 0.0738 | 0.0732 |

**k=6 dominates k=1 and k=3 on all metrics.** B2_mean at k=6 is the primary baseline.

## 6. Reference

- `reports/phaseS_trackC/train_grid_results.csv`
- `reports/phaseS_trackB/gateB_summary.md`
- `reports/phaseS_trackA_benchmark/reference_count_scaling.csv`
- `reports/phaseS_trackA_metric_audit/prediction_consistency.csv`
