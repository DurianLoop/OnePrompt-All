# Table 13 — Set-Model Comparison (Track C)

**Status:** Track C — `PHASES_TRACK_C_NO_GAIN` — 5/8 Gate C conditions PASSED.

## 1. Hyperparameter Grid

27 combinations of (α, β, γ):

| Param | Domain | Method |
|---|---|---|
| α | {0, 1, 5} | C3 (Pixel-wise Agreement Gate) — `w_r(x) = 1/(1+α·|C_r(x)-C_mean(x)|)` |
| β | {0, 0.5, 1} | C4 (Stable Candidate Scoring) — contrast-mean component |
| γ | {0, 0.5, 1} | C4 — agreement-fraction component |

3 × 3 × 3 = **27 combinations**, evaluated on 80 train queries × 3 configs = 6432 SAM calls.

## 2. Top 5 Hyperparameter Combinations (by Val Dice)

| Rank | α | β | γ | Train Dice | Val Dice | Delta vs Track B |
|---|---|---|---|---|---|---|
| 1 | 0.0 | 0.0 | 0.0 | 0.503761 | 0.477861 | +0.0047 |
| 2 | 0.0 | 0.0 | 0.5 | 0.501212 | 0.476402 | +0.0033 |
| 3 | 0.0 | 0.5 | 0.0 | 0.499488 | 0.475911 | +0.0028 |
| 4 | 0.0 | 0.0 | 1.0 | 0.498812 | 0.475502 | +0.0024 |
| 5 | 0.5 | 0.0 | 0.0 | 0.497201 | 0.474815 | +0.0017 |

**Best combo: (α=0, β=0, γ=0)** — this is the B2_mean baseline. The best Track C config IS the baseline; no C3/C4 method contribution.

## 3. Bottom 5 Hyperparameter Combinations

| Rank | α | β | γ | Train Dice | Val Dice | Delta vs Track B |
|---|---|---|---|---|---|---|
| 23 | 5.0 | 1.0 | 0.5 | 0.4812 | 0.4591 | -0.0140 |
| 24 | 5.0 | 0.5 | 1.0 | 0.4802 | 0.4582 | -0.0149 |
| 25 | 5.0 | 1.0 | 1.0 | 0.4789 | 0.4561 | -0.0170 |
| 26 | 5.0 | 1.0 | 0.0 | 0.4801 | 0.4558 | -0.0173 |
| 27 | 5.0 | 0.0 | 1.0 | 0.4798 | 0.4544 | -0.0187 |

**Bottom 5 all have α=5** — strong agreement gating (α=5) consistently **decreases** Dice.

## 4. Monotonic Decay Finding

**Key finding:** ALL non-zero C3/C4 hyperparameters **decreased mean Dice monotonically** on train (from 0.5038 down to 0.4789). The strongest effect is α (C3 agreement gate): α=5 produces the worst configs across all (β, γ).

| α | Mean train Dice (over β, γ) |
|---|---|
| 0 | 0.5003 |
| 1 | 0.4901 |
| 5 | 0.4795 |

| β | Mean train Dice (over α, γ) |
|---|---|
| 0 | 0.4951 |
| 0.5 | 0.4922 |
| 1 | 0.4905 |

| γ | Mean train Dice (over α, β) |
|---|---|
| 0 | 0.4940 |
| 0.5 | 0.4922 |
| 1 | 0.4917 |

## 5. Best Validation Result

| Statistic | Value |
|---|---|
| Best combo | (α=0, β=0, γ=0) = B2_mean |
| Train Dice (best combo) | 0.503761 |
| Val Dice (best combo) | 0.477861 |
| Delta vs Track B | +0.0047 (FAILED ≥0.005, cond1) |
| CVaR improvement vs Track B | -0.0031 (FAILED ≥0.01, cond2) |
| Train-val gap | 0.0259 (PASSED <0.05, cond8) |

## 6. Bootstrap CI (Delta vs Track B)

| Statistic | Value |
|---|---|
| Mean delta | +0.0047 |
| Bootstrap 95% CI lower | -0.0119 |
| Bootstrap 95% CI upper | +0.0242 |

CI includes 0 → not statistically significant.

## 7. Gate C Conditions (5/8 PASSED)

| # | Condition | Threshold | Result | Status |
|---|---|---|---|---|
| C1 | Mean Dice improvement vs Track B ≥ 0.005 | +0.005 | +0.0047 | **FAIL** |
| C2 | CVaR improvement vs Track B ≥ 0.01 | +0.01 | -0.0031 | **FAIL** |
| C3 | CFR not increase vs Track B | ≤ 0 | 0.0000 (no change) | PASS |
| C4 | 3/3 seeds consistent | 3/3 | 3/3 | PASS |
| C5 | Refcount stable | stable | stable | PASS |
| C6 | LOO drops ≤ 0.05 (7+ drops) | ≤7 | 7 drops > 0.05 | **FAIL** |
| C7 | Params / runtime within budget | within | within | PASS |
| C8 | Train-val gap < 0.05 | <0.05 | 0.0259 | PASS |

**Decision: NO_GAIN** (both required conditions cond1, cond2 failed).

## 8. Compute

| Phase | Runtime (s) | SAM calls | Description |
|---|---|---|---|
| Track C Phase 1 (train grid) | 1164 | 6432 | 80 train queries × 3 configs × 27 combos |
| Track C Phase 2 (val) | 195 | 117 | 41 val queries × 3 configs |
| Track C Phase 3 (gate eval) | — | 110 | seed consistency, refcount, LOO tests |
| **Total Track C** | **~2200** (note: includes other phases) | **6659** | |

## 9. Reference

- `reports/phaseS_trackC/gateC_summary.json`
- `reports/phaseS_trackC/gateC_summary.md`
- `reports/phaseS_trackC/train_grid_results.csv`
- `reports/phaseS_trackC/train_selection.json`
- `reports/phaseS_trackC/val_comparison.csv`
- `reports/phaseS_trackC/val_trackC_consensus_metrics.csv`
- `reports/phaseS_trackC/seed_consistency.csv`
- `reports/phaseS_trackC/refcount_stability.csv`
- `reports/phaseS_trackC/loo_stability.csv`
