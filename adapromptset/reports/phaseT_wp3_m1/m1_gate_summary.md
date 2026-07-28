# Phase T WP3 M1 Dynamic Reference Budget Gate Summary

Date: 2026-07-23T18:18:43
Best model: **M1-3_handcrafted_threshold**
Selected lambdas: lambda_c=0.0, lambda_r=0.0
Train episodes: 1200 (80 queries)
Val episodes: 615 (41 queries)

## Model comparison (val)

| model | type | lambda_c | lambda_r | mean_dice | avg_k | cvar_0_1 | cfr | cost | mean_u |
|-------|------|----------|----------|-----------|-------|----------|-----|------|--------|
| M1-0_always_k1 | baseline | 0.0 | 0.0 | 0.414136 | 1.0000 | 0.059762 | 0.117073 | 0.166667 | 0.301199 |
| M1-1_always_k3 | baseline | 0.0 | 0.0 | 0.443245 | 3.0000 | 0.054352 | 0.115447 | 0.500000 | 0.321974 |
| M1-2_always_k6 | baseline | 0.0 | 0.0 | 0.458553 | 6.0000 | 0.048999 | 0.125203 | 1.000000 | 0.324782 |
| M1-3_handcrafted_threshold | threshold | 0.0 | 0.0 | 0.457056 | 5.6341 | 0.053718 | 0.120325 | 0.939024 | 0.324810 |
| M1-4_logistic_regression | logistic | 0.0 | 0.0 | 0.455240 | 5.8780 | 0.048977 | 0.125203 | 0.979675 | 0.321977 |
| M1-5_gradient_boosting | gradient_boosting | 0.0 | 0.0 | 0.451626 | 4.9805 | 0.052773 | 0.117073 | 0.830081 | 0.322103 |

## Baselines vs policy (val)

| variant | mean_dice | cvar_0_1 | cfr | avg_k |
|---------|-----------|----------|-----|-------|
| k=6 (baseline) | 0.458553 | 0.048999 | 0.125203 | 6.0000 |
| k=3 (baseline) | 0.443245 | 0.054352 | 0.115447 | 3.0000 |
| policy (M1-3_handcrafted_threshold) | 0.457056 | 0.053718 | 0.120325 | 5.6341 |

## Gate conditions (section 9.6)

### Cost-saving mode (vs k=6)

| # | condition | pass |
|---|-----------|------|
| cs1 | Dice not below k=6 by > 0.003 | PASS |
| cs2 | CVaR_0.1 not decrease by > 0.005 | PASS |
| cs3 | CFR not increase vs k=6 | PASS |
| cs4 | avg_k < 4.5 (>= 25% reduction) | FAIL |

**cs1** (`cs1_dice_not_below_k6_by_0p003`): **PASS** -- Dice diff = -0.001496 (policy=0.457056, k6=0.458553)

**cs2** (`cs2_cvar_not_decrease_by_0p005`): **PASS** -- CVaR diff = +0.004718 (policy=0.053718, k6=0.048999)

**cs3** (`cs3_cfr_not_increase`): **PASS** -- CFR diff = -0.004878 (policy=0.120325, k6=0.125203)

**cs4** (`cs4_avg_ref_count_reduce_25pct`): **FAIL** -- avg_k = 5.6341, reduction = 6.1% (threshold < 4.5)

Cost-saving mode: **FAIL**

### Quality-cost mode (vs k=3)

| # | condition | pass |
|---|-----------|------|
| qc1 | Dice >= k=3 + 0.0075 (and avg_k in [2.5, 3.5]) | FAIL |
| qc2 | CVaR_0.1 improve vs k=3 | FAIL |
| qc3 | CFR not increase vs k=3 | FAIL |

**qc1** (`qc1_dice_ge_k3_plus_0p0075`): **FAIL** -- Dice diff vs k=3 = +0.013811 but avg_k = 5.6341 not in [2.5, 3.5]

**qc2** (`qc2_cvar_improve`): **FAIL** -- CVaR diff vs k=3 = -0.000635 (policy=0.053718, k3=0.054352)

**qc3** (`qc3_cfr_not_increase`): **FAIL** -- CFR diff vs k=3 = +0.004878 (policy=0.120325, k3=0.115447)

Quality-cost mode: **FAIL**

## Decision: **PHASET_M1_DYNAMIC_BUDGET_NO_GAIN**

Pass if EITHER cost-saving mode OR quality-cost mode passes.

