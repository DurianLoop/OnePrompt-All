# Phase Y Gate Y1: Fresh-Query Action Heterogeneity Summary

Date: 2026-07-21T14:55:31
Queries (validation, A∩B): 41
Actions evaluated: 210

## Upper Bounds (Dice)

| Metric | Value |
|---|---:|
| Best fixed action (Set A) | Y_P_TOPK_K50_N0_N_NONE_G1_C1 |
| Fixed-A Dice | 0.4338 |
| Oracle-A (raw) | 0.5007 |
| Oracle-B (raw) | 0.4836 |
| Cross-Ref A→B | 0.4258 |
| Cross-Ref B→A | 0.4539 |
| **Stable Oracle** | **0.4399** |
| Optimism A | 0.0749 |
| Optimism B | 0.0297 |
| Raw gap A | 0.0669 |
| **Stable Oracle Gap** | **0.0061** |
| Bootstrap 95% CI | [-0.0572, 0.0699] |
| Optimism ratio A | 1.1197 |

## Action Stability

| Metric | Value |
|---|---:|
| Exact action agreement | 0.0488 |
| K+ bin agreement (±5) | 0.3902 |
| Selector agreement | 1.0000 |
| Median Spearman ρ | 0.7450 |
| Mean top-3 overlap | 0.0894 |
| Win consistency | 0.8293 |
| Diff from fixed (≥30%) | 0.0000 |

## Gate Y1 Conditions

| # | Condition | Pass |
|---|---|---|
| 1 | cond1_stable_oracle_gap_ge_0p015 | NO |
| 1 | cond2_bootstrap_ci_lo_gt_0 | NO |
| 1 | cond3_30pct_diff_from_fixed | NO |
| 1 | cond4_median_spearman_rho_gt_0p30 | YES |
| 1 | cond5_optimism_le_60pct_gap | NO |
| 1 | cond6_all_one_sam_call | YES |

## Decision: **PHASEY_GATE_Y1_ORACLE_GAP_SMALL**

