# Phase T WP1 MoNuSAC Holdout Confirmatory Benchmark Summary

Date: 2026-07-23T13:00:15
Holdout queries: 40
Resampling seeds: 10
Prompt configs: P20-N1, P30-N0, P50-N0
Reference subset sizes k: 1, 3, 6

## k-scaling design

| k  | subsets per (query, seed) | source | subset_id |
|----|---------------------------|--------|-----------|
| 1  | 6 (individual refs)        | per-reference s_fg pool | 0..5 |
| 3  | 1 (mean consensus first 3) | B2_mean contrast | first3 |
| 6  | 1 (mean consensus all 6)   | B2_mean contrast | all6 |

## Within-query reference variance V_ref(k)

| config | k | n_queries | V_ref | V_query | V_ref_frac | mean_ref_std | best-worst gap |
|--------|---|-----------|-------|---------|------------|--------------|----------------|
| P20-N1 | 1 | 40 | 0.011088 | 0.031122 | 0.2627 | 0.085208 | 0.239294 |
| P20-N1 | 3 | 40 | 0.009507 | 0.032709 | 0.2252 | 0.080181 | 0.262545 |
| P20-N1 | 6 | 40 | 0.008304 | 0.035611 | 0.1891 | 0.069922 | 0.225234 |
| P30-N0 | 1 | 40 | 0.008818 | 0.032969 | 0.2110 | 0.077195 | 0.214120 |
| P30-N0 | 3 | 40 | 0.007938 | 0.035122 | 0.1843 | 0.072750 | 0.230197 |
| P30-N0 | 6 | 40 | 0.006790 | 0.035913 | 0.1590 | 0.061878 | 0.196472 |
| P50-N0 | 1 | 40 | 0.005186 | 0.035860 | 0.1264 | 0.060222 | 0.166836 |
| P50-N0 | 3 | 40 | 0.003903 | 0.038013 | 0.0931 | 0.053054 | 0.169467 |
| P50-N0 | 6 | 40 | 0.002579 | 0.039515 | 0.0613 | 0.040395 | 0.132370 |

## Mean Dice & tail risk by (config, k)

| config | k | mean_dice | mean_worst | min_worst | p10_worst | CVaR_0.1 | CFR |
|--------|---|-----------|------------|-----------|-----------|----------|-----|
| P20-N1 | 1 | 0.461742 | 0.221415 | 0.000000 | 0.000384 | 0.056059 | 0.0729 |
| P20-N1 | 3 | 0.482079 | 0.333483 | 0.000136 | 0.018278 | 0.077208 | 0.0675 |
| P20-N1 | 6 | 0.488852 | 0.364820 | 0.000000 | 0.044408 | 0.076556 | 0.0650 |
| P30-N0 | 1 | 0.487949 | 0.305835 | 0.000000 | 0.037340 | 0.075086 | 0.0621 |
| P30-N0 | 3 | 0.505194 | 0.372114 | 0.001982 | 0.048409 | 0.093938 | 0.0575 |
| P30-N0 | 6 | 0.516846 | 0.400154 | 0.016060 | 0.087525 | 0.102650 | 0.0575 |
| P50-N0 | 1 | 0.478970 | 0.331364 | 0.005334 | 0.084058 | 0.083139 | 0.0550 |
| P50-N0 | 3 | 0.498417 | 0.404445 | 0.016912 | 0.191492 | 0.095159 | 0.0525 |
| P50-N0 | 6 | 0.511686 | 0.440720 | 0.015963 | 0.206773 | 0.098674 | 0.0500 |

## WP1 Gate — 10 conditions (§5.6)

| # | condition | pass |
|---|-----------|------|
| 1 | >= 50% holdout queries best-worst gap >= 0.05 | PASS |
| 2 | mean within-query reference std >= 0.02 | PASS |
| 3 | V_ref_frac >= 0.10 | PASS |
| 4 | mean consensus Dice - random single Dice >= 0.010 | PASS |
| 5 | paired bootstrap 95% CI lower bound > 0 | PASS |
| 6 | catastrophic failure not increase (consensus CFR <= single CFR) | PASS |
| 7 | CVaR_0.1 not decrease (consensus CVaR >= single CVaR - 0.005) | PASS |
| 8 | prediction consistency improve (consensus J_pred >= single J_pred) | PASS |
| 9 | disagreement feature Spearman |rho| >= 0.25 | PASS |
| 10 | no data leakage (ref patients != query patients) | PASS |

**cond1** (`cond1_50pct_queries_best_worst_gap_ge_0p05`): **PASS** — fraction of holdout queries with mean best-worst gap>=0.05 = 0.9250 (n=40); threshold 0.50

**cond2** (`cond2_mean_within_query_ref_std_ge_0p02`): **PASS** — mean within-query reference std (k=1) = 0.074208; threshold 0.02

**cond3** (`cond3_v_ref_frac_ge_0p10`): **PASS** — V_ref_frac (k=1) per config = {'P20-N1': 0.2627, 'P30-N0': 0.211, 'P50-N0': 0.1264}; threshold 0.10 (any config)

**cond4** (`cond4_consensus_gain_ge_0p010`): **PASS** — mean consensus(k=6) Dice = 0.505795, mean B1-single(k=1) Dice = 0.479164, gain = 0.026631; threshold 0.01

**cond5** (`cond5_paired_bootstrap_ci_lower_gt_0`): **PASS** — paired bootstrap (by query, 10000 iters) 95% CI lower bound = 0.014009; n_queries=40

**cond6** (`cond6_consensus_cfr_le_single_cfr`): **PASS** — consensus CFR(k=6) = 0.057500, single CFR(B1) = 0.063333

**cond7** (`cond7_consensus_cvar_ge_single_cvar_minus_tol`): **PASS** — consensus CVaR_0.1(k=6) = 0.091945, single CVaR_0.1(B1) = 0.075133, tolerance 0.005

**cond8** (`cond8_consensus_j_pred_ge_single_j_pred`): **PASS** — mean cross-seed J_pred: consensus(k=6) = 0.718951, B1-single(k=1) = 0.648965

**cond9** (`cond9_disagreement_feature_spearman_rho_ge_0p25`): **PASS** — best |Spearman(prediction_jaccard_k1_within, best_worst_gap_k1)| = 0.7241; per feature {'matching_map_disagreement': 0.1434, 'candidate_jaccard_mean_k1': 0.103, 'prompt_overlap_mean_k1': 0.103, 'prediction_jaccard_k1_within': 0.7241}; threshold 0.25

**cond10** (`cond10_no_data_leakage`): **PASS** — patient-disjoint check: 0 violations / 2400 reference assignments across 10 seeds

## Decision: **PHASET_WP1_MONUSAC_HOLDOUT_CONFIRMED**

(10/10 conditions passed)
