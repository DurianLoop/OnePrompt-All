# Phase T WP2 Cross-Dataset Replication Benchmark Summary (MONUSEG)

Date: 2026-07-23T14:02:53
Dataset: monuseg
Validation queries: 30
Resampling seeds: 5
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
| P20-N1 | 1 | 30 | 0.002312 | 0.012126 | 0.1602 | 0.040934 | 0.117698 |
| P20-N1 | 3 | 30 | 0.001460 | 0.013504 | 0.0976 | 0.030962 | 0.082904 |
| P20-N1 | 6 | 30 | 0.000934 | 0.012401 | 0.0700 | 0.023737 | 0.064261 |
| P30-N0 | 1 | 30 | 0.002602 | 0.014211 | 0.1547 | 0.038700 | 0.110868 |
| P30-N0 | 3 | 30 | 0.001766 | 0.015310 | 0.1034 | 0.026600 | 0.068146 |
| P30-N0 | 6 | 30 | 0.001529 | 0.016009 | 0.0872 | 0.021345 | 0.054957 |
| P50-N0 | 1 | 30 | 0.002635 | 0.015080 | 0.1487 | 0.036889 | 0.105718 |
| P50-N0 | 3 | 30 | 0.001241 | 0.016812 | 0.0687 | 0.019632 | 0.048794 |
| P50-N0 | 6 | 30 | 0.001451 | 0.016751 | 0.0797 | 0.019678 | 0.049758 |

## Mean Dice & tail risk by (config, k)

| config | k | mean_dice | mean_worst | min_worst | p10_worst | CVaR_0.1 | CFR |
|--------|---|-----------|------------|-----------|-----------|----------|-----|
| P20-N1 | 1 | 0.478543 | 0.369944 | 0.219150 | 0.225268 | 0.258025 | 0.0000 |
| P20-N1 | 3 | 0.493297 | 0.447858 | 0.240838 | 0.293441 | 0.267467 | 0.0000 |
| P20-N1 | 6 | 0.497822 | 0.465629 | 0.241714 | 0.315592 | 0.279257 | 0.0000 |
| P30-N0 | 1 | 0.498565 | 0.393593 | 0.223537 | 0.239874 | 0.257123 | 0.0000 |
| P30-N0 | 3 | 0.511970 | 0.478721 | 0.245965 | 0.298153 | 0.263341 | 0.0000 |
| P30-N0 | 6 | 0.514406 | 0.489427 | 0.248325 | 0.301025 | 0.264512 | 0.0000 |
| P50-N0 | 1 | 0.504143 | 0.405978 | 0.222966 | 0.232770 | 0.255320 | 0.0000 |
| P50-N0 | 3 | 0.514653 | 0.489881 | 0.242893 | 0.283182 | 0.260382 | 0.0000 |
| P50-N0 | 6 | 0.516845 | 0.494909 | 0.248433 | 0.275197 | 0.261689 | 0.0000 |

## WP2 Cross-Dataset Gate — Two-Tier

### Tier 1: Directional Support (5 conditions, ALL must pass)

| # | condition | pass |
|---|-----------|------|
| d1 | >= 30% queries best-worst gap >= 0.05 | PASS |
| d2 | mean within-query reference std >= 0.015 | PASS |
| d3 | mean consensus Dice >= random single Dice (not below) | PASS |
| d4 | catastrophic failure rate not increased (consensus CFR <= single CFR) | PASS |
| d5 | prediction consistency not decreased (consensus J_pred >= single J_pred) | PASS |

### Tier 2: Strong Support (4 additional conditions)

| # | condition | pass |
|---|-----------|------|
| s1 | mean consensus Dice >= random single + 0.01 | PASS |
| s2 | bootstrap point estimate positive (CI lower > 0) | PASS |
| s3 | CVaR improved (consensus CVaR >= single CVaR - 0.005) | PASS |
| s4 | instability feature correlation >= 0.20 (best |Spearman|) | PASS |

**d1_30pct_queries_best_worst_gap_ge_0p05** (directional): **PASS** — fraction of queries with mean best-worst gap>=0.05 = 0.8000 (n=30); threshold 0.3

**d2_mean_within_query_ref_std_ge_0p015** (directional): **PASS** — mean within-query reference std (k=1) = 0.038841; threshold 0.015

**d3_consensus_dice_ge_single_dice** (directional): **PASS** — mean consensus(k=6) Dice = 0.509691, mean B1-single(k=1) Dice = 0.494753, gain = 0.014938

**d4_consensus_cfr_le_single_cfr** (directional): **PASS** — consensus CFR(k=6) = 0.000000, single CFR(B1) = 0.000000

**d5_consensus_j_pred_ge_single_j_pred** (directional): **PASS** — mean cross-seed J_pred: consensus(k=6) = 0.829572, B1-single(k=1) = 0.732723

**s1_consensus_gain_ge_0p01** (strong): **PASS** — consensus gain = 0.014938; threshold 0.01

**s2_bootstrap_ci_lower_gt_0** (strong): **PASS** — paired bootstrap (by query, 10000 iters) 95% CI lower bound = 0.006911; n_queries=30

**s3_consensus_cvar_ge_single_cvar_minus_tol** (strong): **PASS** — consensus CVaR_0.1(k=6) = 0.267866, single CVaR_0.1(B1) = 0.258313, tolerance 0.005

**s4_disagreement_feature_spearman_rho_ge_0p20** (strong): **PASS** — best |Spearman(prediction_jaccard_k1_within, best_worst_gap_k1)| = 0.8266; per feature {'matching_map_disagreement': 0.2287, 'candidate_jaccard_mean_k1': 0.4123, 'prompt_overlap_mean_k1': 0.4123, 'prediction_jaccard_k1_within': 0.8266}; threshold 0.2

## Decision: **PHASET_DATASET_STRONG_SUPPORT**

Directional: 5/5 passed. Strong: 4/4 passed.
