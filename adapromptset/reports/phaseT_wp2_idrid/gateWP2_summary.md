# Phase T WP2 Cross-Dataset Replication Benchmark Summary (IDRID)

Date: 2026-07-23T14:12:54
Dataset: idrid
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
| P20-N1 | 1 | 30 | 0.053597 | 0.024716 | 0.6844 | 0.163531 | 0.407983 |
| P20-N1 | 3 | 30 | 0.057824 | 0.060587 | 0.4883 | 0.169122 | 0.407726 |
| P20-N1 | 6 | 30 | 0.041470 | 0.028490 | 0.5928 | 0.121526 | 0.283718 |
| P30-N0 | 1 | 30 | 0.001283 | 0.000552 | 0.6991 | 0.014260 | 0.040578 |
| P30-N0 | 3 | 30 | 0.000304 | 0.000950 | 0.2426 | 0.008807 | 0.023108 |
| P30-N0 | 6 | 30 | 0.000093 | 0.000837 | 0.1003 | 0.004701 | 0.012810 |
| P50-N0 | 1 | 30 | 0.000018 | 0.000079 | 0.1826 | 0.003827 | 0.009947 |
| P50-N0 | 3 | 30 | 0.000009 | 0.000077 | 0.1047 | 0.002459 | 0.006397 |
| P50-N0 | 6 | 30 | 0.000006 | 0.000084 | 0.0636 | 0.001812 | 0.004579 |

## Mean Dice & tail risk by (config, k)

| config | k | mean_dice | mean_worst | min_worst | p10_worst | CVaR_0.1 | CFR |
|--------|---|-----------|------------|-----------|-----------|----------|-----|
| P20-N1 | 1 | 0.194513 | 0.039677 | 0.000226 | 0.026892 | 0.030582 | 0.7089 |
| P20-N1 | 3 | 0.260667 | 0.089054 | 0.028603 | 0.040772 | 0.035664 | 0.6533 |
| P20-N1 | 6 | 0.171558 | 0.060737 | 0.028320 | 0.040877 | 0.035461 | 0.7267 |
| P30-N0 | 1 | 0.057404 | 0.040188 | 0.000136 | 0.029251 | 0.028976 | 0.9556 |
| P30-N0 | 3 | 0.059682 | 0.050873 | 0.027877 | 0.032539 | 0.030183 | 0.9333 |
| P30-N0 | 6 | 0.058178 | 0.052696 | 0.027447 | 0.035119 | 0.030168 | 0.9400 |
| P50-N0 | 1 | 0.043178 | 0.037421 | 0.021394 | 0.028961 | 0.027061 | 1.0000 |
| P50-N0 | 3 | 0.042145 | 0.039130 | 0.022630 | 0.029694 | 0.027859 | 1.0000 |
| P50-N0 | 6 | 0.041581 | 0.039242 | 0.022587 | 0.029710 | 0.026650 | 1.0000 |

## WP2 Cross-Dataset Gate — Two-Tier

### Tier 1: Directional Support (5 conditions, ALL must pass)

| # | condition | pass |
|---|-----------|------|
| d1 | >= 30% queries best-worst gap >= 0.05 | PASS |
| d2 | mean within-query reference std >= 0.015 | PASS |
| d3 | mean consensus Dice >= random single Dice (not below) | FAIL |
| d4 | catastrophic failure rate not increased (consensus CFR <= single CFR) | PASS |
| d5 | prediction consistency not decreased (consensus J_pred >= single J_pred) | PASS |

### Tier 2: Strong Support (4 additional conditions)

| # | condition | pass |
|---|-----------|------|
| s1 | mean consensus Dice >= random single + 0.01 | FAIL |
| s2 | bootstrap point estimate positive (CI lower > 0) | FAIL |
| s3 | CVaR improved (consensus CVaR >= single CVaR - 0.005) | PASS |
| s4 | instability feature correlation >= 0.20 (best |Spearman|) | PASS |

**d1_30pct_queries_best_worst_gap_ge_0p05** (directional): **PASS** — fraction of queries with mean best-worst gap>=0.05 = 0.6333 (n=30); threshold 0.3

**d2_mean_within_query_ref_std_ge_0p015** (directional): **PASS** — mean within-query reference std (k=1) = 0.060539; threshold 0.015

**d3_consensus_dice_ge_single_dice** (directional): **FAIL** — mean consensus(k=6) Dice = 0.090439, mean B1-single(k=1) Dice = 0.102190, gain = -0.011751

**d4_consensus_cfr_le_single_cfr** (directional): **PASS** — consensus CFR(k=6) = 0.888889, single CFR(B1) = 0.895556

**d5_consensus_j_pred_ge_single_j_pred** (directional): **PASS** — mean cross-seed J_pred: consensus(k=6) = 0.853865, B1-single(k=1) = 0.784265

**s1_consensus_gain_ge_0p01** (strong): **FAIL** — consensus gain = -0.011751; threshold 0.01

**s2_bootstrap_ci_lower_gt_0** (strong): **FAIL** — paired bootstrap (by query, 10000 iters) 95% CI lower bound = -0.031757; n_queries=30

**s3_consensus_cvar_ge_single_cvar_minus_tol** (strong): **PASS** — consensus CVaR_0.1(k=6) = 0.029062, single CVaR_0.1(B1) = 0.028745, tolerance 0.005

**s4_disagreement_feature_spearman_rho_ge_0p20** (strong): **PASS** — best |Spearman(prediction_jaccard_k1_within, best_worst_gap_k1)| = 0.9519; per feature {'matching_map_disagreement': 0.1017, 'candidate_jaccard_mean_k1': 0.4014, 'prompt_overlap_mean_k1': 0.4014, 'prediction_jaccard_k1_within': 0.9519}; threshold 0.2

## Decision: **PHASET_DATASET_FAIL**

Directional: 4/5 passed. Strong: 2/4 passed.
