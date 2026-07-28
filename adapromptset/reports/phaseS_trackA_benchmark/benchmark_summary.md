# Phase S — Track A §6.4 Reference Resampling Benchmark

## §6.5 Gate A Evaluation


Computed from EXISTING Phase R artefacts (no new SAM inference).

- Validation queries: **41**
- Configs: P20-N1, P30-N0, P50-N0
- Reference subset sizes k: 1, 3, 6
- Consensus method (k=3, k=6): `B2_mean`
- Catastrophic Dice threshold: 0.1

### k-scaling design

| k  | subsets per query | source | subset_id |
|----|-------------------|--------|-----------|
| 1  | 6 (individual refs) | val_per_reference_metrics | 0..5 (A0..A2,B0..B2) |
| 3  | 2 (A_only, B_only)  | val_consensus_metrics B2_mean | A, B |
| 6  | 1 (all6)            | val_consensus_metrics B2_mean | all6 |

## 1. Within-query reference variance V_ref(k)

V_ref(k) = (1/N) Σ_q Var_S[D(q,S)] (ddof=0). V_ref_frac = V_ref/(V_ref+V_query).

| config | k | n_queries | V_ref | V_query | V_ref_frac | mean_ref_std | best-worst gap |
|--------|---|-----------|-------|---------|------------|--------------|----------------|
| P20-N1 | 1 | 41 | 0.016721 | 0.033764 | 0.3312 | 0.100595 | 0.277557 |
| P20-N1 | 3 | 41 | 0.009870 | 0.042672 | 0.1879 | 0.065126 | 0.130252 |
| P20-N1 | 6 | 41 | 0.000000 | 0.052546 | 0.0000 | 0.000000 | 0.000000 |
| P30-N0 | 1 | 41 | 0.013340 | 0.040968 | 0.2456 | 0.089273 | 0.244623 |
| P30-N0 | 3 | 41 | 0.006285 | 0.046338 | 0.1194 | 0.053438 | 0.106876 |
| P30-N0 | 6 | 41 | 0.000000 | 0.057056 | 0.0000 | 0.000000 | 0.000000 |
| P50-N0 | 1 | 41 | 0.007837 | 0.049102 | 0.1376 | 0.068555 | 0.192391 |
| P50-N0 | 3 | 41 | 0.003261 | 0.057042 | 0.0541 | 0.037877 | 0.075755 |
| P50-N0 | 6 | 41 | 0.000000 | 0.062573 | 0.0000 | 0.000000 | 0.000000 |

## 2. Mean Dice & tail risk by (config, k)

| config | k | mean_dice | mean_worst_dice | min_worst_dice | p10_worst_dice | CVaR_0.1 | catastrophic_rate |
|--------|---|-----------|-----------------|----------------|----------------|----------|-------------------|
| P20-N1 | 1 | 0.397354 | 0.262775 | 0.000000 | 0.055420 | 0.041347 | 0.1626 |
| P20-N1 | 3 | 0.416641 | 0.351514 | 0.000542 | 0.051897 | 0.047074 | 0.1463 |
| P20-N1 | 6 | 0.448427 | 0.448427 | 0.052178 | 0.099372 | 0.071859 | 0.1220 |
| P30-N0 | 1 | 0.421681 | 0.303322 | 0.000380 | 0.053588 | 0.046793 | 0.1423 |
| P30-N0 | 3 | 0.450920 | 0.397481 | 0.039026 | 0.074526 | 0.057682 | 0.1098 |
| P30-N0 | 6 | 0.475608 | 0.475608 | 0.044442 | 0.105024 | 0.073764 | 0.0732 |
| P50-N0 | 1 | 0.435764 | 0.331010 | 0.006865 | 0.073341 | 0.055842 | 0.1545 |
| P50-N0 | 3 | 0.461503 | 0.423626 | 0.033672 | 0.082705 | 0.063134 | 0.1341 |
| P50-N0 | 6 | 0.483654 | 0.483654 | 0.037492 | 0.084736 | 0.068341 | 0.1220 |

## 3. Pairwise reference deviation (mean MAD)

| config | k | mean_mad | n_queries |
|--------|---|----------|-----------|
| P20-N1 | 1 | 0.126193 | 41 |
| P20-N1 | 3 | 0.130252 | 41 |
| P20-N1 | 6 | 0.000000 | 41 |
| P30-N0 | 1 | 0.113918 | 41 |
| P30-N0 | 3 | 0.106876 | 41 |
| P30-N0 | 6 | 0.000000 | 41 |
| P50-N0 | 1 | 0.086674 | 41 |
| P50-N0 | 3 | 0.075755 | 41 |
| P50-N0 | 6 | 0.000000 | 41 |

## 4. Prediction consistency (cross-set A vs B, validation)

| config | mean_prediction_overlap_AB | mean_single_prediction_overlap_AB | mean_prompt_overlap_AB | n_queries |
|--------|----------------------------|-----------------------------------|------------------------|-----------|
| P20-N1 | 0.518292 | 0.523542 | 0.269348 | 41 |
| P30-N0 | 0.607517 | 0.594181 | 0.296754 | 41 |
| P50-N0 | 0.711258 | 0.697748 | 0.341907 | 41 |

## 5. Key findings

- **Reference instability is real but modest**: at k=1, V_ref_frac ranges {'P20-N1': 0.3312, 'P30-N0': 0.2456, 'P50-N0': 0.1376}, max for P20-N1 (0.3312).
- **Subset-size trend**: V_ref(k=1) > V_ref(k=3) > V_ref(k=6) holds for P20-N1 (k=6 V_ref=0 by construction since one subset).
- **Consensus dampens instability**: k=6 has V_ref=0 (single subset); k=3 retains only A-vs-B disagreement.

## 6. Gate A — 7 conditions (§6.5)

| # | condition | pass |
|---|-----------|------|
| 1 | `cond1_corrected_metrics_significant` | PASS |
| 2 | `cond2_30pct_queries_gap_ge_005` | PASS |
| 3 | `cond3_mean_ref_std_ge_002` | PASS |
| 4 | `cond4_metric_definitions_match_code` | PASS |
| 5 | `cond5_subset_size_trend_explainable` | PASS |
| 6 | `cond6_stability_metric_linked_to_catastrophic` | PASS |
| 7 | `cond7_test_time_safe_feature_predicts_instability` | PASS |

**cond1** (`cond1_corrected_metrics_significant`): **PASS** — V_ref>0 and V_ref_frac>5% for >=1 config; per-config V_ref_frac={'P20-N1': 0.3312, 'P30-N0': 0.2456, 'P50-N0': 0.1376}

**cond2** (`cond2_30pct_queries_gap_ge_005`): **PASS** — fraction of val queries with best-worst gap>=0.05 per config={'P20-N1': 0.902, 'P30-N0': 0.829, 'P50-N0': 0.805}

**cond3** (`cond3_mean_ref_std_ge_002`): **PASS** — mean within-query ref std (k=1) per config={'P20-N1': 0.1006, 'P30-N0': 0.0893, 'P50-N0': 0.0686}

**cond4** (`cond4_metric_definitions_match_code`): **PASS** — Track A §6.3 audit verified code/metric consistency (variance_code_trace.md).

**cond5** (`cond5_subset_size_trend_explainable`): **PASS** — V_ref(k=1)>V_ref(k=3)>V_ref(k=6) for >=1 config; per-config={'P20-N1': (0.016721, 0.00987, 0.0), 'P30-N0': (0.01334, 0.006285, 0.0), 'P50-N0': (0.007837, 0.003261, 0.0)}

**cond6** (`cond6_stability_metric_linked_to_catastrophic`): **PASS** — Spearman(ref_std|gap, catastrophic) per config; best |rho|={'P20-N1': 0.031, 'P30-N0': 0.0192, 'P50-N0': -0.3953}

**cond7** (`cond7_test_time_safe_feature_predicts_instability`): **PASS** — best Spearman |rho| (TRAIN) = 0.8832 feature=prediction_jaccard_mean metric=std_dice config=P50-N0; threshold 0.25

## Final state

**PHASES_TRACK_A_REFERENCE_INSTABILITY_BENCHMARK_VALID**  (7/7 conditions passed)
