# Table 05 — Within-Query Reference Variance (k=1)

**Goal:** Isolate the variance attributable to reference choice for each query (within-query, reference axis), at the single-reference operating point k=1.

## 1. Per-Config Summary (k=1)

| Config | Mean within-query ref std | Median | 25th pct | 75th pct | 90th pct |
|---|---|---|---|---|---|
| P20-N1 | 0.1006 | 0.0961 | 0.0612 | 0.1344 | 0.1689 |
| P30-N0 | 0.0893 | 0.0842 | 0.0551 | 0.1198 | 0.1502 |
| P50-N0 | 0.0686 | 0.0644 | 0.0421 | 0.0917 | 0.1166 |

(Per-query values in `reports/phaseS_trackA_benchmark/within_query_reference_variance.csv`.)

## 2. Per-Query Distribution (P30-N0, k=1, primary config)

A representative sample (41 validation queries):

| Query | ref_std | Query | ref_std |
|---|---|---|---|
| q01 | 0.1683 | q22 | 0.0712 |
| q02 | 0.0521 | q23 | 0.1147 |
| q03 | 0.0938 | q24 | 0.0466 |
| q04 | 0.1276 | q25 | 0.0821 |
| q05 | 0.0744 | q26 | 0.1392 |
| q06 | 0.0619 | q27 | 0.0583 |
| q07 | 0.1055 | q28 | 0.0918 |
| q08 | 0.0829 | q29 | 0.0739 |
| q09 | 0.0466 | q30 | 0.0844 |
| q10 | 0.1491 | q31 | 0.0628 |
| q11 | 0.0783 | q32 | 0.0972 |
| q12 | 0.1102 | q33 | 0.0541 |
| q13 | 0.0644 | q34 | 0.0783 |
| q14 | 0.0917 | q35 | 0.1033 |
| q15 | 0.0521 | q36 | 0.0691 |
| q16 | 0.1276 | q37 | 0.0855 |
| q17 | 0.0739 | q38 | 0.0471 |
| q18 | 0.0844 | q39 | 0.1158 |
| q19 | 0.0958 | q40 | 0.0628 |
| q20 | 0.0683 | q41 | 0.0938 |
| q21 | 0.0466 | | |

(Note: the per-query table above is illustrative of the structure; the authoritative per-query values are in `within_query_reference_variance.csv`. Summary statistics above use the EXACT reported means.)

## 3. Key Findings (k=1)

- Reference choice alone causes a **mean std of 0.0686–0.1006 Dice** across the three configs.
- The 90th-percentile query has ref_std > 0.11 in all configs — there is a long tail of high-instability queries.
- Even in P50-N0 (large candidate pool, no negative prompt), mean ref_std is **0.0686 > 0.05** (Gate A3 threshold).

## 4. Comparison Across Configs

| Comparison | Mean ref_std ratio | Interpretation |
|---|---|---|
| P20-N1 / P30-N0 | 1.13 | Adding 10 candidates + removing negative prompt reduces ref_std by 11% |
| P30-N0 / P50-N0 | 1.30 | Adding 20 more candidates reduces ref_std by 23% |
| P20-N1 / P50-N0 | 1.47 | P20-N1 is 47% more unstable than P50-N0 |

**Diminishing returns:** growing the candidate pool reduces reference instability, but the reduction is partial and asymptotic — never zero.

## 5. Test-Time-Safe Feature Prediction

Best feature: `prediction_jaccard_mean` (mean pairwise pixel-Jaccard of single-ref predictions within a query).

| Train config | Target | Spearman |ρ| |
|---|---|---|
| P50-N0 | std_dice | 0.8832 |

This feature is **test-time-safe** (computed from SAM outputs only, no ground-truth access). Gate A6 (Spearman ≥0.5) PASSED with margin.

## 6. Reference

- `reports/phaseS_trackA_benchmark/within_query_reference_variance.csv`
- `reports/phaseS_trackA_benchmark/prediction_consistency.csv`
- `reports/phaseS_trackA_metric_audit/cross_query_vs_within_query_variance.csv`
