# Table 06 — Pairwise Reference Deviation (MAD_ref)

**Goal:** Quantify the mean absolute Dice difference between reference pairs for each query. Unlike `mean_ref_std` (which uses a variance statistic), MAD_ref is in **Dice units** and is directly interpretable as "how much the answer changes if you swap references."

## 1. Definition

For a query q with k references, sample all `C(k, 2)` pairs (r_i, r_j):

```
MAD_ref(q) = mean_{(i,j)} | Dice(C_{q,r_i}) - Dice(C_{q,r_j}) |
```

Reported as the mean over queries.

## 2. Per (Config, k) MAD_ref

| Config | k | MAD_ref (mean) | MAD_ref (median) | 90th pct |
|---|---|---|---|---|
| P20-N1 | 1 | 0.1421 | 0.1318 | 0.2366 |
| P20-N1 | 3 | 0.1311 | 0.1217 | 0.2188 |
| P20-N1 | 6 | 0.1177 | 0.1094 | 0.1977 |
| P30-N0 | 1 | 0.1262 | 0.1166 | 0.2104 |
| P30-N0 | 3 | 0.1161 | 0.1077 | 0.1941 |
| P30-N0 | 6 | 0.1039 | 0.0961 | 0.1728 |
| P50-N0 | 1 | 0.0972 | 0.0899 | 0.1621 |
| P50-N0 | 3 | 0.0895 | 0.0828 | 0.1492 |
| P50-N0 | 6 | 0.0801 | 0.0741 | 0.1335 |

(Per-query values in `reports/phaseS_trackA_benchmark/pairwise_reference_deviation.csv`.)

## 3. Track B / Track C Comparisons (Mean Pairwise Deviation)

| Setting | Mean pairwise dev | Reduction vs uniform |
|---|---|---|
| Track B uniform (P30-N0, k=6) | 0.1139 | — (baseline) |
| Track B weighted (M4_gbm) | 0.0455 | **60.09% reduction** (PASSED ≥10% threshold, cond3) |

**Note:** Track B reduced pairwise deviation substantially (60%) but did NOT reduce mean within-query ref std (only 2.83% reduction, FAILED cond2) and did NOT improve Dice (FAILED cond1). The pairwise-deviation reduction is therefore a "compression without accuracy gain" effect — predictions get more similar across references but not better on average.

## 4. Track C Contribution

Track C methods (C3 agreement gate, C4 stable candidate scoring) did NOT further reduce MAD_ref below the B2_mean baseline at the best hyperparameters (α=0, β=0, γ=0) — i.e. the best Track C config **is** B2_mean. Non-zero α/β/γ **increased** MAD_ref on average.

## 5. Interpretation

- MAD_ref of 0.10–0.14 (P30-N0 k=1) means that **on average, swapping references changes the Dice by ~0.1–0.14**. This is large relative to the mean Dice (~0.47).
- The reduction from k=1 to k=6 (0.126 → 0.104 in P30-N0) is the natural effect of averaging — it is **not** a method gain.
- Track B's 60% reduction in pairwise dev is a real statistical effect but does not translate into Dice gain (the consensus is more uniform across references but anchored at the same mean).

## 6. Reference

- `reports/phaseS_trackA_benchmark/pairwise_reference_deviation.csv`
- `reports/phaseS_trackA_metric_audit/corrected_pairwise_deviation.csv`
- `reports/phaseS_trackB/val_weighted_consensus_metrics.csv`
- `reports/phaseS_trackC/val_trackC_consensus_metrics.csv`
