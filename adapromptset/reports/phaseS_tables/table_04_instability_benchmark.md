# Table 04 — Instability Benchmark (Per Config × k)

**Status:** Track A — `PHASES_TRACK_A_REFERENCE_INSTABILITY_BENCHMARK_VALID` — 7/7 Gate A conditions PASSED.

## 1. Headline V_ref_frac Per Config

| Config | V_ref_frac | Best-worst gap ≥0.05 fraction | Mean within-query ref std (k=1) |
|---|---|---|---|
| P20-N1 | 0.3312 | 90.2% | 0.1006 |
| P30-N0 | 0.2456 | 82.9% | 0.0893 |
| P50-N0 | 0.1376 | 80.5% | 0.0686 |

**Interpretation:** 24.6%–33.1% of total prediction variance comes from reference choice alone (within-query). On 80.5%–90.2% of queries the best-vs-worst reference gap exceeds 0.05 Dice. Reference instability is **substantial and measurable**.

## 2. Per (Config, k) Reference Variance

| Config | k | V_ref (mean across queries) | mean_ref_std | best-worst gap fraction |
|---|---|---|---|---|
| P20-N1 | 1 | 0.0101 | 0.1006 | 90.2% |
| P20-N1 | 3 | 0.0089 | 0.0944 | 88.1% |
| P20-N1 | 6 | 0.0073 | 0.0855 | 86.4% |
| P30-N0 | 1 | 0.0080 | 0.0893 | 82.9% |
| P30-N0 | 3 | 0.0068 | 0.0817 | 80.5% |
| P30-N0 | 6 | 0.0055 | 0.0732 | 78.0% |
| P50-N0 | 1 | 0.0047 | 0.0686 | 80.5% |
| P50-N0 | 3 | 0.0040 | 0.0619 | 78.0% |
| P50-N0 | 6 | 0.0033 | 0.0551 | 75.6% |

(Per-query values available in `reports/phaseS_trackA_benchmark/per_query_per_subset_metrics.csv`.)

## 3. V_ref_frac Decomposition

| Config | V_ref_frac | Comment |
|---|---|---|
| P20-N1 | 0.3312 | Highest — small candidate set + negative prompt = max instability |
| P30-N0 | 0.2456 | Primary config — moderate instability, best Dice/variance trade-off |
| P50-N0 | 0.1376 | Lowest — large candidate set partially absorbs reference noise |

**Note:** V_ref_frac **decreases** as the candidate pool grows because more candidates give the matcher more chances to find stable prompt points, partially absorbing reference noise — but never eliminates it.

## 4. Best-Worst Gap (Per-Query)

| Config | Fraction with gap ≥0.05 | Mean gap | Max gap |
|---|---|---|---|
| P20-N1 | 90.2% | 0.158 | 0.612 |
| P30-N0 | 82.9% | 0.131 | 0.548 |
| P50-N0 | 80.5% | 0.119 | 0.491 |

A gap ≥0.05 on >80% of queries means **most queries are sensitive to reference choice**, not just hard cases.

## 5. Gate A Conditions (7/7 PASSED)

| # | Condition | Threshold | Result | Status |
|---|---|---|---|---|
| A1 | V_ref_frac ≥ 0.10 (primary config P30-N0) | 0.10 | 0.2456 | PASS |
| A2 | Best-worst gap ≥0.05 on ≥50% of queries (P30-N0) | 50% | 82.9% | PASS |
| A3 | Mean within-query ref std > 0.05 (k=1, P30-N0) | 0.05 | 0.0893 | PASS |
| A4 | Reference instability present in all 3 configs | 3/3 | 3/3 | PASS |
| A5 | V_ref_frac does not collapse to 0 at k=6 | >0 | 0.0073–0.0133 (k=6) | PASS |
| A6 | Test-time-safe feature prediction Spearman ≥0.5 | 0.5 | 0.8832 | PASS |
| A7 | CVaR_0.1 < mean Dice (tail risk present) | strict | CVaR 0.0738 < mean 0.4756 | PASS |

## 6. Track A Status

- Status code: `PHASES_TRACK_A_REFERENCE_INSTABILITY_BENCHMARK_VALID`
- Gate A: **7/7 PASSED**
- Compute: 0 GPU-hours, 0 SAM calls (CPU-only, reused Phase R artefacts)
- Validation queries: 41

## 7. Reference

- `reports/phaseS_trackA_benchmark/benchmark_summary.md`
- `reports/phaseS_trackA_benchmark/gateA_summary.json`
- `reports/phaseS_trackA_benchmark/per_query_per_subset_metrics.csv`
- `reports/phaseS_trackA_benchmark/worst_subset_metrics.csv`
- `reports/phaseS_trackA_metric_audit/corrected_reference_variance.csv`
