# Table 09 — Reference-Count Scaling

**Goal:** Quantify how V_ref and Dice change as the number of references k grows from 1 to 6.

## 1. V_ref(k) Trend (Mean Across Queries)

| Config | V_ref(k=1) | V_ref(k=3) | V_ref(k=6) | Reduction (1→6) |
|---|---|---|---|---|
| P20-N1 | 0.0101 | 0.0089 | 0.0073 | 27.7% |
| P30-N0 | 0.0080 | 0.0068 | 0.0055 | 31.3% |
| P50-N0 | 0.0047 | 0.0040 | 0.0033 | 29.8% |

## 2. mean_ref_std(k) Trend

| Config | std(k=1) | std(k=3) | std(k=6) |
|---|---|---|---|
| P20-N1 | 0.1006 | 0.0944 | 0.0855 |
| P30-N0 | 0.0893 | 0.0817 | 0.0732 |
| P50-N0 | 0.0686 | 0.0619 | 0.0551 |

## 3. Dice(k) Trend (B2_mean consensus)

| Config | Dice(k=1) | Dice(k=3) | Dice(k=6) | Gain (1→6) |
|---|---|---|---|---|
| P20-N1 | 0.3912 | 0.4288 | 0.4521 | +0.0609 |
| P30-N0 | 0.4188 | 0.4521 | 0.4756 | +0.0568 |
| P50-N0 | 0.4402 | 0.4683 | 0.4891 | +0.0489 |

## 4. V_ref_frac(k) Trend

| Config | V_ref_frac(k=1) | V_ref_frac(k=3) | V_ref_frac(k=6) |
|---|---|---|---|
| P20-N1 | 0.3312 | 0.2811 | 0.2455 |
| P30-N0 | 0.2456 | 0.2066 | 0.1733 |
| P50-N0 | 0.1376 | 0.1144 | 0.0955 |

## 5. Key Findings

- **V_ref decreases** with k (consensus reduces reference variance) — expected algebraic effect.
- **Dice increases** with k (averaging helps) — also expected.
- **V_ref_frac decreases** with k but never reaches 0 — even at k=6, reference choice still contributes 9.6%–24.6% of total variance.
- **Diminishing returns:** the gain from k=3→k=6 is smaller than from k=1→k=3 (the curve is concave).

## 6. Per-Query Dice Curve (P30-N0, illustrative)

| Query | Dice(k=1) | Dice(k=3) | Dice(k=6) |
|---|---|---|---|
| q01 | 0.231 | 0.301 | 0.348 |
| q02 | 0.512 | 0.548 | 0.571 |
| q03 | 0.388 | 0.421 | 0.442 |
| ... | ... | ... | ... |
| Mean | 0.4188 | 0.4521 | 0.4756 |

(Authoritative values in `reports/phaseS_trackA_benchmark/reference_count_scaling.csv`.)

## 7. CVaR and CFR vs k (P30-N0)

| k | CVaR_0.1 | CFR |
|---|---|---|
| 1 | 0.0583 | 0.1463 |
| 3 | 0.0661 | 0.1122 |
| 6 | 0.0738 | 0.0732 |

- CVaR increases with k (the worst 10% get less bad).
- CFR decreases with k (fewer catastrophic failures).
- Both improvements are the natural effect of averaging.

## 8. Gate A5 — V_ref_frac Does Not Collapse at k=6

| Condition | Threshold | Result | Status |
|---|---|---|---|
| A5 | V_ref_frac > 0 at k=6 (all configs) | >0 | 0.0955–0.2455 | PASS |

Reference instability is **not removed** by consensus at k=6 — it is reduced but persists.

## 9. Reference

- `reports/phaseS_trackA_benchmark/reference_count_scaling.csv`
- `reports/phaseS_trackA_benchmark/per_query_per_subset_metrics.csv`
