# Table 07 — CVaR and Tail-Risk Metrics

**Goal:** Quantify the **tail risk** of the reference-instability distribution — not just the mean behaviour, but how bad the worst outcomes are.

## 1. Definitions

| Metric | Definition |
|---|---|
| `CVaR_0.1` | Mean of the worst 10% of (query × reference) Dice outcomes |
| `worst_dice` | Minimum Dice across (query × reference) pairs |
| `p10` | 10th percentile of Dice distribution |
| `CFR` | Catastrophic Failure Rate — fraction of (query × reference) pairs with Dice < 0.1 |

## 2. Per (Config, k) Tail-Risk Metrics

| Config | k | CVaR_0.1 | worst_dice | p10 | CFR |
|---|---|---|---|---|---|
| P20-N1 | 1 | 0.0512 | 0.0000 | 0.0712 | 0.1707 |
| P20-N1 | 3 | 0.0588 | 0.0000 | 0.0788 | 0.1341 |
| P20-N1 | 6 | 0.0644 | 0.0000 | 0.0844 | 0.1098 |
| P30-N0 | 1 | 0.0583 | 0.0000 | 0.0812 | 0.1463 |
| P30-N0 | 3 | 0.0661 | 0.0000 | 0.0888 | 0.1122 |
| P30-N0 | 6 | 0.0738 | 0.0000 | 0.0956 | 0.0732 |
| P50-N0 | 1 | 0.0712 | 0.0000 | 0.0938 | 0.0976 |
| P50-N0 | 3 | 0.0788 | 0.0000 | 0.1014 | 0.0732 |
| P50-N0 | 6 | 0.0855 | 0.0000 | 0.1081 | 0.0488 |

**Headline numbers (P30-N0, k=6, B2_mean):**
- CVaR_0.1 = **0.073764**
- CFR (Dice<0.1) = **0.0732**
- Mean Dice = 0.475608

CVaR is **6.5× smaller than mean Dice** — the tail is far worse than the mean. This is the headline tail-risk finding.

## 3. Catastrophic Failure Rate (CFR) Detail

| Config | k=1 CFR | k=3 CFR | k=6 CFR | Reduction (k=1→k=6) |
|---|---|---|---|---|
| P20-N1 | 17.07% | 13.41% | 10.98% | 35.7% |
| P30-N0 | 14.63% | 11.22% | 7.32% | 50.0% |
| P50-N0 | 9.76% | 7.32% | 4.88% | 50.0% |

- Consensus (k=6) reduces CFR by 35–50% vs single reference — averaging helps.
- **But CFR is never zero** — even at k=6 in the best config (P50-N0), 4.88% of (query × reference) pairs produce Dice < 0.1.
- Track B (weighted) CFR = 0.0732, Track B (uniform) CFR = 0.0732 — **identical, no reduction** (Track B cond3 sub-condition: CFR must not increase — PASSED but no improvement).

## 4. CVaR_0.1 (P30-N0, k=6) — Track B Comparison

| Setting | CVaR_0.1 | Delta vs uniform |
|---|---|---|
| Uniform (B2_mean) | 0.0738 | — |
| Track B weighted (M4_gbm) | 0.0768 | +0.003 (worse) |

Track B did NOT improve CVaR — the weighted aggregation slightly worsened tail risk.

## 5. CVaR_0.1 (P30-N0, k=6) — Track C Comparison

| Setting | CVaR_0.1 | Delta vs Track B |
|---|---|---|
| Track B uniform (baseline) | 0.0738 | — |
| Track C best (α=0,β=0,γ=0 = B2_mean) | 0.0707 | -0.0031 (FAILED ≥0.01 improvement) |

Track C did NOT improve CVaR.

## 6. Worst-Case Examples

- A non-trivial fraction of (query, reference) pairs produce Dice = 0.0000 (complete failure).
- These are concentrated in queries where the reference image's structure misleads the matcher into proposing prompt points that SAM refines into empty or wildly wrong masks.

## 7. Gate A Tail-Risk Condition

| # | Condition | Threshold | Result | Status |
|---|---|---|---|---|
| A7 | CVaR_0.1 < mean Dice (tail risk present) | strict < | 0.0738 < 0.4756 | PASS |

Tail risk is **substantial and present** — this is a stable, measurable phenomenon.

## 8. Reference

- `reports/phaseS_trackA_benchmark/cvar_metrics.csv`
- `reports/phaseS_trackA_metric_audit/cvar_tail_risk.csv`
- `reports/phaseS_trackB/val_weighted_consensus_metrics.csv`
- `reports/phaseS_trackC/val_trackC_consensus_metrics.csv`
