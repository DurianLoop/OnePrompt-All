# Phase S Metric Definitions

Date: 2026-07-22
Branch: `phaseS-multitrack-reference-robust-research`

All metric definitions are frozen before any Phase S inference. Track A §6.3
may correct Phase R's variance code trace; the corrected definitions below are
the authoritative Phase S reference.

## 1. Cross-Query Variance (NOT reference instability)

```
V_query = Var_q[ D(q) ]
```

where `D(q)` is the mean Dice of query `q` across reference subsets.

This measures case-difficulty variation, NOT reference-induced instability.
Reporting V_query as "reference variance" is prohibited.

## 2. Within-Query Reference Variance (THE reference instability metric)

```
V_ref = (1/N) Σ_q Var_S[ D(q, S) ]
```

where `S` indexes different reference subsets (or individual references).

Phase R computed this as `var_single` = Var across 6 references of Dice for a
single (query, seed, config). Phase S Track A will verify the computation axis
and extend to subset-level variance.

## 3. Pairwise Reference Deviation

```
MAD_ref = (1/N) Σ_q E_{S_i, S_j}[ |D(q, S_i) - D(q, S_j)| ]
```

Robust to outliers; complements V_ref.

## 4. Prediction Consistency

```
J_pred = E_{S_i, S_j}[ IoU( M̂(q, S_i), M̂(q, S_j) ) ]
```

Measures prediction-mask overlap across reference subsets, independent of GT.

## 5. Tail Risk Metrics

- **Worst-subset Dice**: min over reference subsets of D(q, S).
- **10% CVaR**: mean Dice of the worst 10% (query, subset) pairs.
- **Catastrophic failure rate (CFR)**: fraction of (query, subset) pairs with
  Dice < 0.1.
- **Worst-reference gap**: max over references of |D(q, R_r) - mean_q D|.

## 6. Consensus Comparison Metrics

For consensus method `m` vs random single `B1`:
- **Δ Dice**: mean(D_m) - mean(D_B1).
- **Variance reduction**: 1 - V_ref(m) / V_ref(B1).
- **CFR change**: CFR(m) - CFR(B1).
- **A/B prediction overlap**: IoU of predictions using reference set A vs set B.
- **Bootstrap 95% CI**: paired bootstrap over queries, 10000 resamples.

## 7. Reliability Prediction Metrics (Track B)

- **Spearman ρ**: rank correlation between predicted utility `û(q, R_r)` and
  actual utility `u(q, R_r)` on validation.
- **Weight collapse**: max weight share `max_r w_r / Σ w_r` > 0.95 indicates
  collapse.
- **Weight floor**: `w_r ≥ ε` enforced post-softmax.

## 8. Risk-Aware Utility (Track B §7.5)

```
U_robust = E_S[ D(q, S) ] - λ * Std_S[ D(q, S) ]
```

with `λ ∈ {0, 0.25, 0.5, 1.0}`. Optional CVaR variant uses α-quantile.

## 9. Reference Count Scaling

For `k ∈ {1, 3, 6}` references per subset:
- Mean Dice vs k.
- Within-query std vs k.
- Worst-subset Dice vs k.
- Cost (SAM calls, runtime) vs k.

## 10. Cost / Runtime

- SAM calls per prediction (must be 1 for B/C, may be 0 for D's matching-only
  baselines).
- Wall-clock runtime per query (ms).
- GPU-hours cumulative.

## Statistical Reporting

For every comparison, report:
- mean, median, paired difference
- bootstrap 95% CI
- win/tie/loss counts
- effect size (Cohen's d or rank-biserial)
- seed mean/std (across resampling seeds)
- within-query variance, CVaR, worst-reference, failure rate, runtime, cost

Multi-model comparisons use Holm correction.

## Statistical Units

- Query / case is the independent unit.
- Reference episodes are repeated measurements, NOT independent samples.
- Points and patches are not statistical units.
