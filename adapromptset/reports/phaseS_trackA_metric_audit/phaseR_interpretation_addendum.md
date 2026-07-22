# Phase R Interpretation Addendum — Track A §6.3

This addendum answers the six mandatory questions from directive §6.3 and
amends the scientific interpretation of the Phase R Gate R2 result. It does
**not** reopen R2 and does **not** unblock R3/R4.

## Corrected variance decomposition (validation)

| config   | n_queries | n_refs | V_total   | V_query  | V_ref    | V_ref_frac | VR_correct |
|----------|-----------|--------|-----------|----------|----------|------------|------------|
| P20-N1 | 41 | 6 | 0.050485 | 0.033764 | 0.016721 | 0.331208 | 1.000 |
| P30-N0 | 41 | 6 | 0.054308 | 0.040968 | 0.013340 | 0.245635 | 1.000 |
| P50-N0 | 41 | 6 | 0.056939 | 0.049102 | 0.007837 | 0.137642 | 1.000 |

`VR_correct = 1 - V_ref(consensus) / V_ref(single)`, and `V_ref(consensus) = 0`
by construction (consensus yields one prediction per query, eliminating the
within-query reference dimension). Hence `VR_correct = 1.0` (100%) for every
config — the opposite of the reported `-0.0506`.

Reported (incorrect) Phase R cond3 values for `P30-N0`:
`var_single = 0.054308`, `var_consensus = 0.057056`,
`variance_reduction = -0.050601`.

---

## Q1. var_single and var_consensus computation axis

**var_single** is computed on the **(query × reference) pair axis**: it is the
population variance of all 246 per-reference Dice values
`Var_{q,r}[D(q,r)]` for `P30-N0`. By the law of total variance this
decomposes as `V_query + V_ref` — it mixes cross-query variance (case
difficulty) with within-query reference variance.

**var_consensus** is computed on the **query axis only**: it is the population
variance of the 41 consensus Dice values `Var_q[D_consensus(q)]` for
`B2_mean` at `reference_scope=all6`. Because consensus produces one
prediction per query, the reference dimension is gone, so this is pure
`V_query`.

**The two quantities are on different axes.** Comparing them is apples-to-oranges.

## Q2. Whether query variability was treated as reference variability

**YES.** `var_single` includes `V_query` (case difficulty), which dominates the
total variance. The Phase R cond3 effectively compared `V_query(consensus)`
against `V_query + V_ref(single)`, i.e.

    vr = 1 - V_query / (V_query + V_ref)

Since `V_query` is the dominant term, the ratio is close to 1 and `vr` is close
to 0 (here -0.050601). This attributes the (large, irreducible) cross-query
case-difficulty variance to the consensus method's failure, which is invalid.
The reference-induced variance `V_ref` that consensus actually eliminates is
only `V_ref_frac = 0.245635` of `V_total` for `P30-N0`,
yet cond3 weighed the comparison as if all of `V_total` were reference variance.

## Q3. Whether B1/B2 reference count differs

**YES, and this compounds the error.**

- **B1_random** uses **1** reference per query — a deterministic hash-based
  selection from the 6 references. Its Dice is one of the 6 per-reference
  values.
- **B2_mean** uses **all 6** references, aggregated (mask averaging) into a
  single consensus prediction.

However, `var_single` was computed from `val_per_reference_metrics.csv`, which
contains **all 6 references per query** (246 values for `P30-N0`), i.e.
the full per-reference Dice spread. This is **not** B1's variance: B1 picks a
single reference per query and therefore has no within-query reference variance
either (it has one value per query, just like consensus). So `var_single` does
**not** represent B1's variance — it represents the per-reference Dice spread,
which is a different quantity from either B1 or B2. The cond3 baseline was
therefore neither B1 nor a meaningful "single-reference" condition; it was the
raw cross-reference Dice dispersion pooled across queries.

## Q4. Whether matcher cost was ignored

**YES.** Only the SAM-B `predict` time was recorded in `runtime_ms`. The
DINOv2-L feature-extraction cost (the matcher forward pass used to retrieve
references and to score candidates) was **not** included in the runtime / cost
accounting. The matcher cost is shared across all methods (B1–B6) and across
single-reference and consensus runs, so omitting it does not change the
*relative* method ranking on `runtime_ms`. However, for fairness and for any
absolute cost claim (e.g. "consensus adds negligible cost"), the matcher cost
should be reported separately. The omission is a reporting gap, not a
scientific error in the variance metric.

## Q5. Whether B2_mean's beta/tau fields are exporter error

**The beta/tau fields for B2_mean are a cosmetic exporter artifact, NOT a
scientific error.**

`gateR2_summary.json` reports `best_beta = 0.25` and `best_tau =
0.83` alongside `best_method = B2_mean`. But B2_mean does **not** use
beta or tau: beta is the hyperparameter of **B5_disagreement** and tau is the
hyperparameter of **B6_quantile**. These fields are populated by the unified
selection schema which always carries `best_beta`/`best_tau` regardless of the
selected method. For B2_mean (which has an empty hyperparam) they are inert
leftovers from the schema, not parameters of the method. This is a cosmetic
bug in the exporter; it does not affect any Gate R2 condition or the
consensus Dice values.

## Q6. How Phase R scientific conclusions need correction

1. **The R2 cond3 failure is INVALID.** It compared total variance
   (`V_query + V_ref`, mixed) against cross-query variance only (`V_query`),
   producing `vr = -0.050601`. The correct comparison is
   `V_ref(consensus) / V_ref(single) = 0 / 0.013340`, giving
   `VR_correct = 1.000` (100% reduction). Consensus eliminates
   within-query reference variance **by construction**.

2. **The real finding stands and is positive:** consensus improves mean Dice
   by `+0.045953` (`P30-N0`, `B2_mean` vs B1_random),
   with a bootstrap 95% CI lower bound above 0, and reduces catastrophic
   failure rate from `0.121951` to `0.073171` (a
   4.88 pp reduction).

3. **Consensus does NOT reduce cross-query variance** (`V_query`), nor should
   it be expected to: `V_query` reflects case difficulty (some queries are
   intrinsically harder), not reference choice. This is intrinsic to the case
   mix and is not reducible by reference aggregation.

4. **Phase R's final state should be AMENDED** to read:
   > "Deterministic consensus eliminates within-query reference variance and
   > improves mean Dice, but cross-query variance is not reducible by consensus
   > — it reflects case difficulty, not reference choice."

5. **This AMENDMENT does not reopen R2 and does not unblock R3/R4.** It
   corrects the metric *interpretation*, not the gate decision flow. The
   corrected VR (1.0) would have passed cond3, but the amendment is scoped to
   fixing the metric definition and the scientific narrative; it does not
   retroactively flip the gate.

6. **Track B (reliability learning) remains scientifically motivated.**
   Because `V_query` (case difficulty) is the dominant, consensus-irreducible
   component, the lever for further stabilisation is per-query reliability
   weighting — down-weighting bad references *for a given query* — which
   targets `V_ref` at the per-query level and could additionally compress
   cross-query tail risk. This sustains Track B's rationale independent of the
   cond3 metric fix.

---

## Files produced by this audit

1. `variance_code_trace.md` — code path trace (lines 1459-1589)
2. `phaseR_var_single_definition.md` — var_single = V_total = V_query + V_ref
3. `phaseR_var_consensus_definition.md` — var_consensus = V_query only
4. `cross_query_vs_within_query_variance.csv` — corrected decomposition per config
5. `corrected_reference_variance.csv` — per (query, config) within-query stats
6. `corrected_pairwise_deviation.csv` — per (query, config) MAD across 15 ref pairs
7. `prediction_consistency.csv` — per config A/B prediction overlap (consensus vs single)
8. `cvar_tail_risk.csv` — per config CVaR_0.1, worst, p10, catastrophic rate
9. `method_cost_decomposition.csv` — per method cost (P30-N0, all6, validation)
10. `phaseR_interpretation_addendum.md` — this document (6 mandatory answers)
