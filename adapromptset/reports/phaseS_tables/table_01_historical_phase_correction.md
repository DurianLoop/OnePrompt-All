# Table 01 — Historical Phase Correction (Phase R → Phase S)

**Status:** Phase R cond3 INVALID — corrected in Phase S Track A metric audit.
**Source:** `reports/phaseS_trackA_metric_audit/` (variance_code_trace.md, phaseR_var_single_definition.md, phaseR_var_consensus_definition.md, phaseR_interpretation_addendum.md).

## 1. The Phase R cond3 Error

Phase R cond3 ("variance reduction from consensus") compared two quantities that were computed on **different variance axes**:

| Quantity | Phase R definition | Variance axis |
|---|---|---|
| `var_single` | variance of single-reference predictions | **(query × reference) pair axis** = `V_total = V_query + V_ref` |
| `var_consensus` | variance of consensus (mean) predictions across reference subsets | **query axis only** = `V_query` (V_ref component removed by aggregation) |

The ratio `VR = 1 - var_consensus / var_single` therefore divided variance on different axes, producing an **apparent** reduction that mixed (a) removal of V_ref by aggregation and (b) cross-query variance. This is a definition error, not a measurement error.

## 2. Corrected Decomposition

| Term | Definition | Axis |
|---|---|---|
| `V_query` | Variance attributable to query difficulty | query |
| `V_ref`   | Variance attributable to reference choice (within a query) | reference (within query) |
| `V_total` | `V_query + V_ref` (single-reference predictions) | query × reference |
| `V_ref_frac` | `V_ref / V_total` | fraction |

**Corrected VR (consensus vs single-reference)**:

- By construction, consensus aggregation across k references reduces `V_ref` toward 0 as k grows.
- The "reduction" claimed by Phase R cond3 is **not a method gain** — it is the algebraic consequence of aggregating over the reference axis.
- Corrected `VR = 1.0` when comparing consensus (query-axis variance) to single-reference (query×ref-axis variance) on a within-query basis.

## 3. Cross-query vs Within-query Variance Distinction

| Axis | Question answered | Phase R status |
|---|---|---|
| Cross-query | "Do harder queries have more reference instability?" | Phase R implicitly used this in cond3 |
| Within-query | "How much does the prediction change across references for ONE query?" | Phase S Track A correct axis for V_ref |

Phase S Track A recomputes all reference-instability metrics on the **within-query reference axis**:
- `V_ref` is the variance across k references for a fixed query.
- `V_ref_frac = V_ref / (V_query + V_ref)` per query, then averaged.
- This isolates the *reference-induced* component from query difficulty.

## 4. §6.3 Audit Findings (Track A)

| # | Finding | Resolution |
|---|---|---|
| F1 | `var_single` computed on (query×ref) pair axis = `V_total` | Recompute as within-query reference variance |
| F2 | `var_consensus` computed on query axis only = `V_query` | Acknowledge as the query-axis component |
| F3 | cond3 compared axes that were not the same | Corrected VR = 1.0 by construction; cond3 retired |
| F4 | Matcher cost was ignored in Phase R runtime accounting | Phase S adds matcher cost to runtime |
| F5 | `B2_mean`'s `beta` / `tau` fields were cosmetic exporter artifact | Removed from metric computation; B2_mean is plain reference-mean consensus |

## 5. Implication for Phase S

- Track A benchmark is recomputed with the corrected axis.
- Phase R cond3 ("VR > 0.5") is **not** a valid gate; Track A uses V_ref_frac, mean_ref_std, best-worst gap, CVaR, CFR and prediction_consistency instead (Gate A, 7/7 PASSED).
- All numbers reported in Phase S supersede Phase R cond3 numbers.

## 6. Reference Artefacts

- `reports/phaseS_trackA_metric_audit/variance_code_trace.md`
- `reports/phaseS_trackA_metric_audit/phaseR_var_single_definition.md`
- `reports/phaseS_trackA_metric_audit/phaseR_var_consensus_definition.md`
- `reports/phaseS_trackA_metric_audit/phaseR_interpretation_addendum.md`
- `reports/phaseS_trackA_metric_audit/cross_query_vs_within_query_variance.csv`
- `reports/phaseS_trackA_metric_audit/corrected_reference_variance.csv`
- `reports/phaseS_trackA_metric_audit/corrected_pairwise_deviation.csv`
