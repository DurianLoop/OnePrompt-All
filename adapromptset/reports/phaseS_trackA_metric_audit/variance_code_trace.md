# Track A §6.3 — Variance Code Trace

Source file: `src/adapromptset/scripts/run_phaseR_gateR2.py`
Function: `compute_gate_r2(...)` (lines 1459-1597)
Condition audited: `cond3_variance_reduction_ge_25pct` (lines 1510-1518)

## 1. The exact lines that computed `var_single`, `var_consensus`, `vr`

```python
# cond3: variance reduction >= 25%
per_ref_dices = [
    _row_dice(r) for r in val_per_ref_rows
    if r["config_name"] == best_config and _row_dice(r) is not None
]
var_single = float(np.var(per_ref_dices)) if len(per_ref_dices) > 1 else 0.0
var_cons = float(np.var(cons_dices)) if len(cons_dices) > 1 else 0.0
vr = 1.0 - var_cons / var_single if var_single > 0 else 0.0
cond3 = vr >= 0.25
```

## 2. What each variable contains

### `per_ref_dices` (lines 1511-1514)
A flat list of **every** per-reference Dice value for `best_config` (P30-N0)
drawn from `val_per_reference_metrics.csv`. There are 41 queries × 6 references
= **246 values**. Each value is `D(q, r)` — the Dice score for query `q` when
prompted from reference `r`. This list is NOT grouped by query; it is a flat
pool that mixes variation across queries AND variation across references within
each query.

### `var_single` (line 1515)
`var_single = np.var(per_ref_dices)` — the population variance (`ddof=0`) of
the 246 pooled Dice values. By the law of total variance this equals:

    Var_{q,r}[D(q,r)] = Var_q[ E_r[D(q,r)] ] + E_q[ Var_r[D(q,r)] ]
                      = V_query            + V_ref

i.e. **total** variance = cross-query variance (case difficulty) + within-query
reference variance (the instability we actually want to quantify). See
`phaseR_var_single_definition.md`.

### `cons_dices` (line 1495)
`cons_dices = [cons_by_q[q] for q in common_q]` — one consensus Dice value per
query (41 values) for `best_method` (B2_mean) at `reference_scope=all6`. The
consensus step collapses the 6 per-reference predictions into a single
prediction per query, so the reference dimension is eliminated.

### `var_cons` / `var_consensus` (line 1516)
`var_cons = np.var(cons_dices)` — the population variance of the 41 consensus
Dice values. Because there is exactly one value per query, this is **pure
cross-query variance**: `Var_q[ D_consensus(q) ] = V_query` only. The
within-query reference variance is gone by construction (the consensus already
aggregated the 6 references). See `phaseR_var_consensus_definition.md`.

### `vr` (line 1517)
`vr = 1.0 - var_cons / var_single = 1.0 - V_query / (V_query + V_ref)`

## 3. The metric definition error

`cond3` compares two variances measured on **different axes**:

| quantity        | axis                        | decomposition |
|-----------------|-----------------------------|---------------|
| `var_single`    | Var over (query × ref) pairs| V_query + V_ref |
| `var_consensus` | Var over query only         | V_query        |

`vr = 1 - V_query / (V_query + V_ref)` is therefore **not** the reduction in
reference-induced variance. It is dominated by `V_query` (case difficulty),
which is intrinsic and is NOT reduced by consensus. With `V_ref` being a small
fraction of `V_total`, the ratio `V_query / (V_query + V_ref)` is close to 1,
so `vr` is close to 0 (here -0.0506), making cond3 impossible to satisfy even
though consensus eliminates 100% of the within-query reference variance.

## 4. The correct metric (directive §6.2)

    V_ref(single)    = (1/N) Σ_q Var_r[ D(q,r) ]        # the instability
    V_ref(consensus) = 0                                 # by construction
    VR_correct       = 1 - V_ref(consensus) / V_ref(single) = 1.0

The corrected variance reduction is **1.0 (100%)** for every config, because
consensus produces one prediction per query and therefore has zero
within-query reference variance by construction.

## 5. Numeric confirmation (from gateR2_summary.json)

    var_single        = 0.054308   (246 pooled P30-N0 Dice values)
    var_consensus     = 0.057056   (41 consensus P30-N0 Dice values)
    variance_reduction = -0.050601 (the reported, INCORRECT cond3 metric)

The corrected decomposition (computed in
`cross_query_vs_within_query_variance.csv`) shows that `V_ref` is only a small
fraction of `V_total`; the rest (`V_query`) is case difficulty that consensus
cannot and should not be expected to remove.
