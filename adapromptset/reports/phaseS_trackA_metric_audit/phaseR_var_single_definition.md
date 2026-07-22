# Phase R `var_single` definition

## Definition

`var_single` was computed in `run_phaseR_gateR2.py` line 1515 as:

```python
per_ref_dices = [_row_dice(r) for r in val_per_ref_rows
                 if r["config_name"] == best_config and _row_dice(r) is not None]
var_single = float(np.var(per_ref_dices))
```

For `best_config = P30-N0`, `per_ref_dices` is the flat list of all 41 × 6 = 246
per-reference Dice values `D(q, r)` drawn from `val_per_reference_metrics.csv`.
`np.var` uses population variance (`ddof=0`).

Therefore:

    var_single = Var_{q,r}[ D(q, r) ]    (total variance over all query×ref pairs)

## Law of total variance decomposition

With equal reference count N=6 per query:

    Var_{q,r}[D] = Var_q[ E_r[D(q,r)] ] + E_q[ Var_r[D(q,r)] ]
                   = V_query            + V_ref

- `V_query = Var_q[ E_r[D(q,r)] ]` — variance across queries of the per-query
  mean Dice. This is **case difficulty** spread (some queries are simply harder).
- `V_ref = (1/N) Σ_q Var_r[D(q,r)]` — average within-query variance across the 6
  references. This is the **reference-induced instability** that Gate R1
  confirmed and that consensus is designed to eliminate.

## Key point

`var_single` is **NOT** within-query reference variance. It is the *total*
variance, of which reference-induced variance is only a small fraction. Using
`var_single` as the "single-reference variance" baseline against
`var_consensus` therefore conflates case difficulty with reference instability.

## Numeric values (validation, P30-N0)

| quantity  | value     |
|-----------|-----------|
| V_total (= var_single) | 0.054308 |
| V_query                | 0.040968 |
| V_ref                  | 0.013340 |
| V_ref_frac             | 0.245635 |

Reported `var_single` in `gateR2_summary.json` = 0.054308, which matches
`V_total` for P30-N0 (rounding aside), confirming the trace.
