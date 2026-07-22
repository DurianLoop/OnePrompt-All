# Phase R `var_consensus` definition

## Definition

`var_consensus` was computed in `run_phaseR_gateR2.py` line 1516 as:

```python
cons_dices = [cons_by_q[q] for q in common_q]      # line 1495
var_cons   = float(np.var(cons_dices))               # line 1516
```

`cons_by_q` maps each query to a single consensus Dice value for `best_method`
(B2_mean) at `reference_scope = all6`. There is exactly **one** consensus Dice
per query (41 values for P30-N0).

Therefore:

    var_consensus = Var_q[ D_consensus(q) ]    (variance across queries only)

## Why this is pure cross-query variance

The consensus step (B2_mean) aggregates the 6 per-reference SAM predictions
into a single prediction per query. After aggregation the reference dimension
no longer exists: for each query there is one Dice value, not six. Any
within-query reference variance (`V_ref`) has been eliminated by construction.

Applying the law of total variance to the consensus output:

    Var_q[D_consensus] = V_query + V_ref(consensus)
                       = V_query + 0
                       = V_query

So `var_consensus` measures **only** `V_query` — the cross-query case-difficulty
spread. It contains **no** reference-induced component.

## The apples-to-oranges comparison

| quantity        | axis                 | decomposition |
|-----------------|----------------------|---------------|
| `var_single`    | (query × ref) pairs  | V_query + V_ref |
| `var_consensus` | query only           | V_query          |

Because the two quantities are on different axes, the ratio
`var_consensus / var_single = V_query / (V_query + V_ref)` is close to 1
(V_query dominates), producing `vr ≈ 0` (here -0.0506 for P30-N0).

## Numeric value (validation, P30-N0, B2_mean, all6)

| quantity        | value     |
|-----------------|-----------|
| var_consensus (reported) | 0.057056 |

This matches `gateR2_summary.json` (`var_consensus = 0.057056`), confirming the
trace. Note `var_consensus (0.057056) > var_single (0.054308)` is itself a red
flag: consensus variance should never exceed total single-reference variance if
the two were on the same axis. The fact that it does confirms they are on
different axes.
