# Phase Y — Statistics Protocol

Date: 2026-07-21

Directive section: 13, 14.

## 1. Statistical Unit

The statistical unit is the **query** (case). Reference episodes are NOT
independent samples. Multiple references per query are aggregated WITHIN the
query (mean over the 3 references in a set) before the query-level statistic
is computed.

## 2. Primary Metric

Mean Dice over query-level aggregated Dice. The strongest-fixed baseline is
the validation-selected `Top20 + N1 N_BOTTOMK + G_ONE_GROUP + 1 call`.

## 3. Bootstrap

- Method: paired patient-cluster bootstrap
- Iterations: 10,000
- CI level: 95%
- Unit: query (case), with patient as the cluster
- Paired: (policy - baseline) per query

### 3.1 Procedure

1. For each bootstrap iteration, sample patients with replacement.
2. Collect all queries from sampled patients.
3. Compute mean Delta Dice (policy - baseline).
4. Repeat 10,000 times.
5. 95% CI = 2.5th and 97.5th percentiles.
6. Pass condition: CI lower bound > 0.

## 4. Permutation Test

- Paired sign-flip permutation
- Iterations: 10,000
- Null: Delta Dice symmetric around 0
- Reported alongside bootstrap

## 5. Multiple Testing

When comparing multiple models / baselines on the same validation set:

- Benjamini-Hochberg FDR at 0.05
- Family: all learned models vs strongest fixed
- Holdout: single primary comparison (no correction needed)

## 6. Effect-Size Reporting

Every comparison reports:

- mean Delta Dice
- bootstrap 95% CI
- Cohen's d (paired)
- win / tie / loss counts (threshold 0.001 Dice)
- paired permutation p-value

## 7. Oracle Optimism

```text
Optimism = raw_oracle - cross_reference_oracle
Optimism_ratio = Optimism / raw_oracle_gap
```

Reported for both bank oracle (Y1) and search oracle (Y2).

## 8. Action Stability

- exact action agreement rate (A vs B per query)
- K+ bin agreement rate
- K- agreement rate
- selector agreement rate
- Spearman rho distribution (per-query action-rank correlation)
- top-3 action overlap distribution

## 9. Gate Decision Rules (Frozen)

| Gate | Metric | Threshold | CI |
|---|---|---|---|
| Y1 | stable oracle gap | >= 0.015 | lower > 0 |
| Y2 | cross-ref search gap | >= 0.03 | lower > 0 |
| Y3 | bank policy Delta | >= 0.010 | lower > 0 |
| Y4 | point-subset Delta | >= 0.015 OR >= 0.0075 over bank | lower > 0 |

Y3 also requires `oracle recovery >= 30%`.
Y1 also requires `optimism <= 60%`, `>= 30% queries differ`, `Spearman > 0.30`.
Y2 also requires `optimism <= 50%`, `median |K_A-K_B| <= 8`, `>= 70% win`.

## 10. Pre-Registration Lock

All thresholds, CI levels, and decision rules are frozen at the
preregistration commit. No post-hoc threshold adjustment. If a gate fails,
the corresponding fail-state is recorded and the protocol's routing logic
(section 15) determines the next allowed step.

## 11. Historical 10 Episodes

Phase X 10 episodes are labeled `EXPLORATORY_PHASEX_EVIDENCE` and are
excluded from all confirmatory statistics. They may appear only in a
historical appendix for sanity check.

## 12. Holdout

- Single run of the frozen best model.
- No model selection on holdout.
- No threshold adjustment on holdout.
- Holdout CI is reported but the pass/fail decision is based on the
  pre-registered validation thresholds. Holdout is confirmatory, not
  exploratory.
