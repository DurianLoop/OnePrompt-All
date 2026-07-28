# Historical Oracle Correction

Date: 2026-07-20

## What the historical "Oracle" actually was

The MoNuSAC Phase 5 / Gate F "Oracle" was a **Finite-Bank Oracle**: for each
query, the highest-Dice action among the 12 valid actions in
`configs/actions/prompt_set_bank_v2.json` was selected.

It was *not*:

- A Global Prompt Oracle over all possible positive/negative counts, subsets,
  groupings, and clustering algorithms.
- A search oracle.
- A privileged-coordinate oracle.

## Renaming

The historical claim

> Oracle gap ≈ 0.005

is renamed to

```text
FINITE_BANK_ORACLE_GAP_SMALL
```

It is prohibited to continue abbreviating this as "Oracle gap proves adaptive
prompting is ineffective".

## What the small gap actually implies

Given the historical anchors:

- mean oracle Dice = 0.376250
- mean best-fixed Dice = 0.371238
- gap = 0.005011
- strongest fixed action = `B_top5_pos3neg`
- bank max K+ = 5, max K- = 3, max calls = 5
- bank never tested K+ in {10, 20, 50, 75, 100}
- bank never tested K- in {10, 20, 30, 50}
- bank never tested dense kmeans/grid/graph/component groupings
- bank never tested call budgets above 5

the small gap only implies:

> Within the current 12 historical actions, per-query action selection beats the
> strongest fixed historical action by a small margin.

It does *not* imply:

- Larger positive counts are ineffective.
- More negatives are ineffective.
- Denser spatial coverage is ineffective.
- Top-50 / Top-100 actions are ineffective.
- Other groupings, clusterings, or multi-call strategies are ineffective.
- Adaptive prompt allocation has no value in a broader action space.

## Cross-references

- `reports/phaseX_historical_bank_audit/historical_claim_correction.md`
- `reports/phaseX_historical_bank_audit/old_bank_oracle_summary.md`
- `docs/action_space_problem_statement.md`
