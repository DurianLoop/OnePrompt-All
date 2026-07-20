# Action-Space Saturation Protocol (Phase X, Sections 11-12)

Date: 2026-07-20

## Nested action banks (section 10)

A_small subset A_medium subset A_large subset A_search_augmented.

| Bank | max K+ | max K- | max calls | Notes |
|---|---|---|---|---|
| A_small | 5 | 4 | 3 | matches historical bank scale |
| A_medium | 20 | 10 | 5 | medium dense prompts |
| A_large | 100 | 30 | 20 | large dense prompts + dense grouping |
| A_search_augmented | >= A_large | >= A_large | >= A_large | + greedy/beam-discovered actions |

A_search_augmented entries must be discovered on the train-only audit subset,
abstracted into a query-GT-free deterministic rule, frozen before validation,
and registered as a single allowed action-bank redesign. GT-guided search
results cannot be silently promoted.

Each action carries a unique `action_id`, `action_hash`, `selection_rule`,
`K+`, `K-`, `grouping`, `cluster_count`, `call_budget`, `point_budget`, and
`fusion`.

## Fixed baseline saturation (section 11)

For each nested bank A_j, compute

```text
D_fixed(A_j) = max over a in A_j of  (1/N) sum_q D(q, a)
```

Report per bank:

- strongest fixed action and exact config
- mean / median Dice
- IoU, AJI
- component recall
- calls, runtime
- win/tie/loss vs strongest fixed of A_small

Plots:

- best fixed Dice vs action-bank size
- best fixed Dice vs max positive K
- best fixed Dice vs point budget
- best fixed Dice vs call budget
- best fixed component recall vs max positive K

Saturation question: does best fixed Dice keep rising from K+ = 5 to K+ = 100?
If the argmax shifts from low K to high K, the historical baseline was
inadequate.

## Finite-bank oracle saturation (section 12)

For each nested bank A_j, compute

```text
D_bank_oracle(A_j) = (1/N) sum_q max over a in A_j of D(q, a)
Delta_bank(A_j) = D_bank_oracle(A_j) - D_fixed(A_j)
```

Plots:

- bank oracle Dice vs action-bank size
- fixed Dice vs action-bank size
- oracle gap vs action-bank size
- oracle gap vs point budget
- oracle gap vs call budget
- oracle action distribution vs target count

## Interpretation (section 12)

| Pattern | Code |
|---|---|
| Fixed and Oracle both keep rising | `ACTION_SPACE_NOT_SATURATED` |
| Oracle rises, Fixed saturates | `ADAPTIVE_HETEROGENEITY_EMERGES` |
| Fixed and Oracle rise together, gap small | `STRONG_FIXED_DENSE_PROMPT_SUFFICIENT` |
| Both saturated, gap small | `ACTION_SPACE_SATURATED_ORACLE_GAP_SMALL` |

Only the last state justifies rejecting adaptive value under the current
budget.

## Absolute performance gate (section 15)

Reports must always include absolute Dice, not just gaps. Joint judgement uses
Dice + component recall + catastrophic failure + merge rate, relative to the
strongest simple baseline.
