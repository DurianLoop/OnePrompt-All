# Phase X: Nested Action Bank Saturation Summary

Date: 2026-07-20
Stage: nested_bank_saturation (Gate X4)
Episodes: 10
Positive limit: 100
Device: cpu
SAM inference run: True
Elapsed seconds: 647.258
GPU-hours: 0.0

## Bank constraints (frozen at preregistration)

| Bank | Max K+ | Max K- | Max calls |
|---|---|---|---|
| A_small | 5 | 4 | 3 |
| A_medium | 20 | 10 | 5 |
| A_large | 100 | 30 | 20 |

## Gate X4 Status

**OLD_ACTION_BANK_INADEQUATE**

A_large best_fixed=0.5411 vs A_small best_fixed=0.3713; gain=0.1699 >= 0.05

## Saturation curve

| Bank | Actions | Episodes | Best fixed action | Best fixed Dice | Oracle Dice | Oracle gap | Best fixed calls | Oracle calls | Win/Tie/Loss |
|---|---|---|---|---|---|---|---|---|---|
| A_small | 12 | 10 | A_small__G_ONE_GROUP__K5__N3__N_BOTTOMK | 0.3713 | 0.3842 | 0.0129 | 1 | 1.2 | 5/5/0 |
| A_medium | 45 | 10 | A_medium__G_ONE_GROUP__K20__N1__N_BOTTOMK | 0.5411 | 0.5668 | 0.0257 | 1 | 1.0 | 8/2/0 |
| A_large | 91 | 10 | A_large__G_ONE_GROUP__K20__N1__N_BOTTOMK | 0.5411 | 0.6119 | 0.0708 | 1 | 1.5 | 8/2/0 |

## Interpretation

- `OLD_ACTION_BANK_INADEQUATE`: A_large best fixed substantially beats A_small best fixed (>= 0.05 Dice).
- `ACTION_SPACE_SATURATED`: A_large oracle gap < 0.005.
- `ACTION_SPACE_NOT_SATURATED`: A_large oracle gap >= 0.005.
- `ACTION_SPACE_INSUFFICIENT_DATA`: missing bank data.
