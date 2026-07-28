# Phase X Stage S4: Negative Scaling Summary

Date: 2026-07-20
Stage: negative_scaling (Gate X4 preliminary, negative axis)
Episodes: 10
Positive limit: 100
K+ (fixed): 20
Grouping (fixed): G_ONE_GROUP
Device: cpu
SAM inference run: True
Elapsed seconds: 62.166
GPU-hours: 0.0

## Negative selectors (frozen)

- Available: N_NONE, N_BOTTOMK, N_HIGH_BACKGROUND
- Deferred (require candidate generation logic): N_BOUNDARY_HARD, N_LOCAL_CONFUSION, N_DIVERSITY

## Gate X4 (preliminary, negative axis only)

**NEGATIVE_BEST_BEATS_NONE**

best K-=1 selector=N_BOTTOMK mean_dice=0.5411 vs N_NONE baseline mean_dice=0.5237; gain=0.0174 >= 0.010

## Negative scaling curve

| K- | Selector | Episodes | Mean Dice | Mean IoU | Mean AJI | Mean component recall | Mean calls | Mean runtime (ms) | Actual negatives |
|---|---|---|---|---|---|---|---|---|---|
| 0 | N_NONE | 10 | 0.5237 | 0.3642 | 0.3642 | 0.9678 | 1.0 | 45.5 | 0.0 |
| 1 | N_NONE | 10 | 0.5237 | 0.3642 | 0.3642 | 0.9678 | 1.0 | 45.2 | 0.0 |
| 3 | N_NONE | 10 | 0.5237 | 0.3642 | 0.3642 | 0.9678 | 1.0 | 44.9 | 0.0 |
| 5 | N_NONE | 10 | 0.5237 | 0.3642 | 0.3642 | 0.9678 | 1.0 | 42.4 | 0.0 |
| 10 | N_NONE | 10 | 0.5237 | 0.3642 | 0.3642 | 0.9678 | 1.0 | 41.3 | 0.0 |
| 20 | N_NONE | 10 | 0.5237 | 0.3642 | 0.3642 | 0.9678 | 1.0 | 44.9 | 0.0 |
| 1 | N_BOTTOMK | 10 | 0.5411 | 0.3846 | 0.3846 | 0.9777 | 1.0 | 42.4 | 1.0 |
| 3 | N_BOTTOMK | 10 | 0.4814 | 0.3215 | 0.3215 | 0.9893 | 1.0 | 44.2 | 3.0 |
| 5 | N_BOTTOMK | 10 | 0.4592 | 0.3013 | 0.3013 | 0.9918 | 1.0 | 44.6 | 5.0 |
| 10 | N_BOTTOMK | 10 | 0.4281 | 0.2757 | 0.2757 | 0.9940 | 1.0 | 44.3 | 10.0 |
| 20 | N_BOTTOMK | 10 | 0.4308 | 0.2795 | 0.2795 | 0.9979 | 1.0 | 47.1 | 20.0 |
| 1 | N_HIGH_BACKGROUND | 10 | 0.5062 | 0.3476 | 0.3476 | 0.9893 | 1.0 | 44.3 | 1.0 |
| 3 | N_HIGH_BACKGROUND | 10 | 0.4632 | 0.3052 | 0.3052 | 0.9817 | 1.0 | 47.9 | 3.0 |
| 5 | N_HIGH_BACKGROUND | 10 | 0.4511 | 0.2944 | 0.2944 | 0.9906 | 1.0 | 44.6 | 5.0 |
| 10 | N_HIGH_BACKGROUND | 10 | 0.4329 | 0.2833 | 0.2833 | 0.9971 | 1.0 | 46.9 | 10.0 |
| 20 | N_HIGH_BACKGROUND | 10 | 0.4144 | 0.2651 | 0.2651 | 1.0000 | 1.0 | 48.9 | 20.0 |

## Interpretation

- `NEGATIVE_BEST_BEATS_NONE`: a (K-, selector) combo beats N_NONE by >= 0.01 Dice.
- `NEGATIVE_NONE_DOMINATES`: N_NONE baseline is within 0.01 Dice of the best negative config.
- `NEGATIVE_INSUFFICIENT_DATA`: missing N_NONE baseline or no valid K->0 data.

Note: this is a *preliminary* read on the negative axis only. The full Gate X4
verdict requires nested action bank saturation (section 11) and search-oracle
adequacy (section 13).
