# Table 13: Absolute Dice and Component Recall (Summary Across Stages)

| Stage | Configuration | Mean Dice | Mean Component Recall | Episodes |
|---|---|---|---|---|
| Gate X1 (Historical bank) | best fixed = B_top5_pos3neg | 0.3712 | N/A | 10 |
| Gate X3 (K+ scaling) | best K+ = 20 | 0.5237 | 0.9678 | 10 |
| Negative scaling (K+ = 20) | K- = 1, sel = N_BOTTOMK | 0.5411 | 0.9777 | 10 |
| Gate X4 (A_small best_fixed) | A_small__G_ONE_GROUP__K5__N3__N_BOTTOMK | 0.3713 | N/A | 10 |
| Gate X4 (A_small oracle) | per-episode oracle | 0.3842 | N/A | 10 |
| Gate X4 (A_medium best_fixed) | A_medium__G_ONE_GROUP__K20__N1__N_BOTTOMK | 0.5411 | N/A | 10 |
| Gate X4 (A_medium oracle) | per-episode oracle | 0.5668 | N/A | 10 |
| Gate X4 (A_large best_fixed) | A_large__G_ONE_GROUP__K20__N1__N_BOTTOMK | 0.5411 | N/A | 10 |
| Gate X4 (A_large oracle) | per-episode oracle | 0.6119 | N/A | 10 |
| Gate X5 (Search oracle) | greedy add/remove/swap + add-neg | 0.7256 | 0.9425 | 10 |
