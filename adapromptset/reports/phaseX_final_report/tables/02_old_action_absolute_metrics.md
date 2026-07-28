# Table 2: Old Action Absolute Performance

| Rank | Action ID | Family | Mean Dice | Mean IoU | Mean AJI | Mean SAM Calls | Mean Runtime (ms) | Oracle Wins | Episodes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | B_top5_pos3neg | single_group_positive_negative | 0.3712 | 0.2283 | 0.2283 | 1.00 | 41.4 | 5 | 10 |
| 2 | B_top5_pos1neg | single_group_positive_negative | 0.3425 | 0.2080 | 0.2080 | 1.00 | 42.1 | 2 | 10 |
| 3 | B_top3_pos3neg | single_group_positive_negative | 0.3297 | 0.1982 | 0.1982 | 1.00 | 48.9 | 1 | 10 |
| 4 | A_top5_one_group | single_group_positive | 0.3260 | 0.1967 | 0.1967 | 1.00 | 49.0 | 1 | 10 |
| 5 | B_top3_pos1neg | single_group_positive_negative | 0.2975 | 0.1781 | 0.1781 | 1.00 | 40.8 | 0 | 10 |
| 6 | R_split5_2groups_localneg | rescue_group_specific_negatives | 0.2866 | 0.1716 | 0.1716 | 2.00 | 107.1 | 1 | 10 |
| 7 | R_split5_2groups | rescue_split_groups | 0.2588 | 0.1531 | 0.1531 | 2.00 | 91.1 | 0 | 10 |
| 8 | A_top3_one_group | single_group_positive | 0.2561 | 0.1524 | 0.1524 | 1.00 | 34.3 | 0 | 10 |
| 9 | R_top5_independent_localneg | rescue_group_specific_negatives | 0.1507 | 0.0880 | 0.0880 | 5.00 | 448.5 | 0 | 10 |
| 10 | C_top5_independent | independent_positive_groups | 0.1195 | 0.0691 | 0.0691 | 5.00 | 390.8 | 0 | 10 |
| 11 | C_top3_independent | independent_positive_groups | 0.0653 | 0.0361 | 0.0361 | 3.00 | 175.1 | 0 | 10 |
| 12 | A_top1_one_group | single_group_positive | 0.0292 | 0.0160 | 0.0160 | 1.00 | 61.9 | 0 | 10 |
