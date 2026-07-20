# Table 1: Old 12-Action Exact Definition

| Action ID | Family | Valid | Groups | K+ | K- | SAM Calls | Point Budget | Positive Rule | Grouping |
|---|---|---|---|---|---|---|---|---|---|
| A_top1_one_group | single_group_positive | True | 1 | 1 | 0 | 1 | 1 | topk_by_contrast_score | G_ONE_GROUP |
| A_top3_one_group | single_group_positive | True | 1 | 3 | 0 | 1 | 3 | topk_by_contrast_score | G_ONE_GROUP |
| A_top5_one_group | single_group_positive | True | 1 | 5 | 0 | 1 | 5 | topk_by_contrast_score | G_ONE_GROUP |
| A_top10_one_group | single_group_positive | False | 0 | 0 | 0 | 0 | 0 | topk_by_contrast_score | G_ONE_GROUP |
| B_top3_pos1neg | single_group_positive_negative | True | 1 | 3 | 1 | 1 | 4 | topk_by_contrast_score | G_ONE_GROUP |
| B_top3_pos3neg | single_group_positive_negative | True | 1 | 3 | 3 | 1 | 6 | topk_by_contrast_score | G_ONE_GROUP |
| B_top5_pos1neg | single_group_positive_negative | True | 1 | 5 | 1 | 1 | 6 | topk_by_contrast_score | G_ONE_GROUP |
| B_top5_pos3neg | single_group_positive_negative | True | 1 | 5 | 3 | 1 | 8 | topk_by_contrast_score | G_ONE_GROUP |
| C_top3_independent | independent_positive_groups | True | 3 | 3 | 0 | 3 | 3 | topk_by_contrast_score | G_ONE_POINT_PER_GROUP |
| C_top5_independent | independent_positive_groups | True | 5 | 5 | 0 | 5 | 5 | topk_by_contrast_score | G_ONE_POINT_PER_GROUP |
| R_split5_2groups | rescue_split_groups | True | 2 | 5 | 0 | 2 | 5 | topk_split_or_local_neg | G_FIXED_CHUNK_3_2 |
| R_split5_2groups_localneg | rescue_group_specific_negatives | True | 2 | 5 | 2 | 2 | 7 | topk_split_or_local_neg | G_FIXED_CHUNK_OR_ONE_PER_GROUP_WITH_LOCAL_NEG |
| R_top5_independent_localneg | rescue_group_specific_negatives | True | 5 | 5 | 5 | 5 | 10 | topk_split_or_local_neg | G_FIXED_CHUNK_OR_ONE_PER_GROUP_WITH_LOCAL_NEG |
