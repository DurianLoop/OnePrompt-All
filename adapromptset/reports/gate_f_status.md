# Gate F Pilot Status

| Check | Status | Notes |
|---|---|---|
| action utility pilot | FAIL_ORACLE_GAP_SMALL | episodes=10 records=120 |
| oracle vs best fixed | FAIL_ORACLE_GAP_SMALL | oracle=0.3762 best_fixed=0.3712 gap=0.0050 |
| AJI oracle vs best fixed | AVAILABLE | oracle=0.23216115704701398 best_fixed=0.22834095079987268 gap=0.0038202062471413 |
| oracle action concentration | FAIL_ORACLE_GAP_SMALL | top_fraction=0.500; counts={'R_split5_2groups_localneg': 1, 'B_top3_pos3neg': 1, 'A_top5_one_group': 1, 'B_top5_pos3neg': 5, 'B_top5_pos1neg': 2} |
| policy decision | PIVOT_F_REQUIRED | rescue action bank already failed; do not start policy, pivot to fixed-rule/action-oracle analysis or a predeclared new-domain pilot |
