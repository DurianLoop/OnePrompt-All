# Oracle Analysis Paper Assets

Status: `oracle_analysis_only`

These assets are generated from real Gate-F pivot artifacts. They do not contain simulated results and do not claim adaptive-policy success.

| Item | Value |
|---|---|
| Decision | `PIVOT_ORACLE_ANALYSIS` |
| Best fixed action | `B_top5_pos3neg` |
| Mean best fixed Dice | 0.371238 |
| Mean oracle Dice | 0.376250 |
| Oracle gap | 0.005011 |
| AJI oracle gap | 0.003820 |
| Allowed claim | `oracle_analysis_only` |
| Policy training | `prohibited` |
| Representation pivot | `PIVOT_REPRESENTATION_ANALYSIS` |
| Representation claim | `oracle_analysis_only_insufficient_separation` |
| Mean top-2 Dice margin | 0.020441 |
| Near-tie episode fraction | 0.400000 |
| Candidate point precision | 0.520000 |
| Candidate component coverage | 0.052005 |
| Prohibited after representation pivot | `monusac_policy_training` |
| Prohibited after representation pivot | `posthoc_monusac_action_redesign` |
| Prohibited after representation pivot | `test_time_action_search` |

Generated files:
- `macros`: `D:\SAM\adapromptset\paper\generated\oracle_analysis_macros.tex`
- `pivot_table`: `D:\SAM\adapromptset\paper\tables\table_oracle_pivot.tex`
- `action_table`: `D:\SAM\adapromptset\paper\tables\table_action_summary.tex`
- `report`: `D:\SAM\adapromptset\reports\oracle_analysis_paper_assets.md`
- `representation_table`: `D:\SAM\adapromptset\paper\tables\table_representation_pivot.tex`

Claim boundary:

- MoNuSAC policy training remains prohibited after the failed Gate F.
- Post-hoc MoNuSAC action-bank redesign remains prohibited.
- GlaS evidence is a separate extension smoke unless a later predeclared protocol provides enough episodes.
