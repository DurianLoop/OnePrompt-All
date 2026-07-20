# Representation Pivot Analysis

Status: `PIVOT_REPRESENTATION_ANALYSIS`

| Item | Value |
|---|---:|
| Episodes | 10 |
| Actions | 12 |
| Best fixed action | `B_top5_pos3neg` |
| Mean regret to oracle | 0.005011 |
| Mean top-2 Dice margin | 0.020441 |
| Near-tie episode fraction | 0.400 |
| Oracle differs from best fixed fraction | 0.500 |
| Mean candidate point precision | 0.520000 |
| Mean candidate component coverage | 0.052005 |

Representation claim: `oracle_analysis_only_insufficient_separation`

Interpretation:

- This report is computed from existing MoNuSAC validation pilot artifacts only.
- It is a Gate-F-pivot analysis artifact, not a policy-training result.
- Small oracle margins and near-ties are treated as evidence against starting MoNuSAC policy training.

Allowed Next Actions:
- `fixed_rule_action_oracle_analysis`
- `representation_analysis_without_policy_training`
- `predeclared_new_domain_pilot`

Prohibited Next Actions:
- `monusac_policy_training`
- `posthoc_monusac_action_redesign`
- `test_time_action_search`
