# Phase 5 Decision Checkpoint

Status: `PIVOT_ORACLE_ANALYSIS`

Rationale: no meaningful oracle gap after the one allowed redesign; analyze fixed-rule near-oracle behavior and action-bank limits.

| Item | Value |
|---|---|
| v1 Gate F | FAIL_ORACLE_GAP_SMALL |
| v1 Dice oracle gap | 0.004667 |
| v2 Gate F | FAIL_ORACLE_GAP_SMALL |
| v2 Dice oracle gap | 0.005011 |
| Gate F minimum Dice gap | 0.010 |
| v2 best fixed action | B_top5_pos3neg |
| v2 top oracle action fraction | 0.500 |

Allowed next actions:

- `fixed_rule_action_oracle_analysis`
- `predeclared_new_domain_pilot`

Prohibited next actions:

- `policy_training_on_monusac`
- `posthoc_monusac_action_redesign`
