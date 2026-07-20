# Gate G: Utility-Label Reliability

Status: `NOT_RUN_AFTER_GATE_F_PIVOT`
Pass/fail: `NOT_RUN`

## Rule Applied

Utility-label reliability is evaluated before policy training only when Gate F permits policy learning.

## Diagnostics

| Field | Value |
|---|---|
| phase5_status | `PIVOT_ORACLE_ANALYSIS` |

## Commands


## Input Hashes

- `reports/decision_checkpoint/phase5_decision.json`: `B6EC74EA3AF9984EA37F17EC38C35DDD67758F94D88E1FE9A1764143768FBE90`

## Allowed Next Actions

- `fixed_rule_action_oracle_analysis`
- `predeclared_new_domain_pilot`

## Prohibited Next Actions

- `policy_training_on_monusac`
- `posthoc_monusac_action_redesign`

## Deviations

- Not executed because Gate F selected the current oracle-analysis pivot.
