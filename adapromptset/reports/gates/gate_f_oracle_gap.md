# Gate F: Oracle Gap

Status: `FAIL_ORACLE_GAP_SMALL`
Pass/fail: `FAIL`

## Rule Applied

Policy learning may continue only if the predeclared oracle gap and action-heterogeneity criteria pass after at most one rescue bank.

## Diagnostics

| Field | Value |
|---|---|
| episode_count | `10` |
| record_count | `120` |
| mean_oracle_dice | `0.376249802278007` |
| mean_best_fixed_dice | `0.37123847211635835` |
| dice_oracle_gap | `0.0050113301616486194` |
| aji_available | `True` |
| mean_oracle_aji | `0.23216115704701398` |
| mean_best_fixed_aji | `0.22834095079987268` |
| aji_oracle_gap | `0.0038202062471413` |
| gate_f_min_dice_gap | `0.01` |
| top_oracle_action_fraction | `0.5` |
| best_fixed_action | `B_top5_pos3neg` |
| phase5_status | `PIVOT_ORACLE_ANALYSIS` |

## Commands

- `scripts/run_oracle_gap_pilot.py --action-bank configs/actions/prompt_set_bank_v2.json`
- `scripts/write_decision_checkpoint.py`

## Input Hashes

- `reports/oracle_gap_pilot_v2/oracle_gap_pilot_summary.json`: `F4CBE717CF9CA94755DF6CFC85F7623E47CD0FE4FB425F46A6A8D5A82EF92FF5`
- `reports/decision_checkpoint/phase5_decision.json`: `B6EC74EA3AF9984EA37F17EC38C35DDD67758F94D88E1FE9A1764143768FBE90`
- `reports/gate_f_status.md`: `B706348FF5191D8221C1EE864B55AF0101030E7F9833A9C7A870801EE399AC6E`

## Allowed Next Actions

- `fixed_rule_action_oracle_analysis`
- `predeclared_new_domain_pilot`

## Prohibited Next Actions

- `policy_training_on_monusac`
- `posthoc_monusac_action_redesign`

## Deviations

- Gate F failed after the one allowed MoNuSAC rescue bank; MoNuSAC policy training remains prohibited.
