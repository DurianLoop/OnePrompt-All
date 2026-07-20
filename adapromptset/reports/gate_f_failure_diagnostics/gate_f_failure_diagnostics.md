# Gate F Failure Diagnostics

Policy training remains prohibited for the current MoNuSAC evidence.

| Diagnostic | Status | Key value |
|---|---|---:|
| Candidate coverage | `LOW_COMPONENT_COVERAGE` | 0.052004904966278354 |
| Proposal separation | `ACTIONS_SEPARATED` | 0.020441 top-2 Dice margin |
| SAM call/group sensitivity | `NO_HIGHER_CALL_GAIN` | -0.020570 high-call delta |

Interpretation flags:

- `candidate_pool_limited_coverage`
- `no_clear_sam_call_grouping_gain`

Allowed next actions:

- fixed-rule/action-oracle analysis
- representation analysis
- separately predeclared new-domain pilot

Prohibited next actions:

- MoNuSAC policy training
- post-hoc MoNuSAC action-bank redesign
- test-time action search
