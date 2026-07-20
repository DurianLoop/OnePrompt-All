# New-Domain Pilot Readiness

Status: `READY_WITH_BLOCKED_FORMAL_DATASETS`

This audit is read-only: it does not reopen MoNuSAC policy training and does not train a new policy.

## Dataset Decisions

| Dataset | Status | Formal policy training | Key next action |
|---|---|---:|---|
| GlaS | `BLOCKED_INSUFFICIENT_EPISODES` | false | `predeclare_glas_trainval_diagnostic_without_official_test` |
| DDR | `READY_FOR_INFRASTRUCTURE_ONLY` | false | `ddr_stage1_union_loader_smoke` |
| IDRID | `READ_ONLY_NON_TUNABLE` | false | `historical_negative_result_summary` |
| CPM-17 | `DEFERRED_MISSING_OR_INCOMPLETE` | false | `write_missing_asset_report` |

## Gate Supplements

### GlaS
- `predeclared_trainval_diagnostic` when `episode_count_below_gate_minimum`: Use source-train data only; preserve official testA/testB; report whether enough query-level units exist before any policy work.

### DDR
- `stage1_infrastructure_smoke_only` when `pathology_gate_f_failure_may_be_domain_shape_specific`: Exercise loaders, mask union, coordinate transforms, and candidate diagnostics without policy labels.

### IDRID
- `none` when `fresh_unobserved_subset_cannot_be_proven`: Keep IDRID as read-only historical or boundary evidence.

### CPM-17
- `missing_asset_report` when `complete_public_version_missing`: Record expected source, license, 32/32 split assumptions, and parser requirements.

## Global Prohibited Next Actions

- `monusac_policy_training`
- `posthoc_monusac_action_redesign`
- `test_time_action_search`
- `promote_insufficient_episode_smoke_to_gate_pass`
- `formal_new_domain_policy_training_without_predeclared_gate_f`