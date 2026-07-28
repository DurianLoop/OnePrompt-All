# GlaS Phase 2 Split Report

Status: `PASS_PHASE2_GLAS_SPLIT`
Protocol: `adapromptset.phase2.glas_strict.v2`
Seed: `20260719`

## Role Counts

- `locked_testA`: 60
- `locked_testB`: 20
- `strict_val`: 30
- `utility_train`: 55

## Episode Counts

- `train`: 165
- `val`: 90

## Boundary

- Formal test boundary: `locked_not_fully_blind_due_to_prior_manifest_profile_exposure`
- testA/testB locked cases are recorded but no Phase 2 test episodes are generated.
- Query-level aggregation is the Gate F unit; per-reference rows are retained for sensitivity.

## Artifacts

- `case_assignments`: `manifests/phase2/glas/case_assignments.csv`
- `episodes_train`: `manifests/phase2/glas/episodes_train.csv`
- `episodes_val`: `manifests/phase2/glas/episodes_val.csv`
- `locked_test_cases`: `manifests/phase2/glas/locked_test_cases.csv`
- `protocol_metadata`: `manifests/phase2/glas/protocol_metadata.json`
- `report_json`: `reports/phase2_glas/split/split_report.json`
- `report_md`: `reports/phase2_glas/split/split_report.md`
