# Phase Y — Test Lock Notice

Date: 2026-07-21

Directive section: 4.3.

## 1. Locked Datasets

The following formal test sets remain LOCKED for the duration of Phase Y:

| Dataset | Split | Cases | Status |
|---|---|---:|---|
| MoNuSAC | test | 101 | LOCKED_UNUSED |
| GlaS | testA | 60 | LOCKED_UNUSED |
| GlaS | testB | 20 | LOCKED_UNUSED |
| DDR | test | 225 | LOCKED_UNUSED |
| IDRID | test | — | LOCKED_UNUSED |

## 2. Prohibited Uses of Test

No formal test access for:

- K+ selection
- K- selection
- action bank design
- router / policy training
- feature engineering
- model capacity selection
- seed selection
- threshold tuning
- early stopping
- calibration
- claim reporting (until test-freeze protocol)

## 3. Test Access Conditions

Test may be accessed ONLY after ALL of:

1. Phase Y gates Y1-Y4 are complete.
2. A single frozen best model is selected on validation.
3. Holdout has been run once.
4. A test-freeze protocol is written and approved.
5. The user explicitly approves test access.

## 4. Violation Handling

Any test access before the above conditions triggers:

- `TEST_LOCK_VIOLATION`
- immediate halt of all Phase Y activity
- the violating result is discarded
- the violation is logged in `docs/decision_log.md`
- the protocol must be re-preregistered before resuming

## 5. Test Lock Audit

The test lock is verified by:

- `reports/phaseY_data_audit/case_identity_map.csv` — all test cases have
  `case_role = test` and are NOT present in any Phase Y manifest.
- `manifests/phaseY/policy_train_queries.csv` — train cases only.
- `manifests/phaseY/policy_validation_queries.csv` — train cases only.
- `manifests/phaseY/confirmatory_holdout_queries.csv` — train cases only.
- `manifests/phaseY/reference_pool.csv` — train cases only.

No test case_id appears in any Phase Y manifest.

## 6. Status

```text
TEST_LOCKED_UNUSED
```

This notice is frozen at the preregistration commit.
