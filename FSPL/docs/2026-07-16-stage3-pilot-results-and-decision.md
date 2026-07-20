# Stage 3 Single-Reference Pilot: Result and Decision Record

## Protocol completed

- Dataset: official MoNuSeg, 37 train and 14 held-out test cases.
- Seed: `20260716`.
- Reference: `TCGA-21-5784-01Z-00-DX1` from the training split.
- Policy train/validation/test queries: 30 / 6 / 14.
- Actions: `top1_n0`, `top3_n0`, `top5_n0`, `top5_n3`.
- Every train/validation action utility was generated offline from SAM Dice;
  the test masks were excluded from fitting and model selection.

## Evidence

| Method | Validation selected Dice | Official-test Dice |
|---|---:|---:|
| Fixed `top5_n0` | — | 0.3533 |
| Fixed `top5_n3` | — | 0.3830 |
| Ridge utility policy | 0.4420 | 0.3555 |
| ExtraTrees utility policy | 0.4484 | 0.3197 |

The fixed-rule test results and all policy artifacts are stored under
`artifacts/stage2_fixed_baselines/` and `artifacts/stage3_utility_policy/`.

## Diagnosis

The single-reference policy is **not a publishable positive result**:

- the oracle action is `top5_n3` for 23/30 training, 4/6 validation and 13/14
  test queries, so the action distribution is highly imbalanced;
- test matching-feature distance to the training set (2.0705 standardized
  nearest-neighbour distance) is not worse than validation (2.1095), which
  rules out a simple feature-distribution shift explanation;
- six validation queries are too few to select a stable high-capacity ranker.

## Design decision

Do not tune on MoNuSeg test.  The next method iteration uses many training
reference/query episodes, preserving a separate validation reference/query
set.  The larger locally available MoNuSAC split (209 train / 101 test) is the
appropriate next training ground for that protocol.
