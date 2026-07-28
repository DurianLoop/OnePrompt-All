# Gate AA1 — 41 M2 Validation Queries Eligibility Audit

**Date**: 2026-07-27
**Gate**: AA1

## Query Identity

The 41 M2 validation queries were defined in Phase T (WP3 M2 selective prediction). They are a subset of 121 MoNuSAC queries that were split into 80 training + 41 validation for M2 model training.

### Query List (41 total)
train_034, train_035, train_036, train_037, train_038, train_043, train_044, train_045, train_046, train_057, train_058, train_059, train_060, train_061, train_062, train_063, train_064, train_094, train_095, train_096, train_097, train_098, train_099, train_100, train_101, train_102, train_103, train_144, train_145, train_146, train_147, train_163, train_164, train_165, train_166, train_167, train_168, train_169, train_183, train_184, train_185

## Exposure Audit

### Direct Exposure

| Exposure Category | Status | Details |
|---|---|---|
| Development Cohort (80 queries) | **NOT IN** | These 41 queries are completely disjoint from the 80 Development Cohort queries |
| M2 Model Training | **NO** | Used as validation, not training |
| M2 Model Validation | **YES** | Used for M2-0/1/2/3/4 model selection in Phase T |
| Phase W Experiments | **NO** | Not used in WP0-WP7 |
| Phase V Paper v5 | **NO** | Not used in paper results |
| B4 Score Design | **NO** | B4 is parameter-free pairwise IoU |

### Indirect Exposure

| Exposure Category | Status | Details |
|---|---|---|
| Risk Feature Evaluation | **YES** | M2 models used features (entropy, disagreement, etc.) evaluated on these queries |
| Coverage Selection | **YES** | CFR@80% coverage was evaluated on these queries |
| Model Selection | **YES** | M2-2_logistic_regression selected as best model based on these queries (AUROC=0.944, CFR reduction=39.2%) |
| Paper Route Decision | **INDIRECT** | Phase T M2 results informed Route B decision, but the 41 queries were not directly used for route decisions |
| Qualitative Selection | **NO** | No evidence of qualitative selection using these queries |
| Result Visualization | **NO** | No evidence of these queries being used in visualizations |

## Decision

### PHASEAA_VALIDATION41_NOT_HOLDOUT

**Reasons**:
1. These 41 queries were used for M2 model selection in Phase T (M2-2_logistic_regression was selected based on performance on these queries)
2. They were used for coverage evaluation (CFR@80%)
3. They were used for feature evaluation (indirectly through M2 model training)
4. They were NOT used for B4 Prediction Consistency score design (B4 is parameter-free)
5. They were NOT used in Phase W or Phase V experiments

**Classification**: Option B per todo.md §5.2 — Validation Cohort, not holdout

### Consequence
- Cannot be used as Internal Confirmatory Cohort
- Must look for alternative fresh data: unobserved data, unobserved dataset split, or formal test
- Gate AA3 (confirmatory experiment) is BLOCKED unless alternative fresh data is found

## Alternative Fresh Data Options

### Option 1: MoNuSeg (37 train + 14 test)
- Status: External dataset, not used in Phase AA method design
- Risk: Different domain (organ-level vs cell-level), small sample size

### Option 2: IDRID (55 train + 27 test)
- Status: External dataset, retinal lesion segmentation
- Risk: IDRID mean consensus failed (ECE=0.477), different domain

### Option 3: GlaS (85 train + 60 testA + 20 testB)
- Status: External dataset, gland segmentation
- Risk: Different domain, small structures

### Option 4: Formal Test (locked)
- Status: LOCKED, cannot be used

## Recommendation
1. Proceed with Gate AA2 (Development Cohort 80-query grouped CV)
2. Skip Gate AA3 (no valid confirmatory cohort)
3. Use MoNuSeg and IDRID as external validation (Gate AA5)
4. Do not manufacture "confirmation" on the 80 Development Cohort queries