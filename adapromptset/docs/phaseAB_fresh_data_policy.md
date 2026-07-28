# Phase AB Fresh Data Policy

## Current State

PHASEAB_BLOCKED_NO_FRESH_CONFIRMATORY_DATA

No legal confirmatory cohort exists. All available data has been observed:

- MoNuSAC 80 queries: Development Cohort (used for training, feature selection, model selection)
- MoNuSAC 41 queries: Validation Cohort (used for M2 model selection, coverage evaluation)
- MoNuSeg 30 queries: External dataset (observed in Phase T)
- IDRID 30 queries: External dataset (observed in Phase T)

## Freshness Criteria

A query is "fresh" only if ALL of:
- Never used for training
- Never used for feature selection
- Never used for threshold selection
- Never used for qualitative selection
- Never used for model/route decisions
- Never viewed in any Phase report
- Reference pool disjoint from query patient/case
- GT not read before protocol freeze

## Consequence

Without fresh data:
- Cannot write "confirmatory" or "holdout-confirmed"
- Paper is exploratory/benchmark only
- Target venue expectations must be adjusted
- External datasets serve as secondary validation, not confirmation

## User Action Required

To proceed with confirmation, the user must provide:
1. Dataset name and source
2. Download link
3. License information
4. Expected sample size
5. Whether this is formal test data