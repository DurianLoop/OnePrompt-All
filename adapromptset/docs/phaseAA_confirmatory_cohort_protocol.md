# Phase AA Confirmatory Cohort Protocol

**Date**: 2026-07-27

## Eligibility Criteria

The 41 M2 validation queries are eligible as confirmatory cohort ONLY if ALL of the following are true:

1. Never used for risk feature selection
2. Never used for B4 prediction-consistency score design
3. Never used for coverage threshold selection
4. Never used for CFR threshold selection
5. Never used for model selection
6. Never used for paper route decision
7. Never used for qualitative selection
8. Never used for any result visualization

## If Eligible

- Freeze as "Internal Confirmatory Cohort"
- One-shot evaluation only
- All method parameters frozen before evaluation

## If Not Eligible

- Define as "Validation Cohort" (not holdout)
- Seek alternative fresh data
- If no fresh data exists, stop confirmatory experiments