# Phase AB Confirmatory Protocol

## Status

BLOCKED — No fresh confirmatory cohort available.

## Primary Hypothesis

H0: AURC_PC >= AURC_Random
H1: AURC_PC < AURC_Random

Test: query-level paired bootstrap with 10,000 resamples
Alpha: 0.05 (one-sided)

## Primary Endpoint

AURC (Area Under Risk-Coverage curve)

## Secondary Endpoints

- CFR@80 relative reduction
- Selective Dice@80
- AUROC

## Required Sample Size

Minimum: 30 queries with >= 10 catastrophic events
Recommended: 50+ queries (based on power analysis)

## Confirmatory Criteria

All must pass:
1. AURC_PC < AURC_Random (point estimate)
2. AURC paired CI does not cross 0
3. AUROC >= 0.70
4. CFR@80 relative reduction >= 30%
5. Dice@80 within 0.005 of Random accepted set
6. Risk-coverage curve directionally reasonable
7. No GT used in risk construction
8. Query is the statistical unit

## What We Do If Confirmatory Unavailable

- Report all results as exploratory/development
- Use external datasets as secondary validation
- Be honest about data limitations
- Do not claim confirmation