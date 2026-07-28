# Phase AB Power Analysis Protocol

## Objective

Estimate required sample size for confirmatory experiment using Development Cohort effect distributions.

## Method

1. Use Development Cohort (80 queries, L+SAM-B pipeline) query-level paired results
2. Bootstrap resample at N = {20, 30, 40, 50, 75, 100, 150}
3. 5000 iterations per N
4. Compute:
   - Power to detect AURC improvement (alpha=0.05, one-sided)
   - CFR@80 CI width
   - Catastrophic event count distribution
   - Minimum detectable effect size

## Primary Endpoint

AURC difference (PC - Random)

## Output

- Power curve: power vs N
- Event count simulation: expected catastrophic events per N
- Recommended minimum N for power >= 0.80
- Warning if expected events < 10

## Decision Rules

- If power >= 0.80 at N <= 50: feasible
- If power >= 0.80 at N > 50: marginal
- If power < 0.80 at N = 150: underpowered
- If expected events < 10: CFR downgraded to secondary