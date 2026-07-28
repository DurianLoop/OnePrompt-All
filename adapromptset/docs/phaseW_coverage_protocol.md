# Phase W — Coverage Protocol

**Date**: 2026-07-27

## Coverage Grid (per todo.md Section 7.1)

coverage ∈ {50%, 60%, 70%, 80%, 90%, 95%, 100%}

## Metrics per Coverage Level
- Accepted queries
- CFR
- Dice
- AJI
- Component recall
- Selective risk
- Bootstrap CI
- Failure prevalence
- Abstention count

## Key Statistics
- AURC
- Excess AURC
- CFR@50/60/70/80/90/95
- Dice@50/60/70/80/90/95

## Gate
- Risk must monotonically increase with coverage
- Different seeds must show consistent direction
- Cannot use only 80% coverage to support "deployment envelope"