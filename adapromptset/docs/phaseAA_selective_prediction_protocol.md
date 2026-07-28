# Phase AA Selective Prediction Protocol

**Date**: 2026-07-27

## Main Method

1. Use m references to generate m masks independently
2. Compute Prediction Consistency risk score
3. Rank queries by risk (ascending)
4. At given coverage c, accept the c-fraction of lowest-risk queries
5. Abstain on high-risk queries

## Coverage Levels

Standard: 50%, 60%, 70%, 80%, 90%, 95%, 100%

## Primary Endpoint

AURC (Area Under Risk-Coverage curve)

## Secondary Endpoints

- CFR@80 (Catastrophic Failure Rate at 80% coverage)
- Selective Dice@80
- AUROC
- AUPRC

## Accepted Prediction

Mean-consensus prediction (frozen after development).

## What is NOT Included

- Learned weighting
- Adaptive fusion
- Prompt adaptation
- RRN
- Query-specific reference selection