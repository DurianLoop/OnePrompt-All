# Phase R Gate R2: Deterministic Consensus Feasibility Summary

Date: 2026-07-22T14:24:05
Validation queries: 41
Best config: P30-N0
Best consensus method: B2_mean (hyperparam=-, beta=0.25, tau=0.83)

## Consensus vs Random Single (B1)

| Metric | Value |
|---|---:|
| Mean consensus Dice | 0.4756 |
| Mean B1 random single Dice | 0.4297 |
| Delta Dice | 0.0460 |
| Bootstrap 95% CI | [0.0125, 0.0835] |
| Var(single) | 0.0543 |
| Var(consensus) | 0.0571 |
| Variance reduction | -0.0506 |
| CFR consensus | 0.0732 |
| CFR B1 | 0.1220 |
| Mean consensus A/B overlap | 0.6075 |
| Mean single A/B overlap | 0.5942 |
| Consistent config count (of 3) | 3 |

## Gate R2 Conditions

| # | Condition | Pass |
|---|---|---|
| 1 | Delta Dice >= 0.010 (consensus - B1) | YES |
| 2 | Paired bootstrap 95% CI lower > 0 | YES |
| 3 | Variance reduction >= 25% | NO |
| 4 | Catastrophic failure rate not increase | YES |
| 5 | A/B prediction overlap improve | YES |
| 6 | Single SAM call (by construction) | YES |
| 7 | No query GT used (by construction) | YES |
| 8 | >= 2 of 3 configs consistent direction | YES |

## Decision: **PHASER_GATE_R2_CONSENSUS_NO_GAIN**

