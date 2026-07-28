# Phase R Statistics Protocol (Frozen)

Date: 2026-07-22

Per directive sections 10-13. All thresholds are frozen at preregistration.

## Bootstrap

- n_resamples: 10000
- CI level: 0.95
- Statistical unit: query (not reference episode)

## Catastrophic failure

- Threshold: Dice < 0.10

## Gate R1: Reference Instability Replication (5 conditions)

1. >= 30% train queries have best-worst Dice gap >= 0.05
2. mean reference std >= 0.02
3. instability direction consistent across 5 resampling seeds
4. at least one prompt config shows stable reference sensitivity
5. instability NOT caused by code/coordinate/leakage error

Pass: `PHASER_GATE_R1_REFERENCE_INSTABILITY_CONFIRMED`
Fail: `PHASER_GATE_R1_REFERENCE_INSTABILITY_NOT_REPLICATED`

## Gate R2: Deterministic Consensus Feasibility (8 conditions)

1. Best deterministic vs Random Single: Delta Dice >= 0.010
2. paired bootstrap 95% CI lower > 0
3. reference variance reduction >= 25%
4. catastrophic failure does not increase
5. A/B prediction-mask overlap improves
6. still 1 SAM call
7. no query GT
8. at least 2 of 3 prompt configs directionally consistent

Pass: `PHASER_GATE_R2_DETERMINISTIC_CONSENSUS_PASS`
Fail: `PHASER_GATE_R2_CONSENSUS_NO_GAIN`

## Gate R3: Reference Reliability Learnability (9 conditions)

1. Weighted vs best deterministic: Delta Dice >= 0.005
2. Weighted vs Random Single: Delta Dice >= 0.015
3. paired bootstrap CI point estimate > 0
4. reference utility prediction Spearman rho >= 0.30
5. worst-reference risk reduced
6. catastrophic failure does not increase
7. 3 seeds directionally consistent
8. weights do not collapse to same reference position
9. no query GT

Pass: `PHASER_GATE_R3_REFERENCE_WEIGHTING_PASS`
Fail: `PHASER_GATE_R3_REFERENCE_WEIGHTING_NOT_LEARNABLE`

## Gate R4: Confirmatory Holdout (7 conditions)

1. vs Random Single: Delta Dice >= 0.0075
2. paired bootstrap point estimate > 0
3. reference variance reduction >= 20%
4. catastrophic failure does not increase
5. worst-query tail improved
6. 1 SAM call
7. no post-holdout tuning

Pass: `PHASER_REFERENCE_STABILITY_CONFIRMED`
Fail: `PHASER_HOLDOUT_CONFIRMATION_FAIL`
