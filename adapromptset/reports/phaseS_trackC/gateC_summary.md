# Phase S Track C: Stable Candidate Scoring — Gate C Summary

Date: 2026-07-22T18:00:09
Validation queries: 41
Best config: P30-N0
Hyperparameters: α=0.0, β=0.0, γ=0.0

## Track C vs Track B (baseline)

| Metric | Track C | Track B |
|---|---:|---:|
| Mean Dice | 0.4779 | 0.4732 |
| CVaR_0.1 | 0.0738 | 0.0768 |
| CFR (Dice<0.1) | 0.0732 | 0.0732 |

Delta Dice vs Track B: 0.0047
CVaR improvement: -0.0031
Mean train Dice: 0.5148 (train-val gap: 0.0369)
Bootstrap mean delta: 0.0047 (CI [-0.0119, 0.0242])

## Gate C Conditions (§8.4)

| # | Condition | Pass |
|---|---|---|
| 1 | Mean Dice +0.005 vs Track B | NO |
| 2 | CVaR +0.01 vs Track B | NO |
| 3 | Catastrophic failure not increase | YES |
| 4 | 3 seeds consistent (≥2/3 beat B2_mean) | YES |
| 5 | Reference count variation stable (k=3 vs k=6) | YES |
| 6 | LOO no crash / no big drop (<0.05) | NO |
| 7 | Params <5M, 1 SAM call, <2x Track B runtime | YES |
| 8 | Train-val gap < 0.05 | YES |

Conditions passed: 5/8
Gate pass: False
## Decision: **PHASES_TRACK_C_NO_GAIN**

