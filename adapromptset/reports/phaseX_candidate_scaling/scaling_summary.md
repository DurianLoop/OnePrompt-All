# Phase X Candidate Coverage Scaling Summary

Date: 2026-07-20
Stage: candidate_scaling (Gate X2)
Episodes: 10
Pool max positives: 9
Pool max negatives: 9
CPU-only: True
SAM inference run: False

## Gate X2 Status

**CANDIDATE_UNIVERSE_INADEQUATE**

component coverage at K+=10 (grid ceiling for pool max 9) is 0.052 < 0.5; candidate pool too small to reach K+=100 and already inadequate

## Positive coverage curve

| K+ | Episodes | Mean component coverage | Mean point precision | Mean spatial dispersion |
|---|---|---|---|---|
| 1 | 10 | 0.0119 | 0.5000 | 0.00 |
| 3 | 10 | 0.0119 | 0.5000 | 0.00 |
| 5 | 10 | 0.0233 | 0.5000 | 173.16 |
| 10 | 10 | 0.0520 | 0.5111 | 240.70 |
| 20 | 10 | 0.0520 | 0.5111 | 240.70 |
| 30 | 10 | 0.0520 | 0.5111 | 240.70 |
| 50 | 10 | 0.0520 | 0.5111 | 240.70 |
| 75 | 10 | 0.0520 | 0.5111 | 240.70 |
| 100 | 10 | 0.0520 | 0.5111 | 240.70 |

## Negative quality curve

| K- | Episodes | Mean background precision | Mean false-negative risk |
|---|---|---|---|
| 1 | 10 | 1.0000 | 0.0000 |
| 3 | 10 | 1.0000 | 0.0000 |
| 5 | 10 | 0.9600 | 0.0400 |
| 10 | 10 | 0.9444 | 0.0556 |
| 20 | 10 | 0.9444 | 0.0556 |
| 30 | 10 | 0.9444 | 0.0556 |
| 50 | 10 | 0.9444 | 0.0556 |

## Interpretation

- If `OLD_SMALL_K_BANK_INADEQUATE`: the historical 5-positive bank was too small.
- If `CANDIDATE_UNIVERSE_INADEQUATE`: the matcher/candidate generator is the bottleneck.
- If `CANDIDATE_SCALING_SUPPORTS_LARGE_K`: dense prompts are viable.
- If `CANDIDATE_SCALING_SATURATES_SMALL_K`: small K+ is sufficient.
- If `CANDIDATE_SCALING_INSUFFICIENT_DATA`: regenerate the pool with positive_limit>=100.

