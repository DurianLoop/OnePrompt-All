# Phase X Candidate Coverage Scaling Summary

Date: 2026-07-20
Stage: candidate_scaling (Gate X2)
Episodes: 10
Pool max positives: 100
Pool max negatives: 100
CPU-only: True
SAM inference run: False

## Gate X2 Status

**CANDIDATE_UNIVERSE_INADEQUATE**

even at K+=100, component coverage is 0.433 < 0.5; problem is matcher/candidate generator, not policy

## Positive coverage curve

| K+ | Episodes | Mean component coverage | Mean point precision | Mean spatial dispersion |
|---|---|---|---|---|
| 1 | 10 | 0.0119 | 0.5000 | 0.00 |
| 3 | 10 | 0.0327 | 0.5333 | 224.84 |
| 5 | 10 | 0.0397 | 0.4800 | 227.63 |
| 10 | 10 | 0.0689 | 0.4600 | 211.15 |
| 20 | 10 | 0.1155 | 0.4900 | 231.07 |
| 30 | 10 | 0.1623 | 0.4867 | 238.75 |
| 50 | 10 | 0.2453 | 0.5060 | 241.12 |
| 75 | 10 | 0.4012 | 0.4947 | 259.74 |
| 100 | 10 | 0.4328 | 0.4070 | 271.85 |

## Negative quality curve

| K- | Episodes | Mean background precision | Mean false-negative risk |
|---|---|---|---|
| 1 | 10 | 1.0000 | 0.0000 |
| 3 | 10 | 0.9333 | 0.0667 |
| 5 | 10 | 0.9400 | 0.0600 |
| 10 | 10 | 0.9500 | 0.0500 |
| 20 | 10 | 0.9650 | 0.0350 |
| 30 | 10 | 0.9700 | 0.0300 |
| 50 | 10 | 0.9700 | 0.0300 |

## Interpretation

- If `OLD_SMALL_K_BANK_INADEQUATE`: the historical 5-positive bank was too small.
- If `CANDIDATE_UNIVERSE_INADEQUATE`: the matcher/candidate generator is the bottleneck.
- If `CANDIDATE_SCALING_SUPPORTS_LARGE_K`: dense prompts are viable.
- If `CANDIDATE_SCALING_SATURATES_SMALL_K`: small K+ is sufficient.
- If `CANDIDATE_SCALING_INSUFFICIENT_DATA`: regenerate the pool with positive_limit>=100.

