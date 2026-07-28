# Phase X Stage S2: Single-Variable K+ Scaling Summary

Date: 2026-07-20
Stage: single_variable_kpos (Gate X3)
Episodes: 10
Positive limit: 100
Device: cpu
SAM inference run: True
Elapsed seconds: 59.507
GPU-hours: 0.0

## Gate X3 Status

**DENSE_PROMPT_VIABILITY_ESTABLISHED**

K=5 -> K=100 Dice gain = 0.097, component recall gain = 0.108; dense prompts improve both Dice and recall

## K+ scaling curve

| K+ | Episodes | Mean Dice | Mean IoU | Mean AJI | Mean component recall | Mean runtime (ms) |
|---|---|---|---|---|---|---|
| 1 | 10 | 0.0292 | 0.0160 | 0.0160 | 0.0127 | 23.0 |
| 3 | 10 | 0.2561 | 0.1524 | 0.1524 | 0.6560 | 22.4 |
| 5 | 10 | 0.3260 | 0.1967 | 0.1967 | 0.8916 | 23.4 |
| 10 | 10 | 0.4668 | 0.3148 | 0.3148 | 0.9420 | 24.7 |
| 20 | 10 | 0.5237 | 0.3642 | 0.3642 | 0.9678 | 26.1 |
| 30 | 10 | 0.5173 | 0.3569 | 0.3569 | 1.0000 | 26.5 |
| 50 | 10 | 0.4419 | 0.2857 | 0.2857 | 1.0000 | 31.3 |
| 75 | 10 | 0.3972 | 0.2482 | 0.2482 | 1.0000 | 32.3 |
| 100 | 10 | 0.4234 | 0.2692 | 0.2692 | 1.0000 | 33.9 |

## Interpretation

- `DENSE_PROMPT_VIABILITY_ESTABLISHED`: dense prompts improve both Dice and component recall.
- `DENSE_PROMPT_NO_GAIN`: K+ scaling does not improve Dice or component recall.
- `DENSE_PROMPT_COST_ONLY_GAIN`: dense prompts improve recall but not Dice (cost-only gain).
- `DENSE_PROMPT_ABSOLUTE_PERFORMANCE_LOW`: even at K=100, absolute Dice is too low.
- `DENSE_PROMPT_INSUFFICIENT_DATA`: missing K=5 or K=100 data points.
