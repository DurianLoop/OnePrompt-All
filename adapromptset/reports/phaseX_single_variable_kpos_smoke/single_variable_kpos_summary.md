# Phase X Stage S2: Single-Variable K+ Scaling Summary

Date: 2026-07-20
Stage: single_variable_kpos (Gate X3)
Episodes: 1
Positive limit: 100
Device: cpu
SAM inference run: True
Elapsed seconds: 5.035
GPU-hours: 0.0

## Gate X3 Status

**DENSE_PROMPT_COST_ONLY_GAIN**

K=5 -> K=100 Dice gain = -0.016 < 0.01 but component recall gain = 0.056 >= 0.05; dense prompts improve recall but not Dice

## K+ scaling curve

| K+ | Episodes | Mean Dice | Mean IoU | Mean AJI | Mean component recall | Mean runtime (ms) |
|---|---|---|---|---|---|---|
| 1 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 26.8 |
| 3 | 1 | 0.0019 | 0.0009 | 0.0009 | 0.0300 | 23.0 |
| 5 | 1 | 0.3986 | 0.2489 | 0.2489 | 0.9442 | 27.7 |
| 10 | 1 | 0.4307 | 0.2744 | 0.2744 | 0.9571 | 22.8 |
| 20 | 1 | 0.3594 | 0.2191 | 0.2191 | 1.0000 | 25.0 |
| 30 | 1 | 0.3543 | 0.2153 | 0.2153 | 1.0000 | 25.3 |
| 50 | 1 | 0.3616 | 0.2207 | 0.2207 | 1.0000 | 28.4 |
| 75 | 1 | 0.3668 | 0.2246 | 0.2246 | 1.0000 | 27.6 |
| 100 | 1 | 0.3829 | 0.2368 | 0.2368 | 1.0000 | 37.5 |

## Interpretation

- `DENSE_PROMPT_VIABILITY_ESTABLISHED`: dense prompts improve both Dice and component recall.
- `DENSE_PROMPT_NO_GAIN`: K+ scaling does not improve Dice or component recall.
- `DENSE_PROMPT_COST_ONLY_GAIN`: dense prompts improve recall but not Dice (cost-only gain).
- `DENSE_PROMPT_ABSOLUTE_PERFORMANCE_LOW`: even at K=100, absolute Dice is too low.
- `DENSE_PROMPT_INSUFFICIENT_DATA`: missing K=5 or K=100 data points.
