# Phase S Decision Tree

Date: 2026-07-22
Branch: `phaseS-multitrack-reference-robust-research`

## Routing Logic (directive §10)

```
Track A completes
├── PASS (instability benchmark valid)
│   ├── Reliability predictable (Spearman ≥ 0.30 on train) → Track B
│   │   ├── B PASS (reliability weighting pass)
│   │   │   ├── Set-interaction signal + B plateaus → Track C
│   │   │   │   ├── C PASS → freeze method, holdout
│   │   │   │   └── C NO_ADDED_VALUE → B is final method, holdout
│   │   │   └── No set signal → B is final method, holdout
│   │   ├── B UTILITY_ONLY_PASS → benchmark + utility prediction paper
│   │   ├── B STABILITY_ONLY_PASS → benchmark + stability paper
│   │   └── B NO_GAIN
│   │       ├── Prompt-interface bottleneck confirmed → Track D
│   │       │   ├── D PASS → freeze method, holdout
│   │       │   └── D NO_GAIN → benchmark / negative-result paper
│   │       └── No prompt-interface bottleneck → benchmark / negative-result paper
│   └── Reliability NOT predictable → Track D (if interface bottleneck) or benchmark paper
└── FAIL (metric or phenomenon not stable)
    → Stop reference-stability mainline
    → Evaluate Track D as alternative
    → If D also fails → negative-result paper or program exhausted
```

## Holdout Gate (§11)

Only after a track's validation Gate passes:
1. Freeze method (code, manifest, sampling, features, aggregation, checkpoint,
   prompt config, thresholds, statistics, qualitative selection, cost).
2. Create `docs/phaseS_holdout_freeze.md`.
3. Run holdout exactly ONCE.
4. If pass → cross-dataset; if fail → `PHASES_HOLDOUT_CONFIRMATION_FAIL`,
   no retry.

## Cross-Dataset Gate (§12)

Only after holdout passes:
1. GlaS 55/30 first.
2. DDR one lesion intent second.
3. Distinguish zero-shot frozen transfer vs same-protocol retraining vs
   dataset-specific tuning.

## Publication Routing (§21)

| Outcome | Paper Type |
|---|---|
| B + holdout + cross-dataset pass | Method paper |
| A valid + B partial + cross-dataset | Benchmark + partial method |
| A valid, all methods fail | Benchmark / failure-boundary paper |
| A fails, D passes | Direct-method paper |
| All fail | Negative-result / program-exhausted report |

## Auto-Pause Conditions

- 50 GPU-hours → Track A/B interim review.
- 100 GPU-hours → Primary track decision.
- 160 GPU-hours → Holdout readiness review.
- 210 GPU-hours → STOP, request user approval.
