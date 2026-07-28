# Phase R Claim Registry

Date: 2026-07-22

## Only Allowed Claim Candidate

> Reference-stable aggregation and stability-aware prompt generation can reduce
> performance variance across valid references and improve cross-reference
> segmentation reliability under the same SAM-call budget.

## Allowed Conclusions by Gate Outcome

### Only R1 passes

> Valid references induce substantial segmentation variability under the current
> frozen matching-and-prompting pipeline.

### R2 passes

> Deterministic multi-reference consensus reduces reference-induced variance and
> improves cross-reference reliability under one SAM call.

### R3 passes

> Test-time-safe reference reliability estimation further improves over uniform consensus.

### Holdout passes

> Reference-stable prompting confirms on a fresh internal holdout and is ready
> for a separately frozen formal-test protocol.

## Permanently Forbidden Claims

- "reference instability solved"
- "universal across datasets"
- "SOTA"
- "formal test success"
- "ProSAM/MAUP/SAMIX fully outperformed"
- "reference weighting is model agnostic"
- "mean Dice alone proves robustness"
