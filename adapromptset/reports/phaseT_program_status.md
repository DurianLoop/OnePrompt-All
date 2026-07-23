# Phase T Program Status

Date: 2026-07-23
Branch: `phaseT-publication-completion`
Directive: `D:\SAM\adapromptset\reports\AdaPromptSet_PhaseT_Publication_Completion.md`

## Current State

`PHASES_BENCHMARK_CORE_FINDING_ESTABLISHED`

## Work Packages

| WP | Name | Status | Gate |
|---|---|---|---|
| WP1 | Confirmatory Benchmark (MoNuSAC holdout=40) | PENDING | §5.6 (10 conditions) |
| WP2 | Cross-Dataset + Cross-Method Replication | PENDING | §6.4 + §7.3 |
| WP3 | Risk-Aware Mitigation (M1 + M2) | PENDING | §9.6 + §10.5 |
| WP4 | Paper Construction + Mock Review | PENDING | §20 (12 conditions) |

## Compute Budget

- Cap: 195 GPU-hours
- Used: 0 GPU-hours
- Remaining: 195 GPU-hours
- Checkpoints: 40 / 90 / 140 / 195 GPU-hours

## Gate Summary

| Gate | Status |
|---|---|
| WP1 holdout | PENDING (protocol not yet frozen) |
| WP2 cross-dataset | BLOCKED_BY_WP1 |
| WP2b external baselines | BLOCKED_BY_WP1 |
| WP3 M1 dynamic budget | BLOCKED_BY_WP2 |
| WP3 M2 selective prediction | BLOCKED_BY_WP2 |
| WP4 publication readiness | BLOCKED_BY_WP1/WP2/WP3 |
| Formal test | LOCKED_UNUSED |

## Starting Scientific State (from Phase S)

- Reference instability is real: corrected `V_ref_frac` 0.1376–0.3312.
- Reference instability is predictable: validation Spearman 0.478.
- Reference instability CANNOT be mitigated through reference weighting
  (Track B `NO_GAIN`) or set conditioning (Track C `NO_GAIN`).
- Mean consensus is the current strongest automatic method (+0.046 Dice over
  random single, single SAM call).
- Final Phase S state: `PHASES_BENCHMARK_PAPER_READY`.

## Next Action

Freeze the MoNuSAC holdout benchmark protocol (`docs/phaseT_holdout_protocol.md`
already created; the WP1 freeze commit and
`configs/frozen/phaseT_monusac_benchmark.yaml` are the next concrete
artifacts) and run WP1.

Per directive §26 step 4: freeze MoNuSAC holdout benchmark protocol before
any WP1 inference. The holdout may be observed exactly once per frozen
method; hash verification is required before the run begins.

## Prohibited Next Actions

- Running WP1 inference before the holdout protocol is frozen and the
  preregistration commit is made.
- Accessing formal test under any circumstance.
- Retuning any frozen parameter after holdout observation.
- Treating the historical Phase X 10 queries as fresh holdout evidence.
- Writing `SUBMISSION_READY`, `PUBLICATION_COMPLETED`, or `SOTA` in any
  report until the Publication Readiness Gate (§20) passes.
