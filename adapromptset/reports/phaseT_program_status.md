# Phase T Program Status

Date: 2026-07-23
Branch: `phaseT-publication-completion`
Directive: `D:\SAM\adapromptset\reports\AdaPromptSet_PhaseT_Publication_Completion.md`

## Current State

`PHASET_WP1_MONUSAC_HOLDOUT_CONFIRMED`

## Work Packages

| WP | Name | Status | Gate |
|---|---|---|---|
| WP1 | Confirmatory Benchmark (MoNuSAC holdout=40) | PASS (10/10) | §5.6 (10 conditions) |
| WP2 | Cross-Dataset + Cross-Method Replication | PENDING | §6.4 + §7.3 |
| WP3 | Risk-Aware Mitigation (M1 + M2) | PENDING | §9.6 + §10.5 |
| WP4 | Paper Construction + Mock Review | PENDING | §20 (12 conditions) |

## Compute Budget

- Cap: 195 GPU-hours
- Used: 0 GPU-hours (WP1 ran on CPU)
- Remaining: 195 GPU-hours
- Checkpoints: 40 / 90 / 140 / 195 GPU-hours

## Gate Summary

| Gate | Status |
|---|---|
| WP1 holdout | PASS — PHASET_WP1_MONUSAC_HOLDOUT_CONFIRMED (10/10 conditions) |
| WP2 cross-dataset | UNBLOCKED — audit GlaS/MoNuSeg/DDR |
| WP2b external baselines | UNBLOCKED — build comparator matrix |
| WP3 M1 dynamic budget | BLOCKED_BY_WP2 |
| WP3 M2 selective prediction | BLOCKED_BY_WP2 |
| WP4 publication readiness | BLOCKED_BY_WP1/WP2/WP3 |
| Formal test | LOCKED_UNUSED |

## WP1 Result Summary

- 40 holdout queries × 10 seeds × 3 configs (P20-N1, P30-N0, P50-N0)
- 9600 SAM calls, 1526.5s on CPU
- 10/10 gate conditions PASSED
- Key metrics:
  - 92.5% queries with best-worst gap ≥ 0.05
  - V_ref_frac (k=1): 0.1264–0.2627
  - Consensus gain: +0.0266 Dice (bootstrap CI lower = 0.014)
  - Best disagreement feature: prediction_jaccard_k1_within (Spearman |ρ| = 0.7241)
  - 0 data leakage violations across 2400 reference assignments

## Starting Scientific State (from Phase S)

- Reference instability is real: corrected `V_ref_frac` 0.1376–0.3312.
- Reference instability is predictable: validation Spearman 0.478.
- Reference instability CANNOT be mitigated through reference weighting
  (Track B `NO_GAIN`) or set conditioning (Track C `NO_GAIN`).
- Mean consensus is the current strongest automatic method (+0.046 Dice over
  random single, single SAM call).
- Final Phase S state: `PHASES_BENCHMARK_PAPER_READY`.

## Next Action

WP1 PASSED. Per directive §26 steps 7-8: audit GlaS/MoNuSeg/DDR datasets and
run cross-dataset replication on at least 2 additional datasets under the
unified protocol (directive §6.3).

## Prohibited Next Actions

- Running WP1 inference before the holdout protocol is frozen and the
  preregistration commit is made.
- Accessing formal test under any circumstance.
- Retuning any frozen parameter after holdout observation.
- Treating the historical Phase X 10 queries as fresh holdout evidence.
- Writing `SUBMISSION_READY`, `PUBLICATION_COMPLETED`, or `SOTA` in any
  report until the Publication Readiness Gate (§20) passes.
