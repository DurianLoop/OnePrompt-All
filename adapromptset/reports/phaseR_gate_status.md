# Phase R Gate Status

Date: 2026-07-22

Phase: EXECUTION

Directive: `AdaPromptSet_PhaseR_ReferenceStable_Feasibility.md`

## Pre-Gate: Preregistration Completeness

Status: `COMPLETE` (commit d925276, 2026-07-22)

## Gate R1 — Reference Instability Replication

Status: `PHASER_GATE_R1_REFERENCE_INSTABILITY_CONFIRMED`

- Date: 2026-07-22T13:40:48
- Train queries: 80, seeds: 5, configs: 3 (P20-N1, P30-N0, P50-N0)
- Total SAM calls: 7200
- Runtime: 501.7s (0.139 GPU-hours)
- % queries best-worst gap >= 0.05: 0.8875 (need >= 0.30) — PASS
- Mean reference std: 0.0737 (need >= 0.02) — PASS
- Seed direction consistent: YES — PASS
- Sensitive config: P20-N1 — PASS
- Not code error: YES — PASS
- All 5 conditions passed

## Gate R2 — Deterministic Consensus Feasibility

Status: `PHASER_GATE_R2_CONSENSUS_NO_GAIN`

- Date: 2026-07-22T14:24:05
- Train: 80 queries × 5 seeds × 3 configs × 10 method variants (B1-B6 with β/τ grid)
- Validation: 41 queries × 3 configs × multiple methods
- Total SAM calls: 3138 (1539 train new + 1599 val)
- Runtime: 271.7s (0.0755 GPU-hours)
- Best config (selected on train): P30-N0
- Best consensus method (selected on train): B2_mean (β=0.25, τ=0.83)
- Mean consensus Dice (val): 0.4756
- Mean B1 random single Dice (val): 0.4297
- Δ Dice: +0.0460
- Bootstrap 95% CI: [0.0125, 0.0835]

### Conditions (7 of 8 passed, 1 failed)

| # | Condition | Result |
|---|---|---|
| 1 | Δ Dice >= 0.010 (consensus - B1) | PASS (0.0460) |
| 2 | Paired bootstrap 95% CI lower > 0 | PASS (0.0125) |
| 3 | Variance reduction >= 25% | **FAIL** (-0.0506, variance increased) |
| 4 | Catastrophic failure rate not increase | PASS (0.0732 vs 0.1220) |
| 5 | A/B prediction overlap improve | PASS (0.6075 vs 0.5942) |
| 6 | Single SAM call (by construction) | PASS |
| 7 | No query GT used (by construction) | PASS |
| 8 | >= 2 of 3 configs consistent direction | PASS (3/3) |

**R2 FAILED**: Variance reduction condition failed — consensus aggregation
increased per-query variance from 0.0543 (B1) to 0.0571 (consensus), a 5.06%
increase rather than the required ≥25% reduction. Although Δ Dice and
catastrophic-failure rate improved, the central R2 hypothesis (deterministic
consensus reduces reference-induced variance) is not supported.

## Gate R3 — Reference Reliability Learnability

Status: `BLOCKED` (R2 failed; directive §12: "只有 Gate R2 通过才训练")

## Gate R4 — Confirmatory Holdout

Status: `BLOCKED` (R2 and R3 both failed; directive §13: "只有 R2 或 R3 通过")

## GlaS Secondary Diagnostic

Status: `BLOCKED` (R2 failed; directive §14: "只有 MoNuSAC R2 通过后运行 GlaS")

## Test Lock

Status: `LOCKED_UNUSED` (directive §22: "不访问 formal test")

## Compute Budget

- Total cap: 85 GPU-hours
- Used: 0.2153 GPU-hours (R1: 0.1394 + R2: 0.0755)
- Remaining: 84.7847 GPU-hours

## Final Phase R State

`PHASER_REFERENCE_STABILITY_NOT_FEASIBLE_DETERMINISTIC`

Reference instability (R1 PASSED) cannot be mitigated by deterministic
consensus (R2 FAILED on variance-reduction condition). Reliability-learnability
(R3) and confirmatory holdout (R4) are blocked. GlaS secondary diagnostic is
blocked. Formal test remains locked and unused.
