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

Status: `IN_PROGRESS` (R1 passed, running B1-B6 consensus on train, then val=41)

## Gate R3 — Reference Reliability Learnability

Status: `NOT_STARTED` (blocked by R2)

## Gate R4 — Confirmatory Holdout

Status: `NOT_STARTED` (blocked by R2 or R3)

## GlaS Secondary Diagnostic

Status: `NOT_STARTED` (blocked by MoNuSAC R2)

## Test Lock

Status: `LOCKED_UNUSED`

## Compute Budget

- Total cap: 85 GPU-hours
- Used: 0.139 GPU-hours (R1 only)
- Remaining: 84.861 GPU-hours
