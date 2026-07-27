# Phase W — Component Ablation Protocol

**Date**: 2026-07-27

## Background

Current component ablation (phaseV_component_ablation_fixed) shows:
- RAMP-full and RAMP-no-weights produce identical Dice (0.4545 for both)
- Only RAMP-no-consensus differs meaningfully
- The "fix" was a post-hoc Dice recalculation, not a real re-run

## Required Variants (per todo.md Section 5.2)

| ID | Risk Weighting | Adaptive Fusion | Abstention | Description |
|----|---------------|-----------------|------------|-------------|
| W0 | no | no | no | Baseline (raw multi-reference mean) |
| W1 | yes | no | no | Weighting only |
| W2 | no | yes | no | Fusion only |
| W3 | no | no | yes | Abstention only |
| W4 | yes | yes | no | Weighting + fusion |
| W5 | yes | no | yes | Weighting + abstention |
| W6 | no | yes | yes | Fusion + abstention |
| W7 | yes | yes | yes | Full RAMP |

## Key Observation

From the fixed ablation, RAMP-full = RAMP-no-weights. This means:
- Weighting component has ZERO independent effect
- The only benefit comes from consensus (multi-reference) vs single-reference
- This is a critical finding for the paper route decision

## Protocol

1. Use real per-reference Dice from phase2_raw_per_call.csv
2. Train risk model on all 80 queries (development cohort)
3. For each variant, compute the actual fused prediction
4. Save mask hashes for each variant
5. Compare at query level with bootstrap CI