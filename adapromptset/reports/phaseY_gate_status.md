# Phase Y Gate Status

Date: 2026-07-21 (final)

Phase: COMPLETE. Final state: `PHASEY_SEARCH_ORACLE_DIAGNOSTIC_ONLY`.

Directive: `AdaPromptSet_PhaseY_Policy_Feasibility.md`

## Pre-Gate: Preregistration Completeness

Status: `COMPLETE`

Preregistration commit: `2d61ef8` on branch `phaseY-policy-feasibility`.
All required artifacts created and hashed before any new SAM inference.

## Gate Y1 — Fresh-Query Action Heterogeneity

Status: `FAILED` → `PHASEY_GATE_Y1_ORACLE_GAP_SMALL`

Scope: 41 validation queries × 2 reference sets (A, B) × 3 references = 246 pairs, 210 actions per pair.
Runtime: ~3.13 GPU-hours (11264.2s).

Results (6 conditions, 2 passed):
1. `stable_oracle_gap = 0.0061` (need ≥0.015) → **FAIL**
2. bootstrap 95% CI = [-0.057, 0.070] (need lower >0) → **FAIL**
3. `diff_from_fixed_rate = 0%` (need ≥30%) → **FAIL**
4. `median Spearman ρ = 0.745` (need >0.30) → **PASS**
5. `optimism_ratio_A = 1.12` (need ≤0.60) → **FAIL**
6. all actions 1 SAM call → **PASS**

Key metrics:
- Strongest fixed: `Y_P_TOPK_K50_N0_N_NONE_G1_C1`, Dice=0.4338
- Raw oracle A: 0.5007 (gap=0.0669)
- Cross A→B: 0.4258 (below fixed!), Cross B→A: 0.4539
- Stable oracle: 0.4399, stable gap: 0.0061
- Optimism A: 0.0749 (112% of raw gap)

Interpretation: Phase X's raw oracle gap (0.0708 from 10 queries) was almost entirely Winner's Curse / selection bias. Under fresh-query cross-reference evaluation, the stable gap collapses to 0.0061.

## Gate Y2 — Search Oracle Stability

Status: `FAILED` → `PHASEY_GATE_Y2_SEARCH_ORACLE_OVERFIT`

Scope: 41 validation queries, GT-guided greedy search on dual reference sets A/B with cross-evaluation.
Runtime: ~8.15 GPU-hours (29357.5s), 148757 SAM calls (search budget, not deployed calls).

Results (5 conditions, 1 passed):
1. `cross-reference search Dice = 0.4105` (need ≥ fixed + 0.03 = 0.4638) → **FAIL**
2. paired bootstrap 95% CI = [-0.089, 0.040] (need lower >0) → **FAIL**
3. `optimism_ratio = 1.1449` (need ≤0.50) → **FAIL**
4. `median |K_A - K_B| = 6.0` (need ≤8) → **PASS**
5. `A beats fixed on B = 26.8%` (need ≥70%) → **FAIL**

Key metrics:
- Fixed baseline: `Y_P_TOPK_K50_N0_N_NONE_G1_C1`, Dice=0.4338
- Raw search Dice: 0.5947 (gap over fixed = 0.1609)
- Cross-reference search Dice: 0.4105 (BELOW fixed 0.4338!)
- Search optimism: 0.1843 (114.49% of raw gap)
- Mean point overlap (Jaccard): 0.3322
- Mean rank overlap (Jaccard): 0.5577
- Mean spatial overlap (IoU): 0.8979

Interpretation: The Phase X search oracle Dice of 0.7256 (from 10 exploratory queries) does NOT persist under cross-reference evaluation. Raw search Dice on the full 41-query validation set is only 0.5947, and cross-reference Dice collapses to 0.4105 — below the strongest fixed action. The search oracle overfits to the reference set's candidate pool and cannot generalize to a different reference.

## Gate Y3 — Bank Policy Validation

Status: `BLOCKED_BY_GATE_Y1_FAIL`

Y1 failure prohibits bank-action policy training per directive section 19.1.

## Gate Y4 — Point-Subset Learnability

Status: `BLOCKED_BY_GATE_Y2_FAIL`

Y2 failure prohibits Point-Subset Track per directive section 19.1 routing (Y1 fail + Y2 fail → diagnostic only).

## Holdout Evaluation

Status: `BLOCKED_BY_GATE_Y1_FAIL`

Y1 failure prohibits holdout access per directive section 19.1.

## Test Lock

Status: `LOCKED_UNUSED`

MoNuSAC formal test (101 cases), GlaS testA/testB, DDR test, IDRID test
remain locked. No test access for K, action bank, router, features,
model capacity, or seed selection.

## Final State

`PHASEY_SEARCH_ORACLE_DIAGNOSTIC_ONLY`

Per directive section 19.1: Y1 fail + Y2 fail → diagnostic only.

Allowed claim (section 21, 情况 A):
> The large raw oracle gap observed in the exploratory 10-query audit did not persist under fresh-query and cross-reference evaluation.

Total Phase Y GPU usage: 11.28 GPU-hours (Y1: 3.13h + Y2: 8.15h) / 155 GPU-hour cap.

Final report: `reports/phaseY_final_report.md` and `reports/phaseY_final_report.json`.
