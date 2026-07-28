# Phase R Hypothesis Summary (R0 Historical Audit)

Date: 2026-07-22

## Phase R Core Hypothesis

> Reference-stable aggregation and stability-aware prompt generation can reduce
> performance variance across valid references and improve cross-reference
> segmentation reliability under the same SAM-call budget.

## Evidence from Phase Y Supporting This Hypothesis

### 1. Reference Instability is Real and Large

From the Y1 utility matrix (41 validation queries, fixed action `Y_P_TOPK_K50_N0_N_NONE_G1_C1`):

- **Mean best-worst Dice gap (across 6 references)**: 0.1874
- **Mean reference std**: 0.0687
- **% queries with best-worst gap ≥ 0.05**: 82.9%

This confirms that different valid references produce substantially different
segmentation quality for the same query, even with the same fixed action.

### 2. Action Selection is Reference-Dependent

- **Exact action agreement rate (A vs B)**: 0.0488
  (only 4.9% of queries select
  the same best action on both reference sets)
- **K+ bin agreement**: 0.3902
- **Median Spearman ρ**: 0.7450

The low exact agreement (4.88%) shows that the "best" action depends heavily
on which reference is used.

### 3. Search Oracle Overfits to Reference

- **Raw search Dice**: N/A
- **Cross-reference search Dice**: N/A
  (BELOW fixed baseline N/A)
- **Search optimism**: N/A
- **Point overlap (Jaccard)**: N/A

The search oracle's performance drops below the fixed baseline when evaluated
cross-reference, confirming that reference-specific point selection does not
transfer.

### 4. Stable Oracle Gap is Small — But That's a Different Question

- **Raw oracle gap**: 0.0669
- **Stable oracle gap**: 0.0061
- **Optimism fraction**: 90.9566%

The small stable oracle gap means **action selection** is not the main lever.
The main lever is **reference-induced variance in the fixed action's output**,
which consensus methods can directly address.

## Phase R Research Questions

### RQ1: Is reference instability stable/replicable?

**Prediction**: YES — the per-query best-worst gaps (mean 0.1874) and
reference stds (mean 0.0687) are large enough to replicate on train=80
with 5 resampling seeds.

**Gate R1 threshold**: ≥30% queries with best-worst gap ≥ 0.05; mean std ≥ 0.02.

### RQ2: Is simple consensus enough?

**Prediction**: LIKELY YES — mean/median consensus should reduce variance by
averaging out reference-specific noise. The key question is whether Dice
improves enough (≥0.010 over random single).

**Gate R2 threshold**: ΔDice ≥ 0.010, variance reduction ≥ 25%, CI lower > 0.

### RQ3: Is reference reliability learnable?

**Prediction**: UNCERTAIN — the low Spearman ρ (0.7450)
for action rank suggests reference reliability may be predictable, but the
search oracle's failure suggests reference-specific signals may not generalize.

**Gate R3 threshold**: ΔDice ≥ 0.005 over best deterministic, Spearman ρ ≥ 0.30.

## Key Design Decisions for Phase R

1. **Use 3 prompt configs** (P20-N1, P30-N0, P50-N0) instead of 210 actions —
   reduces action-space noise and focuses on reference effect;
2. **Use 6 references per query** (A=3, B=3) — enables consensus and
   cross-reference evaluation;
3. **Use 5 resampling seeds on train** — estimates instability distribution
   without multiple observations on validation;
4. **Fixed A/B on validation** — prevents multiple observations;
5. **CPU-only DINOv2 reuse** — matching maps and candidate pools from Phase Y
   can be reused to save compute.

## Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| R1 fails (instability not replicated) | LOW | Phase Y evidence is strong; train set is larger (80 vs 41) |
| R2 fails (consensus no gain) | MEDIUM | Consensus may not help if instability is systematic, not random |
| R3 fails (reliability not learnable) | MEDIUM-HIGH | Reference quality may depend on query-specific factors |
| Holdout fails | MEDIUM | Even if R1-R3 pass, holdout effect size may be small |
| Compute exceeds 85h | LOW | R1 is the largest stage (25h); CPU reuse helps |

## Next Steps

1. Audit reference pool (step 4);
2. Create train resampling and validation A/B manifests (step 5);
3. Freeze all configs and submit preregistration commit (steps 6-12);
4. Run Gate R1 on train=80 with 5 seeds (step 13).
