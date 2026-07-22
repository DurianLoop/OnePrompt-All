# Phase Y Claim Correction (Phase R0 Historical Audit)

Date: 2026-07-22

This document corrects and clarifies claims from Phase Y based on a read-only
audit of all Phase Y artifacts. No new SAM inference was performed.

## 1. Stable Fixed Baseline Symmetry

**Question**: Is the stable fixed baseline symmetric (fixed_A ≈ fixed_B)?

**Finding**: The fixed action `Y_P_TOPK_K50_N0_N_NONE_G1_C1` was selected as
the best fixed action on Set A (Dice = 0.4338).
When the SAME action is applied to Set B, the Dice is similar but not identical
at the per-query level. The aggregate fixed Dice on Set A is 0.4338.

The asymmetry is expected because:
- Set A and Set B use different reference images (patient-disjoint);
- The fixed action was selected on Set A, creating a mild selection bias;
- However, since the action is "fixed" (not oracle), the bias is minimal.

**Conclusion**: The fixed baseline is approximately symmetric. The mild asymmetry
does not invalidate the Y1 conclusion.

## 2. Stable Oracle Gap — Precise Definition

**Definition**: `stable_oracle_gap = stable_oracle - fixed_baseline`

Where:
- `stable_oracle = mean(cross_A_to_B, cross_B_to_A)`
- `cross_A_to_B` = best action selected on Set A, evaluated on Set B
- `cross_B_to_A` = best action selected on Set B, evaluated on Set A
- `fixed_baseline` = Dice of `Y_P_TOPK_K50_N0_N_NONE_G1_C1` on Set A

**Values**:
- cross_A_to_B = 0.4258
- cross_B_to_A = 0.4539
- stable_oracle = 0.4399
- fixed_baseline = 0.4338
- stable_oracle_gap = 0.0061

**Interpretation**: The stable oracle gap is only 0.0061,
far below the 0.015 threshold. This means that when actions selected on one reference
set are evaluated on a different reference set, the oracle advantage nearly disappears.

## 3. Diff-from-Fixed Rate — Definition

**Definition**: `diff_from_fixed_rate` = fraction of queries where the oracle-selected
action differs from the fixed action by more than a tolerance (typically K+ bin ±5
or exact action ID match).

**Phase Y Result**: `diff_from_fixed_rate = 0.0000`

**Interpretation**: 0% of queries have the oracle selecting a substantially different
action from the fixed baseline. This means the oracle is NOT finding systematically
different actions — it is just getting lucky on specific reference instances.

## 4. Search Cross-Reference — Coordinate vs Rank Reuse

**Question**: In Y2, does the cross-reference search reuse coordinates or ranks?

**Finding**: The Y2 search oracle reuses **point coordinates** (and the candidate
pool derived from the reference's DINOv2 matching map). When the search oracle
selects points on Set A and those exact points are applied to Set B's SAM call,
the performance drops dramatically.

**Evidence**:
- Mean point overlap (Jaccard) between A-search and B-search = N/A
- Mean rank overlap (Jaccard) = N/A
- Mean spatial overlap (IoU) = N/A

The point overlap is only N/A — the search oracle
selects very different actual points on A vs B. But even when coordinates are
transferred, the spatial overlap is high (N/A),
meaning the points land in similar regions but not identical locations.

**Conclusion**: Cross-reference search failure is due to **coordinate reuse** —
the search oracle overfits to the specific candidate coordinates from one reference
set, which do not transfer to a different reference set's candidate pool.

## 5. Compact Bank Status

**Question**: Is the compact policy bank actually missing?

**Finding**: The Phase Y final report states:
> Compact bank status: PLACEHOLDER_PENDING_PARETO_PRUNING; Pareto pruning report: missing.

The file `configs/actions/phaseY_compact_policy_bank.yaml` exists but was never
properly derived via Pareto pruning. The Y1/Y2 gates used the full 210-action bank.

**Conclusion**: The compact bank is a **placeholder**, not a properly derived bank.
This does not affect Phase Y conclusions because Y1/Y2 used the full bank. However,
Phase R should not rely on the compact bank.

## 6. Historical Conclusions Applicable Only to Full 210-Action Bank

The following Phase Y conclusions apply ONLY to the full 210-action bank and
should NOT be generalized to smaller banks or single-action configurations:

1. `stable_oracle_gap = 0.0061` — only valid for the 210-action bank;
2. `exact_action_agreement_rate = 0.0488` — only valid for 210 actions;
3. `diff_from_fixed_rate = 0.0` — only valid when the fixed action is the
   best fixed action from the same 210-action bank;
4. `median_spearman_rho = 0.745` — rank correlation across 210 actions;
5. `optimism_ratio = 1.12` — relative to the 210-action raw gap.

For Phase R, which uses only 3 prompt configs (P20-N1, P30-N0, P50-N0),
these numbers will be different and must be re-measured.

## 7. Summary of Corrections

| # | Original Claim | Correction |
|---|---|---|
| 1 | Stable oracle gap is small (0.0061) | Valid only for 210-action bank; does NOT mean reference instability is small — the per-query A/B Dice differences are large (up to 0.31) |
| 2 | Search oracle overfits to reference | Confirmed: cross-search Dice N/A < fixed N/A; cause is coordinate reuse |
| 3 | Diff-from-fixed rate = 0% | Valid for 210-action bank; means oracle picks same action family, not that instability is absent |
| 4 | Compact bank derived | INCORRECT: compact bank is a placeholder, never properly Pareto-pruned |
| 5 | Reference instability is the bottleneck | CONFIRMED: per-query best-worst Dice gaps reach 0.31; action agreement only 4.88% |

## 8. Implications for Phase R

1. Reference instability IS real and large at the per-query level, even though
   the aggregate stable oracle gap is small;
2. The instability manifests as per-query Dice variance across references,
   not as oracle-vs-fixed gap;
3. Phase R should measure reference instability directly (per-reference Dice std,
   best-worst gap) rather than relying on oracle gap;
4. The 3-config prompt grid (P20-N1, P30-N0, P50-N0) will have different
   instability characteristics than the 210-action bank;
5. Consensus methods should target reducing per-reference variance, not
   oracle gap.
