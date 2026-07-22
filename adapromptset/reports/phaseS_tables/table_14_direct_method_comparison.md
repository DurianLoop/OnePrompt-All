# Table 14 — Direct Method Comparison (Track D)

**Status:** `START_CONDITION_NOT_FULLY_MET` — Track D **NOT STARTED**.

## 1. Why Track D Was Not Started

Directive §10 requires Track D to begin only if all 4 start conditions are met. Assessment:

| # | Start condition | Status | Evidence |
|---|---|---|---|
| 1 | Track A confirmed reference instability | MET ✓ | Gate A 7/7 PASSED, V_ref_frac 0.2456 (P30-N0) |
| 2 | Track B/C cannot improve stability | MET ✓ | Both NO_GAIN (cond1+cond2 fail) |
| 3 | Raw matching maps have stable signal but point prompt conversion loses significantly | **NOT MET ✗** | SAM refinement HELPS: consensus Dice 0.48 vs raw matching map Dice <0.3 — conversion GAINS, not loses |
| 4 | SAM refinement gains unstable across references | MET ✓ | Best-worst gap 80%+ of queries exceed 0.05 Dice |

**Decision:** Track D not started (condition 3 not met — the SAM point-prompt interface is actually *helping* relative to the raw matching map, even though it is the bottleneck for cross-reference stability).

## 2. What Track D Would Have Compared (Structure Only — NOT RUN)

If start conditions had been met, Track D would have implemented and compared direct reference-conditioned segmentation methods. The planned comparison structure:

| Method | Description | Reference-conditioning mechanism |
|---|---|---|
| D1_direct_feature_concat | Concatenate reference features into the SAM decoder | Feature concatenation |
| D2_cross_attention | Cross-attention between reference and query features | Cross-attention module |
| D3_reference_token | Reference as a learned token in the SAM mask decoder | Token-conditioned decoding |
| D4_fiery_style | Fiery-style direct regression from reference | Direct regression |
| D5_direct_mask_propagation | Propagate reference mask directly to query | Mask propagation |

**None of these methods were implemented or evaluated.** Track D status: NOT STARTED, 0 SAM calls, 0 GPU-hours.

## 3. Headline Comparison (BLOCKED — not run)

| Method | Mean Dice | CVaR_0.1 | CFR | Status |
|---|---|---|---|---|
| B2_mean (baseline) | 0.4756 | 0.0738 | 0.0732 | available |
| Track B weighted | 0.4732 | 0.0768 | 0.0732 | available |
| Track C best (α=0,β=0,γ=0) | 0.4779 | 0.0707 | 0.0732 | available |
| Track D1–D5 | **BLOCKED — not run** | — | — | not started |

## 4. Gate D Conditions (NOT EVALUATED)

| # | Condition | Threshold | Result | Status |
|---|---|---|---|---|
| D1 | Mean Dice improvement vs B2_mean ≥ 0.01 | +0.01 | not evaluated | BLOCKED |
| D2 | CVaR improvement ≥ 0.02 | +0.02 | not evaluated | BLOCKED |
| D3 | CFR reduction ≥ 30% | 30% | not evaluated | BLOCKED |
| D4 | Cross-config consistency (3/3) | 3/3 | not evaluated | BLOCKED |

**Gate D decision:** NOT EVALUATED (Track D not started).

## 5. Implication for Paper

- The benchmark + failure-boundary paper does **not** include a Track D method contribution.
- The negative finding (Track B/C NO_GAIN + Track D not started) is reported honestly.
- The bottleneck is identified at the **SAM point-prompt interface** (Track A audit) — the raw matching map (Dice <0.3) is too weak to support direct conditioning, while SAM refinement (Dice 0.48) is good but unstable across references.
- This "raw matching map vs SAM-refined" gap is documented in the failure taxonomy (Table 19).

## 6. Reference

- `reports/phaseS_track_selection.json`
- `reports/phaseS_track_selection.md`
- `reports/phaseS_decision_tree.md`
- `reports/phaseS_trackA_benchmark/benchmark_summary.md`
- `reports/phase4_read_only_audit/prompt_interface_failure_map.csv`
