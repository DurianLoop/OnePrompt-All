# Phase Y Gate Y2: Search Oracle Stability

- **Queries evaluated**: 41
- **Fixed action baseline**: `Y_P_TOPK_K50_N0_N_NONE_G1_C1` (Dice=0.4338)
- **Raw search Dice**: 0.5947
- **Cross-reference search Dice**: 0.4105
- **Raw search gap**: 0.1609
- **Search optimism**: 0.1843 (ratio=114.49%)
- **Bootstrap 95% CI**: [-0.0890, 0.0403]
- **Median |K_A - K_B|**: 6.0
- **Median cross Dice diff**: 0.0292
- **A-search beats fixed on B**: 26.8%
- **Mean point overlap (Jaccard)**: 0.3322
- **Mean rank overlap (Jaccard)**: 0.5577
- **Mean spatial overlap (IoU)**: 0.8979
- **Mean prompt area bias**: 0.1406

## Gate conditions

- [FAIL] `cond1_cross_ge_fixed_plus_0p03`
- [FAIL] `cond2_bootstrap_ci_lo_gt_0`
- [FAIL] `cond3_optimism_le_50pct_gap`
- [PASS] `cond4_k_stable_or_cross_dice_stable`
- [FAIL] `cond5_70pct_A_beats_fixed_on_B`

## Decision: `PHASEY_GATE_Y2_SEARCH_ORACLE_OVERFIT`