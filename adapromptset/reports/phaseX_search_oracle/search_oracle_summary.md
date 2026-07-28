# Phase X: GT-Guided Search Oracle Summary

Date: 2026-07-20
Stage: search_oracle (Gate X5)
Episodes: 10
Positive limit: 100
K+ start: 20
K- start: 1
Negative selector: N_BOTTOMK
B_point: 100
B_call: 1
Device: cpu
SAM inference run: True
Elapsed seconds: 966.0
GPU-hours: 0.0

## Search variants (implemented)

- greedy_add_positives
- greedy_remove_positives
- greedy_swap_positives
- greedy_add_negatives

## Deferred variants

- beam_search (CPU budget; beam_width>1 would multiply SAM calls)

## Gate X5 Status

**SEARCH_ORACLE_EXCEEDS_BANK_ORACLE**

search=0.7256 - bank=0.6119 = 0.1137 >= 0.01

## Aggregate metrics

- Mean baseline Dice (best fixed): 0.5411
- Mean search oracle Dice: 0.7256
- Mean search gap: 0.1845
- Mean SAM calls per episode: 558.5
- Win/tie/loss (search vs baseline): {'win': 10, 'tie': 0, 'loss': 0}
- Bank oracle mean Dice (A_large from Gate X4): 0.6119
- Best fixed mean Dice (A_large from Gate X4): 0.5411
- Search - bank oracle: 0.1137
- Search - best fixed: 0.1845

## Per-episode results

| Episode | Baseline Dice | Search Dice | Gap | Final K+ | Final K- | SAM calls |
|---|---|---|---|---|---|---|
| monusac_val__train_111__to__train_004 | 0.3583 | 0.6676 | +0.3093 | 18 | 1 | 735 |
| monusac_val__train_190__to__train_004 | 0.6326 | 0.6891 | +0.0565 | 25 | 1 | 588 |
| monusac_val__train_132__to__train_004 | 0.4323 | 0.6948 | +0.2625 | 26 | 1 | 563 |
| monusac_val__train_204__to__train_004 | 0.4626 | 0.5850 | +0.1224 | 20 | 1 | 460 |
| monusac_val__train_200__to__train_004 | 0.3731 | 0.5456 | +0.1726 | 27 | 5 | 846 |
| monusac_val__train_116__to__train_005 | 0.5974 | 0.7890 | +0.1916 | 17 | 2 | 487 |
| monusac_val__train_123__to__train_005 | 0.7354 | 0.8312 | +0.0957 | 15 | 2 | 462 |
| monusac_val__train_207__to__train_005 | 0.4065 | 0.8270 | +0.4205 | 17 | 1 | 752 |
| monusac_val__train_113__to__train_005 | 0.7512 | 0.8244 | +0.0731 | 16 | 1 | 308 |
| monusac_val__train_017__to__train_005 | 0.6619 | 0.8027 | +0.1408 | 18 | 1 | 384 |

## Interpretation

- `SEARCH_ORACLE_EXCEEDS_BANK_ORACLE`: search oracle - bank oracle >= 0.01 → the finite bank misses meaningful prompt structures.
- `SEARCH_ORACLE_CLOSE_TO_BANK_ORACLE`: search oracle - bank oracle < 0.01 → the finite bank captures most of the achievable performance.
- `SEARCH_ORACLE_ABSOLUTE_PERFORMANCE_LOW`: search oracle < 0.4 → the limitation is the prompt interface or candidate generator, not the action bank.

Note: this search oracle uses GT-guided greedy add/remove/swap over the frozen candidate pool. It is **diagnostic only** (section 10): search outputs must not become automatic actions without a separate train-only abstraction step.