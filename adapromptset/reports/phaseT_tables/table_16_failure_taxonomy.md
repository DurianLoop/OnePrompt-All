# Table 16 — Failure taxonomy (WP1 holdout)

Data-driven failure taxonomy on the WP1 holdout episodes, with the appropriate mitigation strategy per failure mode.

| Failure mode | n episodes | share of episodes | Mitigation |
| --- | --- | --- | --- |
| Stable (gap < 0.05, Dice ≥ 0.5) | 40 | 3.33% | Mean consensus (B2) is sufficient. |
| Low-Dice but stable (gap < 0.05, Dice < 0.5) | 30 | 2.50% | Mean consensus does not help; needs better prompt construction or segmenter (out of scope). |
| Reference-unstable (gap ≥ 0.05, Dice ≥ 0.5) | 589 | 49.08% | Mean consensus (B2) raises Dice and lowers CFR — primary beneficiary of the proposed protocol. |
| Reference-unstable + low-Dice (gap ≥ 0.05, Dice < 0.5) | 475 | 39.58% | Mean consensus helps partially; M2 selective prediction flags these for abstention. |
| Catastrophic (CFR_k1 ≥ 0.5) | 66 | 5.50% | M2 selective prediction (catastrophic-trained) abstains at this mode; M1 cannot rescue. |

**Notes:**
- Episodes are (query, seed, config) triples at k=1 on the WP1 holdout; total n = 1200.
- CFR_k1 ≥ 0.5 means at least 3 of the 6 single-reference subsets in the episode fall below the catastrophic Dice < 0.10 threshold.
- The taxonomy is data-driven (no manual labeling); thresholds are pre-registered (gap ≥ 0.05, Dice ≥ 0.5, CFR ≥ 0.5).
- Reference-unstable + low-Dice episodes are the hardest case: mean consensus improves them but they remain below Dice = 0.5, and M2 abstention is the appropriate action.
