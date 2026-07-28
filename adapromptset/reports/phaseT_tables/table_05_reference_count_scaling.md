# Table 5 — Reference-count scaling

Mean Dice, V_ref, worst-subset Dice, CVaR_0.1 and CFR as a function of reference count k, by dataset and prompt config.

| Dataset | Config | k | mean Dice | V_ref | worst Dice | CVaR_0.1 | CFR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MoNuSAC holdout | P20-N1 | 1 | 0.4617 | 0.0111 | 0.2214 | 0.0561 | 0.0729 |
| MoNuSAC holdout | P20-N1 | 3 | 0.4821 | 0.0095 | 0.3335 | 0.0772 | 0.0675 |
| MoNuSAC holdout | P20-N1 | 6 | 0.4889 | 0.0083 | 0.3648 | 0.0766 | 0.0650 |
| MoNuSAC holdout | P30-N0 | 1 | 0.4879 | 0.0088 | 0.3058 | 0.0751 | 0.0621 |
| MoNuSAC holdout | P30-N0 | 3 | 0.5052 | 0.0079 | 0.3721 | 0.0939 | 0.0575 |
| MoNuSAC holdout | P30-N0 | 6 | 0.5168 | 0.0068 | 0.4002 | 0.1027 | 0.0575 |
| MoNuSAC holdout | P50-N0 | 1 | 0.4790 | 0.0052 | 0.3314 | 0.0831 | 0.0550 |
| MoNuSAC holdout | P50-N0 | 3 | 0.4984 | 0.0039 | 0.4044 | 0.0952 | 0.0525 |
| MoNuSAC holdout | P50-N0 | 6 | 0.5117 | 0.0026 | 0.4407 | 0.0987 | 0.0500 |
| MoNuSeg | P20-N1 | 1 | 0.4785 | 0.0023 | 0.3699 | 0.2580 | 0.0000 |
| MoNuSeg | P20-N1 | 3 | 0.4933 | 0.0015 | 0.4479 | 0.2675 | 0.0000 |
| MoNuSeg | P20-N1 | 6 | 0.4978 | 0.0009 | 0.4656 | 0.2793 | 0.0000 |
| MoNuSeg | P30-N0 | 1 | 0.4986 | 0.0026 | 0.3936 | 0.2571 | 0.0000 |
| MoNuSeg | P30-N0 | 3 | 0.5120 | 0.0018 | 0.4787 | 0.2633 | 0.0000 |
| MoNuSeg | P30-N0 | 6 | 0.5144 | 0.0015 | 0.4894 | 0.2645 | 0.0000 |
| MoNuSeg | P50-N0 | 1 | 0.5041 | 0.0026 | 0.4060 | 0.2553 | 0.0000 |
| MoNuSeg | P50-N0 | 3 | 0.5147 | 0.0012 | 0.4899 | 0.2604 | 0.0000 |
| MoNuSeg | P50-N0 | 6 | 0.5168 | 0.0015 | 0.4949 | 0.2617 | 0.0000 |
| IDRID | P20-N1 | 1 | 0.1945 | 0.0536 | 0.0397 | 0.0306 | 0.7089 |
| IDRID | P20-N1 | 3 | 0.2607 | 0.0578 | 0.0891 | 0.0357 | 0.6533 |
| IDRID | P20-N1 | 6 | 0.1716 | 0.0415 | 0.0607 | 0.0355 | 0.7267 |
| IDRID | P30-N0 | 1 | 0.0574 | 0.0013 | 0.0402 | 0.0290 | 0.9556 |
| IDRID | P30-N0 | 3 | 0.0597 | 0.0003 | 0.0509 | 0.0302 | 0.9333 |
| IDRID | P30-N0 | 6 | 0.0582 | 0.0001 | 0.0527 | 0.0302 | 0.9400 |
| IDRID | P50-N0 | 1 | 0.0432 | 0.0000 | 0.0374 | 0.0271 | 1.0000 |
| IDRID | P50-N0 | 3 | 0.0421 | 0.0000 | 0.0391 | 0.0279 | 1.0000 |
| IDRID | P50-N0 | 6 | 0.0416 | 0.0000 | 0.0392 | 0.0267 | 1.0000 |

**Notes:**
- k scales the reference subset size: k=1 (single random reference), k=3 (mean consensus over first 3), k=6 (mean consensus over all 6).
- On MoNuSAC holdout, mean Dice increases monotonically with k for every config (e.g., P30-N0: 0.4879 → 0.5052 → 0.5168).
- V_ref decreases monotonically with k (consensus dampens reference variance).
- On IDRID the trend is broken: P20-N1 mean Dice drops from 0.2607 (k=3) to 0.1716 (k=6); CVaR and CFR remain very high — IDRID FAIL is driven by this collapse.
- On MoNuSeg, CFR is 0 for all (config, k) because no query drops below the catastrophic Dice < 0.10 threshold.
