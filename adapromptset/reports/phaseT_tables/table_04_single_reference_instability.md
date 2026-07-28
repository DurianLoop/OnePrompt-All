# Table 4 — Single-reference instability (WP1 holdout, k=1)

Reference-induced instability on the 40 MoNuSAC holdout queries under single-reference (k=1) sampling, by prompt config.

| Config | n queries | V_ref | V_ref_frac | mean within-query std | mean best-worst gap | mean MAD_ref | frac queries gap ≥ 0.05 | mean catastrophic rate (k=1) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P20-N1 | 40 | 0.0111 | 0.2627 | 0.0852 | 0.2393 | 0.1070 | 0.9175 | 0.0729 |
| P30-N0 | 40 | 0.0088 | 0.2110 | 0.0772 | 0.2141 | 0.0971 | 0.9025 | 0.0621 |
| P50-N0 | 40 | 0.0052 | 0.1264 | 0.0602 | 0.1668 | 0.0761 | 0.8550 | 0.0550 |

**Notes:**
- All values are at k=1 (single random reference), averaged across 10 seeds.
- MAD_ref = mean absolute pairwise Dice deviation across reference subsets.
- Frac gap ≥ 0.05 is computed at the episode level (40 queries × 10 seeds × 3 configs).
- Gate cond1 requires ≥ 50% of holdout queries to have mean best-worst gap ≥ 0.05; WP1 observed 92.5% (PASS).
- Gate cond2 requires mean within-query reference std ≥ 0.02; WP1 observed 0.0742 (PASS).
- Gate cond3 requires V_ref_frac ≥ 0.10 in any config; all three configs pass.
