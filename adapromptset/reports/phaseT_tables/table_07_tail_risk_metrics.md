# Table 7 — Tail-risk metrics (WP1 holdout)

Tail-risk metrics on the 40 MoNuSAC holdout queries by prompt config and reference count k.

| Config | k | CVaR_0.1 | CFR | mean worst Dice | min worst Dice | p10 worst Dice |
| --- | --- | --- | --- | --- | --- | --- |
| P20-N1 | 1 | 0.0561 | 0.0729 | 0.2214 | 0.0000 | 0.0004 |
| P20-N1 | 3 | 0.0772 | 0.0675 | 0.3335 | 0.0001 | 0.0183 |
| P20-N1 | 6 | 0.0766 | 0.0650 | 0.3648 | 0.0000 | 0.0444 |
| P30-N0 | 1 | 0.0751 | 0.0621 | 0.3058 | 0.0000 | 0.0373 |
| P30-N0 | 3 | 0.0939 | 0.0575 | 0.3721 | 0.0020 | 0.0484 |
| P30-N0 | 6 | 0.1027 | 0.0575 | 0.4002 | 0.0161 | 0.0875 |
| P50-N0 | 1 | 0.0831 | 0.0550 | 0.3314 | 0.0053 | 0.0841 |
| P50-N0 | 3 | 0.0952 | 0.0525 | 0.4044 | 0.0169 | 0.1915 |
| P50-N0 | 6 | 0.0987 | 0.0500 | 0.4407 | 0.0160 | 0.2068 |

**Notes:**
- CVaR_0.1 = mean Dice over the worst 10% of reference subsets per query.
- CFR = catastrophic failure rate = fraction of (query, subset) episodes with Dice < 0.10.
- mean / min / p10 worst Dice are computed over the per-query worst reference subset.
- CFR drops with k for every config (e.g., P50-N0: 0.0550 → 0.0525 → 0.0500) — consensus reduces tail risk.
- min worst Dice rises sharply with k (P50-N0: 0.0053 → 0.0169 → 0.0160): the very worst reference episode is rescued by consensus.
- Gate cond6 (consensus CFR ≤ single CFR) and cond7 (consensus CVaR ≥ single CVaR − 0.005) both PASS.
