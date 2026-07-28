# Table 3 — Correct variance decomposition

Within-query reference variance (V_ref) versus cross-query variance (V_query) and the reference-fraction V_ref_frac, by dataset, prompt config and reference count k.

| Dataset | Config | k | V_ref | V_query | V_ref_frac | mean within-query std | mean best-worst gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MoNuSAC holdout | P20-N1 | 1 | 0.0111 | 0.0311 | 0.2627 | 0.0852 | 0.2393 |
| MoNuSAC holdout | P20-N1 | 3 | 0.0095 | 0.0327 | 0.2252 | 0.0802 | 0.2625 |
| MoNuSAC holdout | P20-N1 | 6 | 0.0083 | 0.0356 | 0.1891 | 0.0699 | 0.2252 |
| MoNuSAC holdout | P30-N0 | 1 | 0.0088 | 0.0330 | 0.2110 | 0.0772 | 0.2141 |
| MoNuSAC holdout | P30-N0 | 3 | 0.0079 | 0.0351 | 0.1843 | 0.0727 | 0.2302 |
| MoNuSAC holdout | P30-N0 | 6 | 0.0068 | 0.0359 | 0.1590 | 0.0619 | 0.1965 |
| MoNuSAC holdout | P50-N0 | 1 | 0.0052 | 0.0359 | 0.1264 | 0.0602 | 0.1668 |
| MoNuSAC holdout | P50-N0 | 3 | 0.0039 | 0.0380 | 0.0931 | 0.0531 | 0.1695 |
| MoNuSAC holdout | P50-N0 | 6 | 0.0026 | 0.0395 | 0.0613 | 0.0404 | 0.1324 |
| MoNuSeg | P20-N1 | 1 | 0.0023 | 0.0121 | 0.1602 | 0.0409 | 0.1177 |
| MoNuSeg | P20-N1 | 3 | 0.0015 | 0.0135 | 0.0976 | 0.0310 | 0.0829 |
| MoNuSeg | P20-N1 | 6 | 0.0009 | 0.0124 | 0.0700 | 0.0237 | 0.0643 |
| MoNuSeg | P30-N0 | 1 | 0.0026 | 0.0142 | 0.1547 | 0.0387 | 0.1109 |
| MoNuSeg | P30-N0 | 3 | 0.0018 | 0.0153 | 0.1034 | 0.0266 | 0.0681 |
| MoNuSeg | P30-N0 | 6 | 0.0015 | 0.0160 | 0.0872 | 0.0213 | 0.0550 |
| MoNuSeg | P50-N0 | 1 | 0.0026 | 0.0151 | 0.1487 | 0.0369 | 0.1057 |
| MoNuSeg | P50-N0 | 3 | 0.0012 | 0.0168 | 0.0687 | 0.0196 | 0.0488 |
| MoNuSeg | P50-N0 | 6 | 0.0015 | 0.0168 | 0.0797 | 0.0197 | 0.0498 |
| IDRID | P20-N1 | 1 | 0.0536 | 0.0247 | 0.6844 | 0.1635 | 0.4080 |
| IDRID | P20-N1 | 3 | 0.0578 | 0.0606 | 0.4883 | 0.1691 | 0.4077 |
| IDRID | P20-N1 | 6 | 0.0415 | 0.0285 | 0.5928 | 0.1215 | 0.2837 |
| IDRID | P30-N0 | 1 | 0.0013 | 0.0006 | 0.6991 | 0.0143 | 0.0406 |
| IDRID | P30-N0 | 3 | 0.0003 | 0.0009 | 0.2426 | 0.0088 | 0.0231 |
| IDRID | P30-N0 | 6 | 0.0001 | 0.0008 | 0.1003 | 0.0047 | 0.0128 |
| IDRID | P50-N0 | 1 | 0.0000 | 0.0001 | 0.1826 | 0.0038 | 0.0099 |
| IDRID | P50-N0 | 3 | 0.0000 | 0.0001 | 0.1047 | 0.0025 | 0.0064 |
| IDRID | P50-N0 | 6 | 0.0000 | 0.0001 | 0.0636 | 0.0018 | 0.0046 |

**Notes:**
- V_ref = (1/N) Σ_q Var_S[D(q,S)]  — within-query reference variance.
- V_query = Var_q[E_S[D(q,S)]]  — cross-query variance of the per-query mean.
- V_ref_frac = V_ref / (V_ref + V_query) — share of total variance attributable to reference choice (the headline instability metric).
- Correct decomposition uses the query as the statistical unit and aggregates reference episodes within each query (no pseudo-replication).
- WP1 holdout V_ref_frac (k=1) per config: P20-N1=0.2627, P30-N0=0.2110, P50-N0=0.1264 (all ≥ 0.10 threshold).
- IDRID P30-N0/P50-N0 show very low V_ref_frac because per-query Dice is near-zero on this lesion intent (EX), collapsing within-query variance.
