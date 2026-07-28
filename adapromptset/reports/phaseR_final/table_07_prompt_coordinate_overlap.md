| Source / Config | Mean prompt-overlap Jaccard | Std | N |
|---|---|---|---|
| P20-N1 | 0.1862 | 0.0816 | 400 |
| P30-N0 | 0.2105 | 0.0851 | 400 |
| P50-N0 | 0.2496 | 0.0890 | 400 |
| R2 val (A vs B, B2_mean) | 0.3027 | 0.2336 | 123 |
| R1 ALL | 0.2155 | 0.0892 | 1200 |

R1: per_query_instability.csv column prompt_overlap_mean (positive point set Jaccard across 6 refs). R2: val_cross_set_stability.csv column prompt_overlap_AB (prompt overlap between reference sets A and B).
