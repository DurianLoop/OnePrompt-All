| Method | P20-N1 | P30-N0 | P50-N0 | Mean(across cfg) |
|---|---|---|---|---|
| B1_random | 0.3900 | 0.4297 | 0.4444 | 0.4213 |
| B2_mean | 0.4484 | 0.4756 | 0.4837 | 0.4692 |
| B3_median | 0.4249 | 0.4408 | 0.4654 | 0.4437 |
| B4_trimmed | 0.4191 | 0.4578 | 0.4823 | 0.4531 |
| B5_disagreement | 0.4235 | 0.4631 | 0.4794 | 0.4553 |
| B6_quantile | 0.4550 | 0.4544 | 0.4769 | 0.4621 |

Source: phaseR_gateR2/val_consensus_metrics.csv (reference_scope=all6). B5_disagreement uses beta=0.25; B6_quantile uses tau=0.83 (train-selected). Best consensus: B2_mean on P30-N0 (mean Dice 0.4756).
