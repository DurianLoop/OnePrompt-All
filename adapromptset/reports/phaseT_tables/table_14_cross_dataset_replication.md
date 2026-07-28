# Table 14 — Cross-dataset replication (WP2)

Side-by-side comparison of the WP1 holdout (MoNuSAC) and the two WP2 cross-dataset replications (MoNuSeg, IDRID) under the unified protocol.

| Indicator | MoNuSAC holdout (WP1) | MoNuSeg (WP2) | IDRID (WP2) |
| --- | --- | --- | --- |
| n queries | 40 | 30 | 30 |
| resampling seeds | 10 | 5 | 5 |
| frac queries best-worst gap ≥ 0.05 | 0.9250 | 0.8000 | 0.6333 |
| mean within-query ref std | 0.0742 | 0.0388 | 0.0605 |
| mean B1 single Dice (k=1) | 0.4792 | 0.4948 | 0.1022 |
| mean consensus Dice (k=6) | 0.5058 | 0.5097 | 0.0904 |
| consensus gain | 0.0266 | 0.0149 | -0.0118 |
| bootstrap CI lower | 0.0140 | 0.0069 | -0.0318 |
| CFR B1 single (k=1) | 0.0633 | 0.0000 | 0.8956 |
| CFR consensus (k=6) | 0.0575 | 0.0000 | 0.8889 |
| CVaR B1 single (k=1) | 0.0751 | 0.2583 | 0.0287 |
| CVaR consensus (k=6) | 0.0919 | 0.2679 | 0.0291 |
| mean J_pred B1 single | 0.6490 | 0.7327 | 0.7843 |
| mean J_pred consensus | 0.7190 | 0.8296 | 0.8539 |
| best disagreement Spearman ρ | 0.7241 | 0.8266 | 0.9519 |
| Directional pass | 10/10 (WP1) | 5/5 | 4/5 |
| Strong pass | — | 4/4 | 2/4 |
| Decision | PHASET_WP1_MONUSAC_HOLDOUT_CONFIRMED | PHASET_DATASET_STRONG_SUPPORT | PHASET_DATASET_FAIL |

**Notes:**
- MoNuSAC holdout = WP1 confirmatory benchmark (10 conditions, 40 queries).
- MoNuSeg and IDRID = WP2 cross-dataset replication (5 directional + 4 strong conditions each, 30 queries each).
- MoNuSeg: STRONG_SUPPORT — consensus gain = +0.0149, CI lower = +0.0069, all 4/4 strong conditions pass; consistent with WP1.
- IDRID: FAIL — directional d3 fails (consensus Dice −0.0118 < single) and strong s1/s2 fail (gain negative, CI lower negative). IDRID lesion intent EX is too hard for the frozen DINOv2+SAM-B stack (mean Dice ≈ 0.10, CFR ≈ 0.90).
- §6.4 minimum: MoNuSAC holdout confirmed + at least 2 additional datasets with directional support. MoNuSeg (STRONG_SUPPORT) + IDRID (4/5 directional pass) jointly satisfy the minimum despite IDRID's overall FAIL.
- The §6.4 prohibition on pooling queries across datasets for aggregate significance tests is respected: each dataset is evaluated separately.
