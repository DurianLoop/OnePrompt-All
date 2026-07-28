# Table 6 — Mean / median / robust consensus comparison

Consensus aggregation strategies compared under the reference-resampling protocol. B1 and B2 are confirmed on the WP1 holdout; B3-B7 are Phase S frozen-model values reported for completeness.

| Aggregation | Description | MoNuSAC holdout Dice (avg over configs) | Source |
| --- | --- | --- | --- |
| B1 Random Single | Pick one reference at random (k=1). | 0.4762 | WP1 holdout (k=1, avg P20/P30/P50) |
| B2 Mean Consensus | Pixel-wise mean of k reference masks (k=6). | 0.5058 | WP1 holdout (k=6, avg P20/P30/P50) |
| B3 Median Consensus | Pixel-wise median of k reference masks. | 0.4691 (Phase S val) | Phase S val (frozen, no holdout re-run) |
| B4 Trimmed Mean | Trim extreme per-pixel values before averaging. | 0.4719 (Phase S val) | Phase S val (frozen, no holdout re-run) |
| B5 Disagreement-Penalized | Down-weight references whose prompt disagrees with the candidate set. | 0.4756 (Phase S val) | Phase S val (frozen, no holdout re-run) |
| B6 Quantile Consensus | Per-pixel upper-quantile vote. | 0.4732 (Phase S val) | Phase S val (frozen, no holdout re-run) |
| B7 Phase S Best Learned Weighting | GBM reference-utility-weighted consensus (Phase S frozen predictor). | 0.4732 (Phase S val) | Phase S val (frozen, no holdout re-run) |

**Notes:**
- B1 and B2 are the two aggregations actually run on the WP1 holdout; the holdout is one-shot so B3-B7 are not re-run there.
- Phase S values (B3-B7) come from the Phase S validation set and are reported for reference; B7 uses the Phase S frozen GBM utility predictor (val Spearman ≈ 0.478) and does NOT beat B2 mean consensus.
- WP1 holdout B2 mean consensus Dice (k=6, avg over configs) = 0.5058 vs B1 random single (k=1, avg) = 0.4792; consensus gain = +0.0266 (paired bootstrap 95% CI lower = 0.0140).
- Phase S finding: learned weighting (B7) is dominated by uniform mean (B2) by ≈ −0.0024 Dice, which is why WP1 only confirms B1 vs B2 on holdout.
