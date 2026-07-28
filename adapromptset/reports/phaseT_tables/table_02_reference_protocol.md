# Table 2 — Reference resampling protocol

Frozen reference-resampling protocol applied uniformly across all datasets and work packages.

| Component | MoNuSAC (WP1) | MoNuSeg (WP2) | IDRID (WP2) |
| --- | --- | --- | --- |
| Matcher | DINOv2-L (frozen) | DINOv2-L (frozen) | DINOv2-L (frozen) |
| Segmenter | SAM-B (frozen) | SAM-B (frozen) | SAM-B (frozen) |
| SAM calls per consensus | 1 | 1 | 1 |
| Reference counts k | {1, 3, 6} | {1, 3, 6} | {1, 3, 6} |
| References per query | 6 (patient-disjoint) | 6 (patient-disjoint) | 6 (patient-disjoint) |
| Resampling seeds | 10 | 5 | 5 |
| Prompt configs | P20-N1, P30-N0, P50-N0 | P20-N1, P30-N0, P50-N0 | P20-N1, P30-N0, P50-N0 |
| Aggregations evaluated | B1 random single, B2 mean consensus (+ B3–B7 in Phase S) | B1 random single, B2 mean consensus | B1 random single, B2 mean consensus |
| Bootstrap | 10,000 iters, 95% CI (by query) | 10,000 iters, 95% CI (by query) | 10,000 iters, 95% CI (by query) |
| Catastrophic threshold | Dice < 0.10 | Dice < 0.10 | Dice < 0.10 |
| Statistical unit | query / case | query / case | query / case |
| Reference-query overlap | Forbidden (verified 0/2400 on holdout) | Forbidden (case_id check) | Forbidden (case_id check) |

**Notes:**
- All components are frozen before any holdout inference; MoNuSAC holdout is run exactly once.
- B3 median, B4 trimmed mean, B5 disagreement-penalized, B6 quantile, B7 Phase S learned weighting were evaluated in Phase S and are reported in Table 6 from the Phase S artifacts; WP1/WP2 focus on B1 vs B2.
- Patient-disjoint reference assignment is verified by an explicit case_id check before every resampling seed (WP1: 0 violations / 2400 reference assignments across 10 seeds).
