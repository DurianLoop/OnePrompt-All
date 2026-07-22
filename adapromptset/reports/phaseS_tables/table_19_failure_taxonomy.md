# Table 19 — Failure Taxonomy

## 1. Track A — Catastrophic Failures (Dice < 0.1)

**Source:** `reports/phaseS_trackA_benchmark/cvar_metrics.csv`

| Config | k | CFR | # catastrophic (query, ref) pairs | Notes |
|---|---|---|---|---|
| P20-N1 | 1 | 17.07% | ~70 of ~410 | worst |
| P20-N1 | 6 | 10.98% | ~45 of ~410 | consensus reduces CFR by 36% |
| P30-N0 | 1 | 14.63% | ~60 of ~410 | |
| P30-N0 | 6 | 7.32% | ~30 of ~410 | consensus reduces CFR by 50% |
| P50-N0 | 1 | 9.76% | ~40 of ~410 | |
| P50-N0 | 6 | 4.88% | ~20 of ~410 | best |

**Headline CFR (P30-N0, k=6, B2_mean): 0.0732** — even with k=6 consensus, 7.32% of (query, reference) pairs produce Dice < 0.1.

## 2. Track A — Failure Modes (Categorical)

| Failure mode | Description | % of catastrophic | Diagnostic feature |
|---|---|---|---|
| F-empty | SAM refines the prompt points into an empty mask (Dice=0) | ~45% | prediction_iou_min ≈ 0 |
| F-wrong-region | SAM latches onto the wrong structure | ~30% | prediction_iou_min > 0 but low Dice |
| F-oversegment | SAM produces a mask that oversegments (high recall, low precision) | ~15% | high recall, low precision |
| F-undersegment | SAM produces a mask that undersegments (low recall, high precision) | ~10% | low recall, high precision |

The dominant failure mode is **F-empty** — SAM refining the prompt into an empty mask. This is concentrated in queries where the matcher (driven by the reference image) proposes prompt points that SAM does not refine into a foreground region.

## 3. Track C — LOO Instability

**Source:** `reports/phaseS_trackC/loo_stability.csv`

| Statistic | Value |
|---|---|
| # LOO drops > 0.05 | **7** |
| LOO threshold (cond6) | ≤ 7 drops |
| Status | **FAIL (cond6)** — at the threshold |

7 LOO drops exceed 0.05 Dice when a single reference is removed. This means that for 7 reference sets, removing one reference shifts the consensus Dice by > 0.05 — the consensus is **not robust** to single-reference removal.

## 4. Track B — "Compression Without Accuracy Gain" Pattern

| Effect | Track B weighted vs uniform | Interpretation |
|---|---|---|
| Pairwise dev | -60.09% | predictions more similar across references |
| Mean ref std | -2.83% | variance barely reduced |
| Mean Dice | -0.0024 | no accuracy gain |
| CFR | 0.0000 | no catastrophic failure reduction |
| CVaR_0.1 | +0.003 | tail slightly worse |

**Failure mode:** The reliability weighting compresses the prediction distribution toward a centroid that is NOT a better segmentation. The utility signal is real (Spearman 0.478) but does not translate into Dice gain.

## 5. Prompt Interface Bottleneck (Headline Failure Boundary)

**Source:** `reports/phaseS_trackA_metric_audit/method_cost_decomposition.md`, `reports/phase4_read_only_audit/prompt_interface_failure_map.csv`

| Component | Dice | Notes |
|---|---|---|
| Raw matching map | <0.3 | too weak to support direct conditioning |
| SAM-refined (single ref, k=1) | ~0.42 | SAM helps (+0.12) |
| SAM-refined (consensus, k=6, B2_mean) | 0.4756 | consensus helps further |
| Best achievable (Track B/C) | 0.4732–0.4779 | no further gain |

**The bottleneck is the SAM point-prompt interface:**
- The matcher produces prompt points from the reference image.
- SAM refines these points into a mask.
- The refined mask is **better** than the raw matching map (0.48 vs <0.3) — so Track D condition 3 is NOT met (refinement helps, not loses).
- BUT the refined mask is **unstable across references** — different references produce different prompt points, which SAM refines into different masks.
- Neither reliability weighting (Track B) nor set-conditioned prompting (Track C) can fix this — the instability is **propagated through** the SAM interface.

## 6. Cross-Reference Instability Taxonomy

| Instability source | Manifestation | Mitigation attempted | Result |
|---|---|---|---|
| Reference image choice | Different matcher outputs → different prompt points | Reliability weighting (Track B) | NO_GAIN |
| Reference subset (k references) | Different consensus centroids | Set-conditioned prompting (Track C) | NO_GAIN |
| SAM refinement stochasticity | Different mask from same prompt | none | (out of scope) |

## 7. Why Track D Was Not Started

Track D condition 3 ("raw matching maps have stable signal but point prompt conversion loses significantly") was NOT met because:
- Raw matching map Dice < 0.3 — too weak.
- SAM-refined Dice 0.48 — substantially better.
- Conversion GAINS, not loses.

The relevant bottleneck is the **cross-reference propagation of instability through the SAM point-prompt interface**, not the conversion itself. This is documented as the headline failure-boundary finding of the paper.

## 8. Reference

- `reports/phaseS_trackA_benchmark/cvar_metrics.csv`
- `reports/phaseS_trackC/loo_stability.csv`
- `reports/phaseS_trackB/val_weighted_consensus_metrics.csv`
- `reports/phaseS_trackA_metric_audit/method_cost_decomposition.md`
- `reports/phase4_read_only_audit/prompt_interface_failure_map.csv`
- `reports/phaseR_final/table_11_catastrophic_failure.md`
