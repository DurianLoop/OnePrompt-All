# Phase S — Novelty Audit (Directive §13)

Date: 2026-07-22
Branch: `phaseS-multitrack-reference-robust-research`
Commit: 3e63733

## Related Works Audited (12)

| # | Work | Relation to Phase S |
|---|---|---|
| 1 | GF-SAM | Guided filtering for SAM; improves prompt robustness via spatial filtering. Differs: we benchmark reference-induced instability, do not propose a new matcher. |
| 2 | SynPo | Synthetic prompt generation; augments point prompts. Differs: we use real references only, no synthetic prompts. |
| 3 | AlignSAM | Alignment-based SAM adaptation. Differs: we do not fine-tune SAM; we use frozen SAM-B. |
| 4 | Point Prompt Defender | Defends against adversarial point prompts. Differs: we study *natural* reference-induced instability, not adversarial perturbations. |
| 5 | ProSAM | Progressive prompt learning for SAM. Differs: we do not learn prompts; we use fixed candidates from matching. |
| 6 | MAUP | Multi-scale anchor unified prompt. Differs: we do not use multi-scale anchors; single-scale DINOv2-L features. |
| 7 | SAMIX | SAM with mixed prompts. Differs: we aggregate reference evidence, not prompt mixing. |
| 8 | Few-shot segmentation | General few-shot segmentation (e.g., PANet, PPNet). Differs: we do not train a segmenter; reference-conditioned prompting only. |
| 9 | Visual reference segmentation | Reference-image-guided segmentation. Differs: we benchmark the *instability* of this paradigm, not just propose a method. |
| 10 | Selective prediction / abstention | Predict-then-abstain methods. Differs: we do not abstain; we attempt to mitigate via weighting/set modeling (which failed). |
| 11 | Robust aggregation | Ensemble / aggregation methods. Differs: we test 6 deterministic aggregators (B1-B6) plus learned weighting; all fail to improve over B2_mean. |
| 12 | Set learning (DeepSets, Set Transformer) | Set-conditioned learning. Differs: we tested C3 (pixel-wise agreement gate) and C4 (stable candidate scoring); both degrade Dice. C1/C2 (DeepSets/attention) not reached because C3/C4 showed no signal. |

## Seven Mandatory Questions

### Q1. Is this work only reference selection?

**NO.** Phase S goes beyond reference selection:
- Track A benchmarks *instability* (variance, tail risk, prediction consistency), not selection quality.
- Track B learns *reliability weights* from test-time-safe features, then aggregates via softmax weighting — this is reliability estimation, not selection.
- Track C tests *set-conditioned* interventions (pixel-wise agreement gating, stable candidate scoring) — these operate on the full reference set jointly, not selecting a subset.
- The finding is that *no* reference-based intervention (selection, weighting, set-conditioning) improves over mean consensus — a negative result that bounds the entire reference-selection/reliability paradigm.

### Q2. Is it only uncertainty weighting?

**NO.** Track B uses Gradient Boosting (M4_gbm) on 18 test-time-safe features to predict per-reference utility, then converts to weights via softmax with temperature. This is *learned reliability estimation*, not uncertainty-based weighting. The features include:
- Reference-only: foreground ratio, component count, compactness, fragmentation, prototype separation, stain stats, embedding stats.
- Reference-query: global embedding similarity, fg/bg matching stats, contrast entropy, top-score decay, peak density, candidate dispersion, candidate agreement, predicted fg ratio, map smoothness, reference-set disagreement.
- Leave-one-reference-out: consensus map change, candidate overlap change, prompt coordinate shift, matching-map proxy change.

Track B's failure (delta Dice = -0.0024) shows that *even with strong utility prediction* (Spearman = 0.478), weighting does not help — the bottleneck is downstream.

### Q3. Is it only mean consensus?

**NO.** Phase S tests 6 deterministic baselines (B1_random, B2_mean, B3_median, B4_trimmed, B5_disagreement, B6_quantile), learned weighted aggregation (M4_gbm, M5_mlp), and set-conditioned methods (C3 pixel-wise gate, C4 candidate scoring). The key finding is that B2_mean (the simplest aggregator) outperforms all alternatives — but this is a *negative result about the reference-weighting paradigm*, not a positive claim about mean consensus.

### Q4. Is it only a more complex point selector?

**NO.** Phase S does NOT improve point selection. The candidate pool is frozen from Phase X (DINOv2-L patch-grid centers). Track C's C4 (stable candidate scoring) *re-ranks* candidates by stability, but this degrades Dice. The finding is that the *point-prompt interface itself* is the bottleneck — SAM-B's response to point prompts is the limiting factor, not which points are selected.

### Q5. Does it truly solve cross-reference stability?

**NO — this is explicitly documented as a failure boundary.** All three method tracks (B, C, D) failed:
- Track B: delta Dice = -0.0024 (weighted vs uniform).
- Track C: best hyperparameters = (0,0,0) = B2_mean baseline; all non-zero hyperparameters degrade Dice.
- Track D: not started (start condition 3 not met — SAM refinement *helps*, contradicting the premise).

Phase S *documents* the boundary: reference instability is real and predictable, but cannot be mitigated through reference weighting, set conditioning, or direct segmentation (the latter not even reaching start conditions). This is a *negative/failure-boundary* contribution, not a solution.

### Q6. Does it propose new metric/protocol?

**YES.** This is a primary novelty:
1. **Corrected V_ref decomposition**: Phase R's cond3 mixed variance axes (V_total vs V_query). Phase S introduces the within-query reference variance `V_ref = (1/N) Σ_q Var_r[D(q,r)]` and `V_ref_frac = V_ref / (V_query + V_ref)`, isolating reference-induced variance from case difficulty.
2. **Tail-risk metrics for reference instability**: CVaR_0.1, catastrophic failure rate (Dice<0.1), worst-subset Dice, worst-reference gap.
3. **Test-time-safe predictability**: Spearman correlation between disagreement features and instability, *without using query GT*.
4. **Preregistered multi-track protocol**: 4 tracks (A/B/C/D) with explicit gates, 210 GPU-hour cap, frozen data splits (train=80/val=41/holdout=40/formal-test=locked), claim registry with allowed/prohibited claims, and mandatory novelty audit + mock review.

### Q7. Does it have stable, falsifiable claims?

**YES.** Two allowed claims, both with predeclared gates and bootstrap CIs:

**Claim 1 (Track A):** "Valid references induce substantial, measurable segmentation variability under the frozen matching-and-prompting pipeline, and this instability is associated with test-time-safe disagreement features."
- Falsifiable: V_ref_frac ≥ 0.10 (result: 0.1376–0.3312); best-worst gap ≥0.05 on ≥30% queries (result: 80.5%–90.2%); prediction Spearman |ρ| ≥ 0.25 (result: 0.8832).
- All 7/7 Gate A conditions PASSED.

**Claim 2 (Track B Utility-Only):** "Reference utility is predictable from test-time-safe features, but weighted consensus does not reduce variance; the utility signal is a scientific finding without a deployable stability gain."
- Falsifiable: utility prediction Spearman ≥ 0.30 (result: 0.4781); weighted Dice does NOT exceed uniform by 0.005 (result: -0.0024); ref std reduction NOT ≥10% (result: 2.83%).
- 1/6 Gate B conditions passed; the claim is *negative* (predictable but not deployable).

Both claims survived a prohibited-claim scan (10 prohibited tokens checked).

## Novelty Summary

| Dimension | Phase S Contribution | Novelty Level |
|---|---|---|
| Metric | Corrected V_ref decomposition + tail-risk suite for reference instability | **New** |
| Protocol | Preregistered multi-track with claim registry + mock review | **New** |
| Benchmark | First systematic reference-instability benchmark on MoNuSAC (41 val queries, 3 configs, k=1/3/6) | **New** |
| Method (Track B) | Test-time-safe reliability features + GBM weighting | Incremental |
| Method (Track C) | Pixel-wise agreement gate + stable candidate scoring | Incremental |
| Finding | Prompt-interface bottleneck documented as failure boundary | **New** |
| Negative result | All method tracks fail; predictability does not imply deployability | **New** |

The core novelty is the **benchmark + failure-boundary** contribution, not a new method. This aligns with the allowed paper type (§21: Benchmark + failure-boundary paper).
