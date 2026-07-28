# Phase S Track Registry

Date: 2026-07-22
Branch: `phaseS-multitrack-reference-robust-research`

## Registered Tracks

### Track A — Metric Correction and Stability Benchmark

- **Goal**: Confirm Phase R variance metric measures the right object; build a
  reusable reference-instability metric suite.
- **Start condition**: Preregistration commit complete.
- **Gate**: 7 conditions (§6.5), including corrected instability significant,
  ≥30% val queries gap≥0.05, mean within-query ref std≥0.02, code/definition
  consistency, reference-count trend explainable, tail-risk linkage, test-time-
  safe disagreement feature Spearman |ρ|≥0.25.
- **States**: `PHASES_TRACK_A_REFERENCE_INSTABILITY_BENCHMARK_VALID` /
  `PHASES_TRACK_A_METRIC_OR_PHENOMENON_NOT_STABLE`.
- **On failure**: Stop reference-stability mainline; evaluate Track D.
- **Compute cap**: 20 GPU-hours.

### Track B — Learned Reference Reliability and Robust Aggregation

- **Goal**: Learn per-reference reliability `û(q, R_r)` and produce weighted
  consensus `C_weighted(x) = Σ_r w_r C_r(x)`.
- **Start condition**: Track A passes (instability benchmark valid).
- **Model ladder**: Uniform Mean → Handcrafted Score → Linear/Ridge →
  Pairwise Ranking → Gradient Boosting → Small MLP (only if positive signal).
- **Gate**: 6+ conditions (§7.6), including ΔDice≥+0.005 OR within-query ref
  std↓≥10%, plus utility prediction Spearman≥0.30, 3-seed consistency, no
  query GT, 1 SAM call, no weight collapse, bootstrap point estimate>0.
- **States**: `PHASES_TRACK_B_RELIABILITY_WEIGHTING_PASS` /
  `PHASES_TRACK_B_UTILITY_ONLY_PASS` /
  `PHASES_TRACK_B_STABILITY_ONLY_PASS` /
  `PHASES_TRACK_B_NO_GAIN`.
- **Compute cap**: 35 GPU-hours.

### Track C — Set-Conditioned Robust Prompting

- **Goal**: Use set-level interactions among references via DeepSets,
  lightweight attention, pixel-wise agreement gates, or stable candidate
  scoring.
- **Start condition**: Track B has stable positive signal but plateaus;
  handcrafted/weighted mean useful but cannot exploit reference interactions;
  utility predictable with Spearman≥0.30.
- **Complexity limits**: <5M params, no DINOv2-L update, no SAM-B update, no
  large decoder, 1 SAM call.
- **Gate**: 8 conditions (§8.4) vs Track B strongest, including ΔDice≥+0.005,
  CVaR+0.01 or ref std↓≥10%, 3-seed consistency, leave-one-ref-out stability.
- **States**: `PHASES_TRACK_C_SET_MODEL_PASS` /
  `PHASES_TRACK_C_SET_MODEL_NO_ADDED_VALUE`.
- **Compute cap**: 45 GPU-hours.

### Track D — Direct Reference-Conditioned Segmentation

- **Goal**: Bypass SAM point-prompt interface and produce a direct
  reference-conditioned segmenter with multi-prototype matching, uncertainty
  fusion, lightweight refiner, or topology-aware loss.
- **Start condition**: Track A proves prompt-interface instability is strong;
  Track B/C cannot improve stability; raw matching maps have stable signal but
  point-prompt conversion loses information; SAM refinement unstable.
- **Minimum technical core**: Not just threshold+morphology; must implement
  multi-prototype / uncertainty fusion / lightweight refiner / topology-aware
  loss.
- **Gate**: 8 conditions (§9.4), including ΔDice≥+0.03 vs raw matching mask,
  ΔDice≥+0.01 vs strongest fixed prompt, GlaS or DDR cross-dataset consistent,
  reference-removal ablation decreases performance.
- **States**: `PHASES_TRACK_D_DIRECT_METHOD_PASS` /
  `PHASES_TRACK_D_DIRECT_METHOD_NO_GAIN`.
- **Compute cap**: 50 GPU-hours.

## Routing Rules (§10)

- Track A strong + reliability predictable → Track B.
- Track B gain → prioritize B.
- Track B plateaus + set interaction signal → Track C.
- Track B/C fail but raw matching valuable → Track D.
- Track A alone sufficient for benchmark, method gains limited → Benchmark paper.
- All tracks fail → Negative / boundary paper assessment.
