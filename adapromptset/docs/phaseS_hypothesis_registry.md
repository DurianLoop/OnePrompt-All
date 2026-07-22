# Phase S Hypothesis Registry

Date: 2026-07-22
Branch: `phaseS-multitrack-reference-robust-research`

All hypotheses are preregistered before any Phase S inference. Hypotheses may
only be rejected or supported by the gate evidence; they may not be redefined
post-hoc.

## Track A Hypotheses

### HA1 — Reference instability is a real, measurable phenomenon
- Prediction: Under corrected metric definitions (within-query reference
  variance, pairwise deviation, CVaR), instability remains significant on
  MoNuSAC validation.
- Falsification: Corrected metrics show <30% queries with best-worst gap≥0.05
  OR mean within-query ref std<0.02.

### HA2 — Test-time-safe disagreement features predict instability
- Prediction: At least one test-time-safe feature (e.g., reference-set
  disagreement, matching-map entropy, candidate dispersion) correlates with
  per-query instability with Spearman |ρ|≥0.25 on validation.
- Falsification: All test-time-safe features have |ρ|<0.25.

### HA3 — Reference-count scaling is explainable
- Prediction: Increasing reference subset size k∈{1,3,6} reduces within-query
  variance in a monotone or otherwise explainable trend.
- Falsification: No explainable trend across k.

## Track B Hypotheses

### HB1 — Reference utility is predictable from test-time-safe features
- Prediction: A lightweight model (linear/ridge/GBM/MLP) trained on train-only
  utility labels achieves Spearman≥0.30 between predicted and actual reference
  utility on validation.
- Falsification: All models achieve Spearman<0.30.

### HB2 — Reliability-weighted consensus reduces within-query variance
- Prediction: Weighted consensus `C_weighted = Σ w_r C_r` reduces within-query
  reference std by ≥10% vs uniform mean on validation.
- Falsification: Weighted consensus does not reduce std by ≥10%.

### HB3 — Reliability-weighted consensus improves mean Dice
- Prediction: Weighted consensus improves mean Dice by ≥+0.005 vs uniform mean.
- Falsification: Dice improvement <0.005.

### HB4 — Reliability-weighted consensus reduces catastrophic failure
- Prediction: Catastrophic failure rate (Dice<0.1) drops by ≥15% relative vs
  uniform mean.
- Falsification: Relative reduction <15%.

## Track C Hypotheses

### HC1 — Set-level reference interactions add value beyond per-reference scoring
- Prediction: A set encoder (DeepSets/attention) improves ΔDice≥+0.005 over
  Track B strongest on validation.
- Falsification: No ΔDice improvement ≥+0.005.

### HC2 — Set models remain stable under reference removal
- Prediction: Leave-one-reference-out does not collapse performance.
- Falsification: Performance drops catastrophically (>20% relative Dice loss).

## Track D Hypotheses

### HD1 — Direct reference-conditioned segmentation outperforms prompt interface
- Prediction: Direct method achieves ΔDice≥+0.03 over raw matching mask and
  ΔDice≥+0.01 over strongest fixed prompt.
- Falsification: Neither threshold met.

### HD2 — Direct method transfers cross-dataset
- Prediction: GlaS or DDR shows directionally consistent improvement.
- Falsification: No directional consistency on either secondary dataset.

## Cross-Track Hypotheses

### HX1 — Prompt interface is the bottleneck
- Prediction: If Track B/C fail but Track D succeeds, the prompt interface
  (point-based) is the dominant bottleneck.
- Falsification: Track D also fails.

### HX2 — Reference instability is systematic, not random
- Prediction: If deterministic consensus fails to reduce variance (Phase R
  finding) and reliability weighting also fails, instability contains a
  systematic component not removable by aggregation.
- Falsification: Reliability weighting successfully reduces variance.
