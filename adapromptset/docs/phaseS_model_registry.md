# Phase S Model Registry

Date: 2026-07-22
Branch: `phaseS-multitrack-reference-robust-research`

All models and baselines are preregistered. No model may be added after Gate
evaluation begins.

## Baselines (Deterministic, from Phase R)

| ID | Method | Train? | Deployable? |
|---|---|---|---|
| B0 | Single reference (fixed) | No | Yes |
| B1 | Random single reference | No | Yes |
| B2 | Mean consensus | No | Yes |
| B3 | Median consensus | No | Yes |
| B4 | Trimmed mean consensus | No | Yes |
| B5 | Disagreement-penalized (β∈{0.25,0.5,1.0}) | β on train | Yes |
| B6 | Quantile (τ∈{0.50,0.67,0.83}) | τ on train | Yes |

## Track B Model Ladder (§7.4)

| ID | Model | Train labels | Test-time-safe features | Hyperparameters |
|---|---|---|---|---|
| M0 | Uniform Mean | None | None | — |
| M1 | Handcrafted Score | None | Reference-only + reference-query | Weights fixed by formula |
| M2 | Linear / Ridge Regression | u(q, R_r) | All test-time-safe | L2 reg α ∈ {0.01, 0.1, 1.0} |
| M3 | Pairwise Ranking | y_ij | All test-time-safe | Margin m ∈ {0.05, 0.1, 0.2} |
| M4 | Gradient Boosting | u(q, R_r) | All test-time-safe | depth ∈ {3, 5}, n_est ∈ {100, 300} |
| M5 | Small MLP (only if M2-M4 positive) | u(q, R_r) | All test-time-safe | hidden ∈ {64, 128}, dropout 0.1 |

### Prohibited Models

- Large Transformer (>5M params)
- RL reference selector
- Validation GT supervision
- Test-time SAM search

### Weighting & Risk Objectives (§7.5)

```
w_r = exp(û_r / T) / Σ_j exp(û_j / T), with w_r ≥ ε
U_robust = E_S[D] - λ * Std_S[D]
```

Hyperparameter grid (train/inner-CV only):
- T ∈ {0.25, 0.5, 1.0, 2.0}
- ε ∈ {0, 0.05}
- λ ∈ {0, 0.25, 0.5, 1.0}
- α ∈ {0.1, 0.2} (for CVaR variant)

## Track C Model Ladder (§8.2)

| ID | Model | Params | Notes |
|---|---|---|---|
| C1 | DeepSets Reliability Encoder | <2M | z_R = ρ(Σ_r φ(z_r)) |
| C2 | Lightweight Set Attention | <5M | 1-2 attention layers |
| C3 | Pixel-wise Agreement Gate | <1M | Low-capacity w_r(x) |
| C4 | Stable Candidate Scoring | <1M | s_i = μ_i - βσ_i + γa_i |

## Track D Methods (§9.2)

| ID | Method | Technical Core |
|---|---|---|
| D1 | Multi-Prototype Matching | Multiple foreground/background prototypes |
| D2 | Reference-Set Uncertainty Fusion | Per-reference uncertainty + fusion |
| D3 | Lightweight Dense Refiner | Small CNN refining matching map |
| D4 | Topology / Component-Aware Loss | Component-aware training objective |

### Track D Baselines

- Raw mean matching mask
- Disagreement-penalized mask
- Threshold + morphology
- Strongest fixed prompt (from Phase R)
- Strongest Track B/C prompting
- Lightweight refiner
- Supervised nnU-Net (upper bound, different input condition)

## Frozen Components (All Tracks)

- DINOv2-L matcher (no fine-tuning)
- SAM-B segmenter (no fine-tuning)
- Image size 518, patch size 14, grid 37
- Prompt configs: P20-N1, P30-N0, P50-N0 (from Phase R)
- Consensus methods: B2_mean, B3_median, B4_trimmed, B5_disagreement, B6_quantile
