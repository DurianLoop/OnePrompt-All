# Appendix

## A. Frozen hyperparameters and configurations

**Prompt configurations (frozen across all Phase T runs).**

| Config | K+ | K- | Negative selector |
|---|---|---|---|
| P20-N1 | 20 | 1 | N_BOTTOMK |
| P30-N0 | 30 | 0 | N_NONE |
| P50-N0 | 50 | 0 | N_NONE |

**Reference protocol.** Six references per query, drawn from the
train-only pool, patient-disjoint. k∈{1,3,6} subset sizes. Ten
resampling seeds on MoNuSAC train; five on MoNuSeg and IDRiD development
splits. Holdout and validation reference allocations fixed at
pre-registration and hash-verified.

**Matcher.** DINOv2-L features, z-score foreground/background
normalization. Per-case feature cache reused across seeds.

**Segmenter.** SAM-B, single call per consensus aggregation. Multimask
output rule frozen. B2_mean fuses reference-conditioned prompts before
the single SAM call.

**Catastrophic threshold.** Dice < 0.1 (frozen; not retunable after
holdout).

**Bootstrap.** B=10,000 iterations, patient-cluster resampling for
MoNuSAC, query-level for MoNuSeg and IDRiD, 95% CI from the 2.5th and
97.5th percentiles. Multiple comparisons: Holm step-down at familywise
α=0.05.

**M2 model.** Logistic regression, L2 regularization, strength selected
by 5-fold cross-validation on train. Four test-time-safe features:
`matching_map_disagreement`, `candidate_jaccard_mean_k1`,
`prompt_overlap_mean_k1`, `prediction_jaccard_k1_within`. Two labels
constructed on train only: instability (best-worst gap > train 90th
percentile) and catastrophic (Dice < 0.1). Coverage threshold for gate:
80%.

**M1 utility.** `U(q,k) = (1-λ_c-λ_r)·Dice - λ_c·Cost - λ_r·Risk`,
λ_c, λ_r ∈ {0, 0.25, 0.5, 0.75, 1.0} selected by cross-validation on
train. Policy outputs k∈{1,3,6}.

## B. Per-config WP1 holdout tables

**Within-query reference variance V_ref(k).**

| config | k | n | V_ref | V_query | V_ref_frac | mean_ref_std | best-worst gap |
|---|---|---|---|---|---|---|---|
| P20-N1 | 1 | 40 | 0.011088 | 0.031122 | 0.2627 | 0.085208 | 0.239294 |
| P20-N1 | 3 | 40 | 0.009507 | 0.032709 | 0.2252 | 0.080181 | 0.262545 |
| P20-N1 | 6 | 40 | 0.008304 | 0.035611 | 0.1891 | 0.069922 | 0.225234 |
| P30-N0 | 1 | 40 | 0.008818 | 0.032969 | 0.2110 | 0.077195 | 0.214120 |
| P30-N0 | 3 | 40 | 0.007938 | 0.035122 | 0.1843 | 0.072750 | 0.230197 |
| P30-N0 | 6 | 40 | 0.006790 | 0.035913 | 0.1590 | 0.061878 | 0.196472 |
| P50-N0 | 1 | 40 | 0.005186 | 0.035860 | 0.1264 | 0.060222 | 0.166836 |
| P50-N0 | 3 | 40 | 0.003903 | 0.038013 | 0.0931 | 0.053054 | 0.169467 |
| P50-N0 | 6 | 40 | 0.002579 | 0.039515 | 0.0613 | 0.040395 | 0.132370 |

**Mean Dice and tail risk by (config, k).**

| config | k | mean_dice | mean_worst | min_worst | p10_worst | CVaR_0.1 | CFR |
|---|---|---|---|---|---|---|---|
| P20-N1 | 1 | 0.461742 | 0.221415 | 0.000000 | 0.000384 | 0.056059 | 0.0729 |
| P20-N1 | 3 | 0.482079 | 0.333483 | 0.000136 | 0.018278 | 0.077208 | 0.0675 |
| P20-N1 | 6 | 0.488852 | 0.364820 | 0.000000 | 0.044408 | 0.076556 | 0.0650 |
| P30-N0 | 1 | 0.487949 | 0.305835 | 0.000000 | 0.037340 | 0.075086 | 0.0621 |
| P30-N0 | 3 | 0.505194 | 0.372114 | 0.001982 | 0.048409 | 0.093938 | 0.0575 |
| P30-N0 | 6 | 0.516846 | 0.400154 | 0.016060 | 0.087525 | 0.102650 | 0.0575 |
| P50-N0 | 1 | 0.478970 | 0.331364 | 0.005334 | 0.084058 | 0.083139 | 0.0550 |
| P50-N0 | 3 | 0.498417 | 0.404445 | 0.016912 | 0.191492 | 0.095159 | 0.0525 |
| P50-N0 | 6 | 0.511686 | 0.440720 | 0.015963 | 0.206773 | 0.098674 | 0.0500 |

## C. WP1 gate condition detail

| # | Condition | Result | Threshold |
|---|---|---|---|
| 1 | fraction of queries best-worst gap ≥ 0.05 | 0.9250 | ≥ 0.50 |
| 2 | mean within-query reference std (k=1) | 0.074208 | ≥ 0.02 |
| 3 | V_ref_frac (k=1) per config | 0.1264–0.2627 | ≥ 0.10 (any) |
| 4 | consensus gain (k=6 vs k=1) | +0.026631 | ≥ 0.010 |
| 5 | paired bootstrap 95% CI lower | 0.014009 | > 0 |
| 6 | consensus CFR (k=6) vs single CFR (k=1) | 0.0575 vs 0.0633 | ≤ |
| 7 | consensus CVaR_0.1 (k=6) vs single (k=1) | 0.091945 vs 0.075133 | ≥ single − 0.005 |
| 8 | J_pred consensus (k=6) vs single (k=1) | 0.718951 vs 0.648965 | ≥ |
| 9 | best |Spearman(feature, gap)| | 0.7241 | ≥ 0.25 |
| 10 | patient-disjoint violations / 2400 | 0 | 0 |

## D. M2 risk-coverage curve (validation, catastrophic model)

| coverage | abstention | selective_dice | selective_cfr | n_covered |
|---|---|---|---|---|
| 0.10 | 0.901 | 0.4040 | 0.0000 | 61 |
| 0.20 | 0.800 | 0.4247 | 0.0244 | 123 |
| 0.30 | 0.699 | 0.4426 | 0.0324 | 185 |
| 0.40 | 0.600 | 0.4505 | 0.0407 | 246 |
| 0.50 | 0.501 | 0.4410 | 0.0423 | 307 |
| 0.60 | 0.400 | 0.4405 | 0.0515 | 369 |
| 0.70 | 0.299 | 0.4366 | 0.0603 | 431 |
| 0.80 | 0.200 | 0.4356 | 0.0711 | 492 |
| 0.90 | 0.099 | 0.4297 | 0.0848 | 554 |
| 1.00 | 0.000 | 0.4141 | 0.1171 | 615 |

## E. M2 validation vs holdout comparison

| metric | validation | holdout | direction |
|---|---|---|---|
| instability AUROC | 0.9443 | 0.9283 | ≥ 0.70 |
| catastrophic AUPRC | 0.4338 | 0.5277 | > prevalence |
| ECE | 0.0425 | 0.0270 | ≤ 0.10 |
| CFR reduction @80% | 0.3924 | 0.7917 | ≥ 0.30 |
| Dice change @80% | +0.0214 | +0.0090 | ≥ 0 |
| prevalence (catastrophic) | 0.1171 | 0.0500 | — |

## F. M1 model comparison (validation)

| model | λ_c | λ_r | mean_dice | avg_k | CVaR_0.1 | CFR | cost |
|---|---|---|---|---|---|---|---|
| M1-0 always_k1 | 0 | 0 | 0.414136 | 1.0000 | 0.059762 | 0.117073 | 0.166667 |
| M1-1 always_k3 | 0 | 0 | 0.443245 | 3.0000 | 0.054352 | 0.115447 | 0.500000 |
| M1-2 always_k6 | 0 | 0 | 0.458553 | 6.0000 | 0.048999 | 0.125203 | 1.000000 |
| M1-3 handcrafted | 0 | 0 | 0.457056 | 5.6341 | 0.053718 | 0.120325 | 0.939024 |
| M1-4 logistic | 0 | 0 | 0.455240 | 5.8780 | 0.048977 | 0.125203 | 0.979675 |
| M1-5 gradient_boosting | 0 | 0 | 0.451626 | 4.9805 | 0.052773 | 0.117073 | 0.830081 |

## G. Disagreement-feature Spearman |ρ| (WP1 holdout, k=1)

| feature | Spearman |ρ| |
|---|---|
| prediction_jaccard_k1_within | 0.7241 |
| matching_map_disagreement | 0.1434 |
| candidate_jaccard_mean_k1 | 0.1030 |
| prompt_overlap_mean_k1 | 0.1030 |

## H. Prohibited-claim audit

The following tokens do not appear in any Phase T paper section except
inside the claim registry as documentation of the prohibition: "SOTA",
"state-of-the-art", "instability solved", "instability eliminated",
"universal across datasets", "mean Dice alone proves robustness",
"submission-ready", "publication-completed". Negated occurrences (e.g.,
"we do not claim SOTA") are permitted and used.
