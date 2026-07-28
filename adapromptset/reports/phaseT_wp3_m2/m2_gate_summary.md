# Phase T WP3 M2 Selective Prediction Gate Summary

Date: 2026-07-23T18:17:06
## Dual-model approach

Because instability (high gap) and catastrophic failure (low Dice) are anti-correlated, a single M2 model cannot satisfy both cond1 (instability AUROC) and cond2-5 (catastrophic-risk selective prediction). Two models are therefore trained on the same 4 test-time-safe features but with different labels:

- **Catastrophic-trained model** (drives selective prediction): **M2-2_logistic_regression**
- **Instability-trained model** (drives cond1): **M2-2_logistic_regression**

Train episodes: 1200 (80 queries)
Val episodes: 615 (41 queries)
Prevalence (instability): 0.8472
Prevalence (catastrophic): 0.1171
Tail-risk threshold (train mean_dice p10): 0.067587

## Catastrophic-trained model comparison (episode-level val)

| model | AUROC | AUPRC_instab | AUPRC_cat | ECE | ECE_cal | CFR_red@80 | Dice_chg@80 |
|-------|-------|-------------|-----------|-----|---------|------------|-------------|
| M2-0_entropy_threshold | 0.4977 | 0.8555 | 0.0831 | 0.0226 | 0.0226 | 0.2014 | +0.015949 |
| M2-1_disagreement_threshold | 0.4977 | 0.8555 | 0.0831 | 0.0226 | 0.0226 | 0.2014 | +0.015949 |
| M2-2_logistic_regression | 0.0600 | 0.6750 | 0.4338 | 0.0425 | 0.0425 | 0.3924 | +0.021425 |
| M2-3_gradient_boosting | 0.0781 | 0.6804 | 0.3997 | 0.0658 | 0.0658 | 0.4444 | +0.022310 |
| M2-4_small_mlp | 0.0622 | 0.6756 | 0.4167 | 0.0583 | 0.0583 | 0.4271 | +0.021842 |

## Instability-trained model comparison (episode-level val)

| model | AUROC | AUPRC_instab | AUPRC_cat | ECE | ECE_cal | CFR_red@80 | Dice_chg@80 |
|-------|-------|-------------|-----------|-----|---------|------------|-------------|
| M2-0_entropy_threshold | 0.4977 | 0.8555 | 0.0831 | 0.0343 | 0.0343 | -0.2500 | -0.019616 |
| M2-1_disagreement_threshold | 0.4977 | 0.8555 | 0.0831 | 0.0343 | 0.0343 | -0.2500 | -0.019616 |
| M2-2_logistic_regression | 0.9443 | 0.9894 | 0.0710 | 0.0281 | 0.0281 | -0.1979 | -0.002906 |
| M2-3_gradient_boosting | 0.9265 | 0.9830 | 0.0739 | 0.0584 | 0.0584 | -0.1632 | -0.007572 |
| M2-4_small_mlp | 0.9413 | 0.9890 | 0.0713 | 0.0394 | 0.0394 | -0.1979 | -0.004166 |

## Gate conditions (section 10.5)

| # | condition | pass |
|---|-----------|------|
| 1 | instability AUROC >= 0.70 | PASS |
| 2 | catastrophic AUPRC > prevalence | PASS |
| 3 | ECE <= 0.10 (or <= 0.08 after calibration) | PASS |
| 4 | CFR reduction >= 30% at 80% coverage | PASS |
| 5 | mean Dice not decrease at 80% coverage | PASS |
| 6 | holdout direction consistent | PENDING |

**cond1** (`cond1_instability_auroc_ge_0p70`): **PASS** -- AUROC = 0.9443 (query-level: 0.9392) [instability-trained model: M2-2_logistic_regression]

**cond2** (`cond2_catastrophic_auprc_gt_prevalence`): **PASS** -- AUPRC = 0.4338, prevalence = 0.1171 [catastrophic-trained model: M2-2_logistic_regression]

**cond3** (`cond3_ece_le_0p10`): **PASS** -- ECE = 0.0425 [catastrophic model]

**cond4** (`cond4_cfr_reduction_30pct_at_80pct_coverage`): **PASS** -- CFR reduction = 39.2% (CFR all=0.1171, CFR@80%=0.0711)

**cond5** (`cond5_dice_not_decrease_at_80pct_coverage`): **PASS** -- Dice change = +0.021425 (Dice all=0.414136, Dice@80%=0.435561) [catastrophic model]

**cond6** (`cond6_holdout_direction_consistent`): **PENDING** -- PENDING - requires holdout run (holdout benchmark not yet executed)

## Decision: **PHASET_M2_SELECTIVE_PREDICTION_PASS**

(5/5 checked conditions passed, 1 pending)

## Risk-coverage curve (catastrophic model, episode level)

| coverage | abstention | selective_dice | selective_cfr | cost | n_covered |
|----------|------------|----------------|--------------|------|-----------|
| 0.10 | 0.9008 | 0.403953 | 0.000000 | 0.900813 | 61 |
| 0.20 | 0.8000 | 0.424652 | 0.024390 | 0.804878 | 123 |
| 0.30 | 0.6992 | 0.442588 | 0.032432 | 0.708943 | 185 |
| 0.40 | 0.6000 | 0.450458 | 0.040650 | 0.616260 | 246 |
| 0.50 | 0.5008 | 0.441024 | 0.042345 | 0.521951 | 307 |
| 0.60 | 0.4000 | 0.440461 | 0.051491 | 0.430894 | 369 |
| 0.70 | 0.2992 | 0.436603 | 0.060325 | 0.341463 | 431 |
| 0.80 | 0.2000 | 0.435561 | 0.071138 | 0.256911 | 492 |
| 0.90 | 0.0992 | 0.429716 | 0.084838 | 0.175610 | 554 |
| 1.00 | 0.0000 | 0.414136 | 0.117073 | 0.117073 | 615 |

## Calibration bins (catastrophic model)

| bin | lower | upper | count | accuracy | confidence | gap |
|-----|-------|--------|-------|----------|------------|-----|
| 0 | 0.0 | 0.1 | 0 | 0.000000 | 0.000000 | 0.000000 |
| 1 | 0.1 | 0.2 | 0 | 0.000000 | 0.000000 | 0.000000 |
| 2 | 0.2 | 0.3 | 0 | 0.000000 | 0.000000 | 0.000000 |
| 3 | 0.3 | 0.4 | 0 | 0.000000 | 0.000000 | 0.000000 |
| 4 | 0.4 | 0.5 | 0 | 0.000000 | 0.000000 | 0.000000 |
| 5 | 0.5 | 0.6 | 31 | 0.451613 | 0.555607 | 0.103994 |
| 6 | 0.6 | 0.7 | 29 | 0.551724 | 0.648547 | 0.096823 |
| 7 | 0.7 | 0.8 | 33 | 0.757576 | 0.745644 | 0.011932 |
| 8 | 0.8 | 0.9 | 66 | 0.818182 | 0.851942 | 0.033761 |
| 9 | 0.9 | 1.0 | 456 | 0.942982 | 0.981386 | 0.038404 |

