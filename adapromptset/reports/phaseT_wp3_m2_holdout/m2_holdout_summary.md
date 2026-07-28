# Phase T WP3 M2 Holdout Confirmation (section 12)

Date: 2026-07-23T18:22:15

## Frozen models (selected on validation)

- Catastrophic-trained model: **M2-2_logistic_regression** (val catastrophic AUPRC = 0.4338)
- Instability-trained model: **M2-2_logistic_regression** (val instability AUROC = 0.9443)

## Data splits

- Train episodes: 1200 (80 queries) -- model fitting
- Val episodes: 615 (41 queries) -- model selection
- Holdout episodes: 1200 (40 queries) -- one-shot confirmation

Holdout prevalence (instability): 0.8917
Holdout prevalence (catastrophic): 0.0500

## Holdout metrics (frozen dual-model)

| metric | value |
|--------|-------|
| instability AUROC | 0.9283 |
| instability AUPRC | 0.9893 |
| catastrophic AUPRC | 0.5277 |
| ECE | 0.0270 |
| ECE (calibrated) | 0.0270 |
| CFR (no abstention) | 0.0500 |
| CFR @80% coverage | 0.0104 |
| CFR reduction @80% | 0.7917 |
| Dice (no abstention) | 0.476220 |
| Dice @80% coverage | 0.485208 |
| Dice change @80% | +0.008987 |

## Validation vs holdout comparison

| metric | validation | holdout | direction consistent |
|--------|-----------|---------|----------------------|
| instability AUROC | 0.9443 | 0.9283 | >=0.70 |
| catastrophic AUPRC | 0.4338 | 0.5277 | >prevalence |
| ECE | 0.0425 | 0.0270 | <=0.10 |
| CFR reduction @80% | 0.3924 | 0.7917 | >=0.30 |
| Dice change @80% | 0.0214 | 0.0090 | >=0 |

## Holdout gate conditions (cond6 = directionally consistent)

| # | condition | pass |
|---|-----------|------|
| 1 | instability AUROC >= 0.70 | PASS |
| 2 | catastrophic AUPRC > prevalence | PASS |
| 3 | ECE <= 0.10 (or <= 0.08 after calibration) | PASS |
| 4 | CFR reduction >= 30% at 80% coverage | PASS |
| 5 | mean Dice not decrease at 80% coverage | PASS |
| 6 | holdout direction consistent (cond1-5 all pass) | PASS |

**cond1** (`cond1_instability_auroc_ge_0p70`): **PASS** -- AUROC = 0.9283 (query-level: 1.0000) [instability-trained model]

**cond2** (`cond2_catastrophic_auprc_gt_prevalence`): **PASS** -- AUPRC = 0.5277, prevalence = 0.0500 [catastrophic-trained model]

**cond3** (`cond3_ece_le_0p10`): **PASS** -- ECE = 0.0270

**cond4** (`cond4_cfr_reduction_30pct_at_80pct_coverage`): **PASS** -- CFR reduction = 79.2% (CFR all=0.0500, CFR@80%=0.0104)

**cond5** (`cond5_dice_not_decrease_at_80pct_coverage`): **PASS** -- Dice change = +0.008987 (Dice all=0.476220, Dice@80%=0.485208)

**cond6** (`cond6_holdout_direction_consistent`): **PASS** -- cond1-5 all pass on holdout (directionally consistent with val)

## Decision: **PHASET_M2_SELECTIVE_PREDICTION_PASS**

(6/6 conditions passed on holdout)

