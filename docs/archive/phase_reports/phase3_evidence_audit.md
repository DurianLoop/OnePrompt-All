# Phase3 Evidence Audit: Significance, Model Combinations, and Strong Baselines

## Why this audit exists

A small Dice improvement is not automatically meaningful. This audit checks three questions:

1. Is the selector improvement statistically reliable on paired case/model samples?
2. Does the method combine meaningfully with different SAM-family backends?
3. Can we honestly claim competitiveness with recent strong medical segmentation baselines?

## 1. Paired significance

The comparison is paired against fixed `auto_box` over the same case/model rows.

| policy | mean delta | 95% bootstrap CI | p-value | wins/ties/losses | note |
| --- | ---: | --- | ---: | --- | --- |
| data_aware_selector_v1 | +0.0115 | [-0.0031, +0.0287] | 0.1693 | 19/79/22 | uncertain_or_small |
| oracle_all_prompts | +0.0636 | [+0.0440, +0.0853] | 0.0001 | 69/51/0 | statistically_positive_and_practically_visible |
| oracle_auto_family | +0.0466 | [+0.0291, +0.0664] | 0.0001 | 48/72/0 | statistically_positive_and_practically_visible |
| ridge_reliability_selector_v2 | +0.0145 | [-0.0017, +0.0328] | 0.1051 | 19/88/13 | uncertain_or_small |
| score_selector_v1 | -0.0273 | [-0.0551, -0.0035] | 0.0349 | 13/90/17 | negative |
| single_auto_box | +0.0000 | [+0.0000, +0.0000] | 1.0000 | 0/120/0 | uncertain_or_small |
| single_auto_medical_box_v1 | -0.0297 | [-0.0546, -0.0070] | 0.0134 | 27/57/36 | negative |
| uncertainty_reliability_selector_v3 | +0.0160 | [-0.0001, +0.0340] | 0.0689 | 18/91/11 | uncertain_or_small |

Interpretation:

- `prompt_selector_v3` has mean delta +0.0160 if available.
- If the confidence interval includes zero, the current 40-case benchmark is not enough for a strong statistical claim.
- Even when positive, a delta around 0.01-0.02 Dice should be described as a reliability/selection gain, not a new SOTA segmentation model.

## 2. Model combination audit

| policy | rows | mean Dice | low Dice < 0.1 | high Dice >= 0.5 |
| --- | ---: | ---: | ---: | ---: |
| fixed_medsam_vit_b_auto_box | 40 | 0.3032 | 11 | 8 |
| fixed_sam2_tiny_auto_box | 40 | 0.3154 | 9 | 11 |
| fixed_sam_vit_b_auto_box | 40 | 0.3170 | 11 | 11 |
| oracle_best_model_auto_box | 40 | 0.3449 | 9 | 11 |
| oracle_best_model_v3_selector | 40 | 0.4002 | 7 | 13 |
| portfolio_v3_predicted_reliability | 40 | 0.3538 | 8 | 12 |

Interpretation:

- The oracle model portfolios estimate complementarity between SAM, MedSAM, and SAM2.
- A predicted-reliability portfolio is harder than within-model prompt selection because reliability scores must be calibrated across models.
- If the deployable portfolio is below a fixed model, the next step is calibration, not another prompt heuristic.

### Portfolio paired test

Baseline: `fixed_sam_vit_b_auto_box`, the strongest fixed single-model auto-box baseline in this run.

| policy | mean delta | 95% bootstrap CI | p-value | wins/ties/losses | note |
| --- | ---: | --- | ---: | --- | --- |
| fixed_medsam_vit_b_auto_box | -0.0137 | [-0.0444, +0.0140] | 0.3722 | 20/0/20 | uncertain_or_small |
| fixed_sam2_tiny_auto_box | -0.0016 | [-0.0230, +0.0174] | 0.8853 | 25/0/15 | uncertain_or_small |
| oracle_best_model_auto_box | +0.0279 | [+0.0150, +0.0430] | 0.0001 | 30/10/0 | statistically_positive_and_practically_visible |
| oracle_best_model_v3_selector | +0.0833 | [+0.0454, +0.1274] | 0.0001 | 31/7/2 | statistically_positive_and_practically_visible |
| portfolio_v3_predicted_reliability | +0.0368 | [-0.0119, +0.0888] | 0.1690 | 14/14/12 | uncertain_or_small |

## 3. Strong baseline positioning

Current OCSAM is a prompt selection and reranking method over SAM-family outputs. It should not yet be claimed to beat recent fully trained medical segmentation models.

Representative recent/stronger baselines include:

- Medical SAM Adapter / Med-SA: light adaptation over many medical tasks.
- SAM-Med2D: large-scale medical 2D fine-tuned SAM over millions of medical images and masks.
- MedSAM / MedSAM2 style medical adaptation.
- Task-specific U-Net/nnU-Net/TransUNet-style models on standard medical datasets.

Our current claim should be narrower:

```text
OCSAM improves automatic prompt selection and reliability over fixed automatic prompts for SAM-family medical segmentation.
```

To claim competitiveness with strong baselines, the project must add:

1. larger validation, at least 100-200 cases across multiple medical domains;
2. at least one trained medical SAM baseline such as SAM-Med2D or Med-SA if weights are usable;
3. a task-specific supervised baseline where possible;
4. statistical tests against each baseline, not only within-SAM prompt ablations.

## Recommendation

Proceed with OCSAM as a prompt reliability paper, not as a universal SOTA medical segmentation paper yet.

Next technical step:

```text
prompt_selector_v4 = perturbation-based uncertainty + cross-model calibration.
```
