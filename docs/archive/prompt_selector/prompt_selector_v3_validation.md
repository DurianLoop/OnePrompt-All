# Prompt Selector v3 Validation

## Component

`prompt_selector_v3` adds prompt-induced uncertainty and mask stability to the v2 reliability scorer.

The component is inspired by uncertainty-guided multi-prompt adaptation and dynamic prompting ideas in the reviewed medical SAM papers, but is implemented locally with the current benchmark artifacts.

It uses no GT on the held-out case. For each case/model, it compares auto-family masks and derives:

- mean mask agreement between candidate prompts;
- minimum mask agreement;
- prompt uncertainty = 1 - agreement;
- area deviation from median candidate mask;
- area deviation from `auto_box` mask;
- SAM/SAM2 raw score when available.

## Results

| method | mean Dice | delta vs auto_box | low Dice < 0.1 | high Dice >= 0.5 |
| --- | ---: | ---: | ---: | ---: |
| single_auto_box | 0.3118 | +0.0000 | 31 | 30 |
| data_aware_selector_v1 | 0.3234 | +0.0115 | 28 | 29 |
| ridge_reliability_selector_v2 | 0.3263 | +0.0145 | 28 | 30 |
| uncertainty_reliability_selector_v3 | 0.3279 | +0.0160 | 27 | 30 |
| oracle_auto_family | 0.3584 | +0.0466 | 21 | 34 |

## Interpretation

If v3 exceeds v2, prompt-induced uncertainty is useful and should remain part of Phase3.
If v3 is close to v2, the stability features are not enough on 40 cases but still provide an interpretable reliability signal for the paper.
If v3 drops below v2, the current uncertainty proxy is too noisy and should be replaced by perturbation-based sampling.
