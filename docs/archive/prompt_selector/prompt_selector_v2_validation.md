# Prompt Selector v2 Validation

## Component

`prompt_selector_v2` is a lightweight reliability scorer inspired by prompt reranking, uncertainty-aware selection, and dynamic prompting components in the reviewed medical SAM papers.

It uses leave-one-case-out validation. GT Dice is used only on the training cases to fit the scorer; the held-out case is selected without GT.

Features:

- dataset one-hot
- model one-hot
- prompt protocol one-hot
- SAM/SAM2 raw score when available
- score missing flag
- predicted mask area ratio
- small/large mask flags

## Results

| method | mean Dice | delta vs auto_box | low Dice < 0.1 | high Dice >= 0.5 |
| --- | ---: | ---: | ---: | ---: |
| single_auto_box | 0.3118 | +0.0000 | 31 | 30 |
| data_aware_selector_v1 | 0.3234 | +0.0115 | 28 | 29 |
| ridge_reliability_selector_v2 | 0.3263 | +0.0145 | 28 | 30 |
| oracle_auto_family | 0.3584 | +0.0466 | 21 | 34 |

## Interpretation

- If v2 is above `data_aware_selector_v1`, the learned reliability route is already useful on 40 cases.
- If v2 is below `data_aware_selector_v1`, the feature set is too weak and the next component should be prompt-induced uncertainty or mask stability, not a larger backbone.
- The gap to `oracle_auto_family` is the remaining Phase3 research space.

## Phase3 Completion Criterion

Phase3 is considered complete for the small-project milestone when:

1. selector is evaluated without GT on held-out cases;
2. it is compared against fixed auto prompt and data-aware rules;
3. the oracle gap is reported honestly;
4. the next component is chosen based on failure evidence.
