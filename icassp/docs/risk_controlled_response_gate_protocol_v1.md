# Risk-controlled response gate — frozen experiment protocol v1

## Research gap and hypothesis

Automatic-prompt SAM work has studied uncertainty maps, prompt augmentation, and prompt localization, but the deployed decision “should a harmful optional background intervention be accepted at all?” is usually optimized for mean Dice without an explicit harm/coverage analysis. The existing nested MoNuSeg gate already produces an ExtraTrees mean predicted gain and tree-wise standard deviation, but its saved decision uses uncertainty weight zero.

**Hypothesis.** A lower-confidence-bound (LCB) response gate can reduce the rate of harmful accepted N5 interventions relative to the mean-only gate while retaining a non-negative held-out Dice change against the validation-selected N0/N5 anchor.

This is a selective-intervention study, not a claim of conformal coverage or clinical safety.

## Frozen inputs

- Source folds: `runs/research_direction/nested_cv5_negative_budget/fold_{0..4}`.
- Existing train-only ExtraTrees models and their fixed N0/N5 candidate outputs.
- Inner validation: each fold's `policy/val_learned_predictions.csv` only.
- Outer evaluation: each fold's `policy/test_learned_predictions.csv` only after all fold-local choices are written.
- The action space, DINOv2-L matcher, SAM ViT-B, K=40, lambda=0.75, split files, and five-reference aggregation remain unchanged.

## Gate family

For an alternative action relative to an inner-validation-selected anchor, let `mu` be the saved post-SAM predicted gain and `sigma` its saved tree-wise predicted-gain standard deviation. The LCB score is

`score = mu - beta * sigma`.

The alternative is accepted only when `score > tau`; otherwise the anchor is retained. Candidate pairs are:

- `beta`: 0, 0.25, 0.5, 1.0, 2.0;
- `tau`: -0.02, 0, 0.005, 0.01, 0.02, 0.05.

`beta=0` is the mean-only family member. A switch is **harmful** if its observed Dice gain over the anchor is strictly negative; ties are non-harmful.

## Fold-local selection rule

For every candidate pair, calculate validation mean Dice, switch coverage, harmful-switch count, and the one-sided 95% Wilson upper bound for harmful-switch rate among switched episodes. A pair is feasible if:

1. it switches at least three validation episodes; and
2. its Wilson upper bound is at most 0.50.

Among feasible pairs choose largest validation mean Dice; ties use smaller Wilson bound, then larger `beta`, then larger `tau`. If no pair is feasible, choose the anchor-only policy. The intentionally modest 0.50 upper-bound constraint is appropriate for the 30-episode validation sample and is evaluated as a *risk-sensitive selection heuristic*, not a statistical guarantee.

## Primary outer measurements

After choices are frozen for all five folds, average five reference episodes per 37 outer queries, then report:

1. Dice change versus the fold anchor with paired bootstrap CI;
2. query coverage (any reference episode switched);
3. harmful-switch episode rate and Wilson upper bound;
4. comparison to the existing mean-only saved gate at the same frozen candidate/action space.

No outer-test labels enter selection, calibration, or model fitting.

## Interpretation rules

- Positive: lower harmful-switch rate than mean-only with non-negative Dice CI lower bound.
- Trade-off: lower harmful-switch rate but Dice CI crossing/below zero.
- Failure: no risk reduction or worse risk and Dice.

Any outcome is retained. The protocol does not modify the primary evidence package unless an independent audit confirms no outer-label access before decisions.

## Literature positioning consulted before freeze

- UR-SAM studies uncertainty rectification through prompt augmentation, rather than an action-level LCB accept/defer policy: https://arxiv.org/abs/2311.10529
- StratCP studies calibrated act/defer decisions for medical foundation models, but not reference-guided SAM background-prompt intervention: https://zitniklab.hms.harvard.edu/projects/StratCP/
- Curriculum prompting studies progressive prompt integration, rather than harm-controlled acceptance of an optional N5 intervention: https://arxiv.org/abs/2409.00695
