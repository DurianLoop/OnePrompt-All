# Adaptive four-budget prompt policy — frozen experiment protocol

## Research question

When foreground–background matching has already proposed positive prompts, should SAM always receive a fixed number of background points, or should the number be selected per support–query response? Existing runs only tested a binary `N0`/`N5` response. This protocol tests the missing action space `A={N0,N5,N10,N20}` while freezing the matcher, positive points, layout, SAM backbone, and nested splits.

## Hypothesis

An anchored two-stage response policy with four negative-point budgets will improve held-out union Dice over both the frozen two-action policy and every fixed budget, because extra background points help only a subset of reference–query episodes.

## Leakage control

For each of the five existing nested folds:

1. Train ExtraTrees gain predictors on the frozen inner-train episodes only.
2. Select the anchor, tree leaf sizes, and gain threshold only on the 30 inner-validation episodes.
3. Apply the fixed fold policy once to outer-test episodes. Ground-truth masks are used only for metric computation after decoding.
4. Aggregate the five support references for each of 37 outer queries, then bootstrap query-level deltas with seed `20260715` and 10,000 resamples.

All four action outcomes had been materialized in the original negative-budget ablation; this experiment makes no new test-time prompt generator, SAM, or label-dependent decision.

## Completed result

The adaptive four-action selector used `N10` or `N20` in 26/185 outer reference episodes (14.1%; 9/37 queries). It achieved Dice 0.55071, exceeding:

- frozen two-action selector by +0.00315, 95% bootstrap CI [0.00074, 0.00633];
- best fixed budget (`N5`, Dice 0.53837) by +0.01234, CI [0.00338, 0.02234].

The per-episode ground-truth oracle ceiling is 0.55773, leaving a +0.00702 oracle gap. That ceiling is diagnostic only, not a deployable comparator.

The direct tree-spread risk gate and cross-reference consensus gate were also run and retained as negative controls: neither improved on the frozen two-action selector. This narrows the claim to **adaptive action-space expansion**, not an unsupported uncertainty-calibration claim.

## Reproducibility artifacts

- Training: `icassp/scripts/train_multibudget_negative_policy.py`
- Five-fold aggregation: `icassp/scripts/aggregate_multibudget_policy.py`
- Fixed-budget/oracle ablation: `icassp/scripts/analyze_multibudget_fixed_action_ablation.py`
- Outputs: `icassp/runs/multibudget_negative_policy_v1/`
