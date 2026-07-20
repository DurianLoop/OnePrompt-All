# Stage 3 Plan: Offline Utility Policy Learning

## Goal

Learn an inference-time prompt-rule policy without differentiating through
Top-K selection or SAM.  The policy observes only frozen matching evidence and
fixed action metadata; ground-truth masks are used offline only to supervise
the utility target.

## Splits and leakage boundary

Keep the Stage 2 seeded protocol unchanged:

- one MoNuSeg training case is the fixed reference;
- six training cases are policy validation;
- the remaining 30 training cases are policy training;
- all 14 official test cases are held out until final evaluation.

For every train and validation query, evaluate every prespecified fixed action
with SAM and store its Dice as offline utility.  No test Dice enters model
fitting, feature normalization, action choice, or hyperparameter selection.

## Policy input and target

For each query/action pair, construct a deterministic feature vector from the
frozen foreground/background matching maps:

- contrast distribution moments and quantiles;
- foreground/background agreement and separation statistics;
- top prompt-score gaps and spatial dispersion;
- action metadata: positive count, negative count, and background weight.

The target is the action's observed Dice.  A compact ExtraTrees utility
regressor is fit on train query/action pairs so that matching evidence can
interact nonlinearly with an action.  Its leaf-size regularization is selected
strictly by validation utility ranking; at inference the action with highest
predicted utility is emitted as the prompt policy decision.

## Outputs

- train/validation/test policy-feature CSVs;
- case/action utility CSV with the source split recorded;
- fit metadata, validation selection evidence, and test predictions;
- a policy-vs-fixed-rule JSON summary suitable for later paper tables.

## Interpretation

This first implementation is a deliberately small offline policy learner.  It
tests the central claim—query-adaptive prompt selection—before adding richer
architectures or more datasets.  It must be described as a single-dataset
pilot until repeated over multiple reference seeds and external datasets.
