# Phase S Research Charter

Date: 2026-07-22
Branch: `phaseS-multitrack-reference-robust-research`

## Research Question

> Can reference-induced instability in visual-reference segmentation be
> measured correctly, predicted from test-time-safe evidence, and mitigated
> through robust aggregation, reliability modeling, risk-aware prompting, or
> direct reference-conditioned segmentation?

## Allowed Final Outputs

1. Method paper
2. Benchmark / diagnostic paper
3. Benchmark + mitigation method hybrid paper
4. Failure-boundary paper with generalizable findings

## Core Principles

1. Strict data-leakage control: train/val/holdout/test roles frozen.
2. No validation-set overfitting: hyperparameters selected on train only.
3. No post-hoc threshold tuning after gate evaluation.
4. Honest negative results: a Gate failure is a scientific finding, not a bug.
5. No manufactured success via threshold lowering, holdout peeking, or claim inflation.
6. Statistical units = query/case; reference episodes are repeated measurements.
7. Formal test locked for the entire Phase S program.

## Prohibited Actions (directive §2)

- Looking only at mean Dice, ignoring within-query reference variance / tail risk.
- Confusing cross-query variance with reference-induced variance.
- Treating Search Oracle as a deployable method.
- Training policy on GT-guided search labels.
- Endlessly expanding K, grouping, or action bank.
- Adding modules repeatedly on the same validation set.
- Tuning after holdout observation.
- Drawing "universal" conclusions from MoNuSAC alone.
- Using formal test to decide route.
- Treating PDF generation as submission completion.

## Programmatic Constraints

- Total compute cap: 210 GPU-hours.
- Budget checkpoints: 50 / 100 / 160 / 210 GPU-hours.
- Must reuse DINOv2 features, matching maps, candidate pools, SAM output cache.
- Holdout may be observed exactly once per frozen method.
- Cross-dataset (GlaS / DDR) only after holdout pass.
- No internet downloads without user approval.

## Track Priority

1. Track A must complete first.
2. Track B is the primary method route.
3. Track C only if B has signal and plateaus.
4. Track D is the fallback if prompt interface is the bottleneck.
5. After any track passes a strong Gate, weaker tracks may be paused.
6. No unbounded four-track expansion.
