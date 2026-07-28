# Phase S Claim Registry

Date: 2026-07-22
Branch: `phaseS-multitrack-reference-robust-research`

## Allowed Claims by Gate Outcome

### Track A Passes (instability benchmark valid)

> Valid references induce substantial, measurable segmentation variability
> under the frozen matching-and-prompting pipeline, and this instability is
> associated with test-time-safe disagreement features.

### Track B Passes (reliability weighting pass)

> Test-time-safe reference reliability estimation, trained on train-only
> utility labels, reduces within-query reference variance and improves
> segmentation robustness under a single SAM call.

### Track B Utility-Only Pass

> Reference utility is predictable from test-time-safe features, but weighted
> consensus does not reduce variance; the utility signal is a scientific
> finding without a deployable stability gain.

### Track B Stability-Only Pass

> Reliability weighting reduces within-query variance but does not improve
> mean Dice; the stability gain does not translate to accuracy gain.

### Track C Passes

> Set-level reference modeling captures interactions beyond per-reference
> scoring and further improves robustness.

### Track D Passes

> Direct reference-conditioned segmentation bypasses the point-prompt
> bottleneck and outperforms prompt-based methods on reference-robust
> segmentation.

### Holdout Passes

> The frozen method confirms on an unseen internal holdout and is ready for a
> separately frozen formal-test protocol.

### Cross-Dataset Passes

> The method's directionally consistent improvement transfers to a secondary
> dataset (GlaS / DDR).

## Permanently Prohibited Claims

Regardless of Gate outcomes, the following claims are PERMANENTLY PROHIBITED:

1. ❌ "Reference instability solved" / "eliminated"
   (Rationale: variance reduction is bounded, not elimination.)
2. ❌ "Universal across datasets"
   (Rationale: only MoNuSAC + at most one secondary dataset tested.)
3. ❌ "SOTA" / "state-of-the-art segmentation"
   (Rationale: no leaderboard comparison; formal test not accessed.)
4. ❌ "Formal test success"
   (Rationale: formal test locked and unused.)
5. ❌ "ProSAM/MAUP/SAMIX fully outperformed"
   (Rationale: no head-to-head under matched protocol.)
6. ❌ "Reference weighting is model-agnostic"
   (Rationale: only tested with DINOv2-L + SAM-B.)
7. ❌ "Mean Dice alone proves robustness"
   (Rationale: robustness requires variance / tail-risk evidence.)
8. ❌ "Reference instability is purely random"
   (Rationale: Phase R showed systematic component; deterministic consensus
   did not reduce variance.)
9. ❌ "The method works without references"
   (Rationale: all tracks are reference-conditioned.)
10. ❌ "Submission-ready" / "publication-completed"
    (Rationale: PDF generation ≠ submission; mock review required.)

## Claim Boundary Auditing

All Phase S reports, tables, figures, and markdown outputs will be scanned
for prohibited tokens before final commit. Any hit blocks the final state
transition.
