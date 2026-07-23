# Phase T Claim Registry

Date: 2026-07-23
Branch: `phaseT-publication-completion`
Directive: `D:\SAM\adapromptset\reports\AdaPromptSet_PhaseT_Publication_Completion.md`

## Allowed Claims

The following claims are ALLOWED, each contingent on the listed evidence
condition being satisfied. No allowed claim may be made before its evidence
condition is met.

### (a) Track A Instability Benchmark Claim

> Valid references induce substantial, measurable segmentation variability
> under the frozen matching-and-prompting pipeline, and this instability is
> associated with test-time-safe disagreement features.

Evidence condition: Phase S Track A PASS (7/7 conditions, already established).
This claim is already supported and may be restated in the Phase T paper.

### (b) Track B Utility-Only Claim

> Reference utility is predictable from test-time-safe features, but weighted
> consensus does not reduce variance; the utility signal is a scientific
> finding without a deployable stability gain.

Evidence condition: Phase S Track B `NO_GAIN` with utility-prediction
Spearman ≥ 0.30 on validation. Already supported (validation Spearman 0.478).

### (c) Mean Consensus Strong Baseline Claim

> Mean consensus across multiple references is a strong, single-SAM-call
> automatic baseline that improves mean Dice and catastrophic-failure rate
> over random single-reference prompting, but does not eliminate
> within-query reference variance.

Evidence condition: Phase R Gate R2 cond1 / cond4 / cond5 PASS (Δ Dice
+0.0460, catastrophic rate 0.0732 vs 0.1220, A/B overlap 0.6075 vs 0.5942).
Already supported; Phase T WP1 must reconfirm on holdout.

### (d) Holdout-Confirmed Instability Claim (CONDITIONAL on WP1 pass)

> On an unobserved MoNuSAC holdout (n=40), reference-induced instability
> replicates with the preregistered magnitude, and mean consensus remains the
> strongest automatic baseline under the frozen protocol.

Evidence condition: WP1 Gate (directive §5.6) passes — all 10 conditions
satisfied, including V_ref_frac ≥ 0.10, ≥50% queries best-worst gap ≥ 0.05,
Δ Dice (consensus vs random single) ≥ 0.010 with paired bootstrap 95% CI
lower > 0, and disagreement-feature Spearman |ρ| ≥ 0.25.

### (e) Risk-Aware Mitigation Claim (CONDITIONAL on M1/M2 pass)

> A simple risk-aware strategy — dynamic reference budget (M1) and/or
> instability-aware selective prediction (M2) — improves the cost-quality
> frontier or the risk-coverage trade-off without retraining the frozen
> matcher and segmenter, and the improvement is holdout-confirmed.

Evidence condition: WP3 M1 Gate (directive §9.6) OR M2 Gate (directive §10.5)
passes on validation, AND the passing method confirms directionally on the
MoNuSAC holdout. For M1: cost-saving mode OR quality-cost mode. For M2:
instability AUROC ≥ 0.70, ECE ≤ 0.10, and CFR relative reduction ≥ 30% at
80% coverage.

## Permanently Prohibited Claims

The following 10 claims are PERMANENTLY PROHIBITED regardless of Gate
outcomes. They may not appear in any Phase T report, table, figure, abstract,
or terminal reply except inside this registry as documentation of the
prohibition itself.

1. ❌ **"Reference instability solved" / "eliminated"**
   Rationale: variance reduction is bounded, not elimination; Phase S Track B
   and Track C both showed NO_GAIN on the central stability hypothesis.

2. ❌ **"Universal across datasets"**
   Rationale: only MoNuSAC (primary) + at most GlaS / MoNuSeg / DDR (secondary
   / tertiary) tested; results may be dataset-specific.

3. ❌ **"SOTA" / "state-of-the-art segmentation"**
   Rationale: no leaderboard comparison; formal test not accessed; frozen
   DINOv2-L + SAM-B is not a SOTA segmenter.

4. ❌ **"Formal test success"**
   Rationale: formal test locked and unused for the entire Phase T program.

5. ❌ **"ProSAM / MAUP / SAMIX fully outperformed"**
   Rationale: no head-to-head under a matched reference-resampling protocol;
   external comparator reproduction is best-effort, not exhaustive.

6. ❌ **"Reference weighting is model-agnostic"**
   Rationale: only tested with DINOv2-L + SAM-B; Track B already showed
   weighting does not improve Dice even in this single pipeline.

7. ❌ **"Mean Dice alone proves robustness"**
   Rationale: robustness requires variance / tail-risk evidence (V_ref, CVaR,
   CFR, best-worst gap); mean Dice without variance is a prohibited
   over-claim.

8. ❌ **"Reference instability is purely random"**
   Rationale: Phase S showed the instability is predictable (validation
   Spearman 0.478); a purely random phenomenon would not be predictable.

9. ❌ **"The method works without references"**
   Rationale: all WP3 mitigation routes are reference-conditioned; M1 selects
   reference count and M2 relies on reference-disagreement features.

10. ❌ **"Submission-ready" / "publication-completed"**
    Rationale: PDF generation ≠ submission; the Publication Readiness Gate
    (§20) must pass before any `PAPER_CANDIDATE_READY` state is written, and
    even then the paper is a candidate, not a completed publication.

## Conditional vs Unconditional Claims

- Claims (a), (b), (c) are UNCONDITIONAL given Phase S evidence — they may be
  restated in the Phase T paper introduction / related work.
- Claims (d), (e) are CONDITIONAL on Phase T WP1 / WP3 outcomes — they may
  only be made after the corresponding Gate passes and the holdout
  confirmation is logged.
- All prohibited claims remain prohibited regardless of any Gate outcome.

## Claim Boundary Auditing

All Phase T reports, tables, figures, and markdown outputs will be scanned
for prohibited tokens before the final state transition. Any hit blocks the
final state. The scan covers exact phrases and case-insensitive substring
matches, with documented exceptions for this registry and for legitimate
negated occurrences.
