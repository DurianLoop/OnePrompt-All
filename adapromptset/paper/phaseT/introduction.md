# Introduction

Visual-reference segmentation pipelines have become a practical recipe for
adapting a frozen generalist segmenter to a target domain: a matcher
retrieves spatial correspondences between a query image and a labeled
reference, the reference's mask is projected into point prompts, and a
frozen segmenter (e.g., SAM) decodes a mask in a single call. The recipe is
attractive because it requires no segmenter fine-tuning and no labeled
query-time data. Yet it shares a hidden degree of freedom with classical
few-shot segmentation — the choice of *reference* — and that degree of
freedom is almost never audited. Practitioners pick one reference, run the
pipeline, and report a mean Dice.

This paper asks three questions about that practice. *How much* does the
choice of a valid reference affect the resulting segmentation? *How should*
that instability be measured, given that references are repeated
measurements on the same query rather than independent samples? And *can* a
simple, risk-aware strategy mitigate the worst outcomes without retraining
the frozen matcher (DINOv2-L) or the frozen segmenter (SAM-B)?

The motivation is empirical. In our pipeline, two valid references for the
same MoNuSAC query can produce Dice scores that differ by more than 0.20,
and a non-trivial fraction of episodes end in catastrophic failure (Dice
below 0.1) under one reference but not another. If a deployed system
conditions on a single reference, the reported mean Dice hides a tail risk
that is attributable not to the query, not to the segmenter, and not to the
matcher, but to a *legitimate* reference choice. This is the kind of risk
that reliability-focused deployment — clinical pathology in our case —
cannot afford to ignore, and that mean-Dice reporting systematically masks.

Measuring this risk requires three corrections to common practice. First,
the statistical unit must be the query, not the reference episode:
references are repeated measurements, and treating them as independent
samples inflates the apparent sample size. Second, the variance must be
*decomposed* into a query-attributable component `V_query` and a
reference-attributable component `V_ref`, because a raw cross-episode
variance conflates "hard queries" with "unstable references". Third, the
*tail* of the per-reference Dice distribution must be reported alongside
the mean — a conditional value-at-risk (CVaR) at the 10% tail, a
catastrophic-failure rate (CFR) at a frozen threshold, and a per-query
best-worst gap — because two systems with the same mean Dice can have very
different worst-reference behaviour.

With these corrections in place, the phenomenon is not only real but
predictable. A within-reference disagreement feature — the expected IoU
between predictions produced by two different single-reference subsets for
the same query — correlates with the best-worst gap at Spearman |ρ|=0.724
on a pre-registered MoNuSAC holdout. Predictability matters: it means the
instability is not purely random, and it opens the door to a *risk-aware*
mitigation that abstains on high-risk episodes rather than trying to fix
them.

We therefore frame the mitigation problem as risk-aware decision-making
over the reference axis, with the matcher and segmenter frozen. We study
two complementary routes. The first, *dynamic reference budgeting* (M1),
asks whether a cost-quality-optimized policy can choose how many references
to use per query and thereby trade cost against quality. The second,
*instability-aware selective prediction* (M2), asks whether a lightweight
predictor on test-time-safe features can flag episodes that are likely to
end in catastrophic failure, so the system can abstain at deployment.

The two routes have opposite outcomes, and both are scientifically
informative. M1 fails: cost-quality optimization selects `lambda_c=lambda_r=0`
across all five policy variants, i.e. the optimum is "always use all six
references". Cost-saving mode reduces the reference count by only 6.1%
versus the 25% gate, and quality-cost mode fails. The dynamic-budget
paradigm cannot meaningfully reduce references without hurting Dice under
this pipeline. M2 passes: a dual-model selective predictor — a logistic
regression on four test-time-safe disagreement features — passes its
pre-registered holdout gate (6/6), reducing the catastrophic-failure rate
by 79.2% at 80% coverage while slightly improving Dice (ECE 0.027). The
dual-model design is not an engineering preference; it is *forced* by an
empirically measured anti-correlation between instability (high best-worst
gap) and catastrophic failure (low Dice), which prevents a single
predictor from satisfying both an instability-AUROC gate and a
catastrophic-risk selective gate.

We also report a cross-dataset failure boundary that bounds the claim. The
protocol replicates with strong support on MoNuSeg (9/9 conditions,
consensus gain +0.015, CI lower 0.007) but *fails* on IDRiD (4/5
directional), where mean consensus *hurts* Dice (−0.012, CI lower −0.032).
The instability is real but not universal: where references are uniformly
poor, aggregating them does not help. We do not claim universality, and we
do not claim to have solved reference instability; we claim to have
measured it, bounded it, and mitigated its worst outcomes on a frozen
pipeline.

**Scope and limitations of claims.** This is not a state-of-the-art
segmentation paper. The matcher (DINOv2-L) and segmenter (SAM-B) are
frozen, the formal test set is locked and unused, and we make no
leaderboard comparison. Mean Dice alone is never used as a robustness
claim; every robustness statement is accompanied by V_ref, CVaR, CFR, or
best-worst-gap evidence. External comparator methods (GF-SAM, MAUP, and
others) are identified by tier and documented; numerical reproduction
requires user authorization for external code download and is reported as
best-effort, not exhaustive.

**Contributions preview.** (i) A holdout-confirmed reference-induced
instability benchmark with a corrected variance decomposition and a
tail-risk metric suite. (ii) A cross-dataset failure boundary (MoNuSeg
strong support, IDRiD failure). (iii) A dual-model selective predictor
(M2) that is holdout-confirmed and deploys four test-time-safe features.
(iv) A documented negative result for dynamic reference budgeting (M1).
(v) A reproducible, pre-registered benchmark protocol, metric suite, and
gate structure. The remainder of the paper develops the protocol
(Sections 3–4), the benchmark and mitigation results (Sections 5–9), the
failure taxonomy and limitations (Sections 10–11), and the conclusion
(Section 12).
