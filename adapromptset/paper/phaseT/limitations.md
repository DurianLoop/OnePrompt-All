# Limitations

We acknowledge several limitations. None invalidate the contributions,
but each bounds the claim and is reported honestly.

**Single primary dataset.** The holdout-confirmed benchmark is on MoNuSAC
alone. The holdout is 40 queries (8 patients), which is small in absolute
terms even though it is pre-registered, patient-disjoint, and
hash-verified. The cross-dataset replications on MoNuSeg (30 queries) and
IDRiD (30 queries) provide directional and strong support on the
pathology-morphology side, but they are smaller still and use five seeds
rather than ten. The pre-registered gate structure is designed to make
small-sample claims falsifiable, but a small sample is still a small
sample; a larger holdout would tighten the CIs.

**Frozen matcher and segmenter.** All results are under DINOv2-L matching
+ frozen SAM-B segmentation. We do not know whether the magnitude of
`V_ref`, the anti-correlation between instability and catastrophic
failure, or the M2 selective-prediction gains transfer to other matchers
(e.g., DINOv2-G, MAE-trained encoders) or other segmenters (e.g., SAM-L,
MedSAM, SAM 2). The reference-resampling protocol and the metric suite
are matcher- and segmenter-agnostic, but the *numbers* are not. We make
no "model-agnostic" claim; the prohibited-claim registry explicitly
forbids it.

**Cross-dataset failure is part of the finding.** IDRiD fails the
consensus-benefit gate: mean consensus hurts Dice by −0.012 (CI lower
−0.032). We do not claim universality, and we report the failure as a
boundary. But the boundary itself is under-explained: we hypothesize
that uniformly poor references (CFR 0.71–1.00 on IDRiD) make aggregation
unhelpful, but we have not isolated whether the cause is the lesion
intent, the reference-pool size, the matcher's failure on retinal
imagery, or a combination. A follow-up ablation across lesion intents
and matchers would be needed.

**M2 is calibrated but not risk-free.** The M2 holdout result is strong
(CFR reduction 79.2% at 80% coverage, ECE 0.027), but it does not
*eliminate* catastrophic failure: CFR at 80% coverage is 0.010, not
zero. The remaining failures are the hardest cases — episodes the model
rates as low-risk but that fail anyway — and a deployed system would
need a human-in-the-loop policy for the abstained 20%. We do not claim
that M2 "solves" instability; the prohibited-claim registry explicitly
forbids that word. M2 is a risk-aware abstention layer, not a fix.

**M2 calibration does not transfer cross-dataset.** Supplementary
transfer evaluation (not a gate condition) shows that M2's *ranking*
generalizes: instability AUROC is 0.874 on MoNuSeg and 0.994 on IDRiD.
But M2's *calibration and threshold* do NOT transfer to IDRiD: ECE
jumps from 0.027 (MoNuSAC holdout) to 0.477 (IDRiD), and CFR reduction
drops from 79.2% to 6.0%. The root cause is distribution shift —
IDRiD's catastrophic prevalence is 0.807 versus MoNuSAC train's 0.117,
so the frozen threshold misclassifies most IDRiD episodes. Deploying M2
on a new dataset would require target-domain recalibration on a small
labeled set. We do not claim M2 is cross-dataset deployable as-is.

**M1 negative is bounded to this pipeline.** The M1 NO_GAIN result
reflects the cost-quality utility we defined and the k∈{1,3,6} budget it
could choose from. A richer action space (e.g., per-reference weighted
aggregation, or k drawn from a continuous budget) might behave
differently. We do not claim M1 is universally hopeless; we claim that
under the frozen DINOv2-L + SAM-B pipeline and the pre-registered
cost-quality utility, dynamic budgeting collapses to "always k=6". A
different utility (e.g., one that prices matching cost more aggressively,
or that prices *risk* rather than reference count) might recover a
non-trivial policy.

**External comparators are best-effort, not exhaustive.** GF-SAM and
MAUP are identified as Tier 2 reproducible, but numerical reproduction
requires user authorization for external code download and is therefore
pending. We do not fabricate Tier 3 numerical results, and we do not
claim any external method is "fully outperformed" — that is a prohibited
claim. The internal B1–B7 families cover at least three distinct
aggregation families, satisfying the strong-internal-baselines provision
of the external-comparator minimum, but a reviewer who requires
head-to-head external numbers will not find them in this submission.

**No formal test access.** The formal test set (101 MoNuSAC cases) is
locked and unused for the entire Phase T program. Access requires a
separate Phase U freeze protocol and explicit user approval. All claims
in this paper are therefore bounded by holdout evidence, not by formal-
test evidence. We make no "formal-test success" claim, and no SOTA or
leaderboard claim.

**Medical imaging focus.** All three datasets (MoNuSAC, MoNuSeg, IDRiD)
are medical. Generalization to natural images is not tested. The
protocol is dataset-agnostic, but we report where it replicates
(MoNuSAC, MoNuSeg) and where it does not (IDRiD), and we do not
extrapolate beyond the tested domains.

**Qualitative figures are rule-selected, not hand-picked.** Qualitative
figures are selected by pre-registered rules (worst-5, best-5, median-5
by `V_ref`), and the rules are frozen at pre-registration. This prevents
cherry-picking but also means the qualitative figures may not be the
most visually compelling; they are the most *representative* under the
rule.

**Single-instability-axis assumption.** M2's dual-model design assumes
that two axes (instability and catastrophic failure) suffice to
characterize reference-induced risk. Other axes (e.g., boundary-quality
instability, instance-count instability) may exist and may not be
captured by the four test-time-safe features. We do not claim the four
features are complete; we claim they are sufficient to pass the
pre-registered gate.
