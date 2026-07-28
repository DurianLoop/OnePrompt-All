# Conclusion

We asked three questions: how much does a valid reference affect
visual-reference segmentation, how should the resulting instability be
measured, and can a simple risk-aware strategy mitigate it without
retraining the frozen matcher and segmenter?

**On measurement.** Reference-induced instability is real, substantial,
and predictable. On a pre-registered, patient-disjoint MoNuSAC holdout
(n=40, 10 seeds, 2,400 reference assignments, zero leakage), the
reference-attributable fraction of total variance is 0.061–0.263, 92.5%
of queries show a best-worst gap ≥ 0.05, and a within-reference
disagreement feature predicts the gap at Spearman |ρ|=0.724. Mean
consensus over six references gains +0.027 Dice over a random single
reference (paired bootstrap 95% CI lower bound 0.014) while reducing
the catastrophic-failure rate and improving prediction consistency. The
corrected variance decomposition `V_total = V_query + V_ref`, the
tail-risk metric suite (CVaR, CFR, worst-subset, best-worst gap), and
the patient-cluster paired bootstrap together convert a hidden degree
of freedom into a measured, falsifiable phenomenon. Mean Dice alone is
not a robustness claim; we never report it without variance or tail-
risk evidence.

**On the boundary.** The phenomenon is dataset-dependent. The protocol
replicates with strong support on MoNuSeg (9/9 conditions, consensus
gain +0.015, CI lower 0.007) but fails on IDRiD, where mean consensus
*hurts* Dice (−0.012, CI lower −0.032) because references are uniformly
poor. We do not claim universality; the failure is part of the
contribution, bounding the claim and delineating where reference
aggregation helps and where it does not.

**On mitigation.** The two routes have opposite outcomes, and both are
informative. Dynamic reference budgeting (M1) collapses to "always use
all six references" under cost-quality optimization (`λ_c=λ_r=0` for all
five policy variants) and fails its gate (avg_k 5.63, only 6.1%
reduction versus the 25% target). The dynamic-budget paradigm cannot
meaningfully reduce references without hurting Dice under this pipeline.
In contrast, a dual-model selective predictor (M2) — forced by an
empirically measured anti-correlation between instability and
catastrophic failure — passes its pre-registered holdout gate (6/6):
instability AUROC 0.928, catastrophic AUPRC 0.528 (prevalence 0.050),
ECE 0.027, catastrophic-failure reduction 79.2% at 80% coverage, Dice
change +0.009. The dual-model design is structural, not preferential: a
single predictor cannot satisfy both an instability-AUROC gate and a
catastrophic-risk selective gate on this feature space. M2 uses four
test-time-safe features and a logistic regression per axis, requires no
query ground truth at deployment, and does not retrain the matcher or
segmenter.

**On what this paper is not.** This is not a state-of-the-art
segmentation paper. The matcher (DINOv2-L) and segmenter (SAM-B) are
frozen, the formal test set is locked and unused, and we make no
leaderboard claim. We do not claim to have solved reference instability,
to be universal across datasets, or to have fully outperformed any
external method. External comparator reproduction is best-effort and
pending authorization; we do not fabricate Tier 3 results.

**Recommended next step.** The M2 holdout confirmation (6/6) is a
candidate for a Phase U formal-test freeze under a separate protocol
and explicit user approval. Until then, the result is a holdout-
confirmed, pre-registered mitigation that reduces catastrophic failure
by 79.2% at 80% coverage on the frozen pipeline, plus a documented
negative result for dynamic budgeting and a cross-dataset failure
boundary. The benchmark protocol, metric suite, and gate structure are
released for reuse.
