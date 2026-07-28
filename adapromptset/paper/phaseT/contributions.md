# Contributions

This paper makes the following evidence-based contributions. Each is tied to
a specific Phase T result file and to an allowed claim in the Phase T claim
registry; no contribution relies on a prohibited claim.

1. **A holdout-confirmed reference-induced instability benchmark.**
   We introduce a reference-resampling protocol (six patient-disjoint
   references per query, ten seeds, k∈{1,3,6}) and a corrected variance
   decomposition `V_total = V_query + V_ref`. On a pre-registered, hash-
   verified MoNuSAC holdout (n=40, 10 seeds, 2,400 reference assignments,
   zero patient leakage), the reference fraction of total variance is
   0.061–0.263, 92.5% of queries have a best-worst gap ≥ 0.05, and a
   within-reference disagreement feature predicts the gap at Spearman
   |ρ|=0.724. All 10 pre-registered gate conditions pass.
   *(Evidence: WP1 10/10 PASS.)*

2. **A cross-dataset failure boundary for the phenomenon.**
   We replicate the protocol on MoNuSeg (strong support, 9/9 conditions,
   consensus gain +0.015, CI lower 0.007) and on IDRiD (FAIL, 4/5
   directional; consensus *hurts* Dice by −0.012, CI lower −0.032). The
   instability is real but dataset-dependent: where references are
   uniformly poor, mean consensus can reduce Dice. This bounds the claim
   and is itself a scientific finding.
   *(Evidence: WP2 MoNuSeg 9/9, WP2 IDRiD 4/5.)*

3. **The anti-correlation finding and a dual-model selective predictor (M2,
   holdout-confirmed).** We document that instability (high best-worst gap)
   and catastrophic failure (low Dice) are anti-correlated on the same
   feature space, so a single predictor cannot satisfy both an instability-
   AUROC gate and a catastrophic-risk selective gate. This forces a
   dual-model design: a logistic regression on four test-time-safe
   disagreement features, trained separately on the instability and the
   catastrophic-failure labels. The frozen dual predictor passes its
   pre-registered holdout gate (6/6): instability AUROC 0.928,
   catastrophic AUPRC 0.528 (prevalence 0.050), ECE 0.027, catastrophic-
   failure reduction 79.2% at 80% coverage, Dice change +0.009.
   *(Evidence: WP3 M2 5/5 validation + 6/6 holdout.)*

4. **A documented negative result for dynamic reference budgeting (M1).**
   Cost-quality-optimized dynamic reference budgeting collapses to
   `lambda_c=lambda_r=0` (i.e., always use all six references) across all
   five policy variants. Cost-saving mode fails (avg_k=5.63, only 6.1%
   reduction versus the 25% gate) and quality-cost mode fails. The
   finding bounds the dynamic-budget paradigm under the frozen DINOv2-L +
   SAM-B pipeline.
   *(Evidence: WP3 M1 NO_GAIN.)*

5. **A reproducible benchmark protocol, metric suite, and pre-registered
   gate structure.** We release the reference-resampling protocol, the
   V_ref / CVaR / CFR / best-worst-gap metric suite, the patient-cluster
   paired-bootstrap procedure, and the pre-registered WP1 / WP2 / WP3 gate
   conditions. The protocol is dataset-agnostic; we report where it
   replicates and where it does not, and we document (not fabricate) the
   external comparator status.
   *(Evidence: phaseT reference/metric/holdout protocols; claim registry
   audit.)*
