# Abstract

Visual-reference segmentation pipelines that condition a frozen segmenter on
a single reference image are widely deployed, yet the choice of reference is
rarely audited. We study how much a *valid* reference affects prediction,
how this instability should be measured, and whether a simple risk-aware
strategy can mitigate it without retraining the matcher or segmenter. We
introduce a reference-resampling protocol in which each query receives six
patient-disjoint references, resampled across ten seeds, and a corrected
variance decomposition `V_total = V_query + V_ref` that isolates the
reference-attributable component. On a pre-registered MoNuSAC holdout
(n=40, 10 seeds, 2,400 reference assignments, zero leakage) the reference
fraction of total variance is 0.061–0.263, 92.5% of queries show a
best-worst gap ≥ 0.05, and mean consensus over six references gains
+0.027 Dice (paired bootstrap 95% CI lower bound 0.014) over a random
single reference while reducing the catastrophic-failure rate. The
instability is predictable: a within-reference disagreement feature
correlates with the best-worst gap at Spearman |ρ|=0.724. Cross-dataset
replication shows the phenomenon is dataset-dependent, with strong support
on MoNuSeg (9/9 conditions) but failure on IDRiD, where consensus *hurts*
Dice (−0.012, CI lower −0.032). We then study two risk-aware mitigations.
Dynamic reference budgeting (M1) collapses to "always use all references"
under cost-quality optimization and fails its gate. In contrast, a
dual-model selective predictor (M2) — forced by an empirically measured
anti-correlation between instability and catastrophic failure — passes its
pre-registered holdout gate (6/6), reducing catastrophic failures by 79.2%
at 80% coverage while slightly improving Dice, using four test-time-safe
features and a single logistic regression per axis. The work delivers a
holdout-confirmed benchmark, a cross-dataset failure boundary, and a
deployable selective-prediction mitigation.
