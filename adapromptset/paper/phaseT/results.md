# Results

We report results in four parts: WP1 MoNuSAC holdout, WP2 cross-dataset
replication, WP3 M2 selective prediction (with holdout confirmation), and
WP3 M1 dynamic budget. All numbers are taken from the frozen gate-summary
files; no parameter was retuned after holdout observation. Mean Dice is
never reported alone: every robustness statement is accompanied by
variance or tail-risk evidence.

## 1. WP1 — MoNuSAC holdout (10/10 PASS)

**Within-query reference variance.** The reference-attributable fraction
of total variance, `V_ref_frac`, ranges 0.061–0.263 across the three
prompt configs and k∈{1,3,6}. At k=1 (single-reference baseline),
`V_ref_frac` is 0.263 for P20-N1, 0.211 for P30-N0, and 0.126 for P50-N0;
at k=6 (full-pool consensus) it drops to 0.189, 0.159, and 0.061
respectively. The mean within-query reference std at k=1 is 0.074 (gate
threshold 0.02). Reference-induced variance is a substantial fraction of
total variance even at the most heavily prompted config, and it shrinks —
but does not vanish — under consensus.

**Best-worst gap.** 92.5% of holdout queries (37/40) have a mean
best-worst gap ≥ 0.05 (gate threshold 50%). The mean best-worst gap at
k=1 is 0.239 for P20-N1, 0.214 for P30-N0, and 0.167 for P50-N0. In
plain terms: for the overwhelming majority of queries, the best and
worst valid references differ in Dice by more than 0.05, and often by
more than 0.20.

**Mean consensus vs random single.** Mean consensus over six references
(B2_mean, k=6) gains +0.027 Dice over the random single-reference
baseline (B1, k=1): 0.506 versus 0.479 averaged across configs. The
paired bootstrap 95% CI lower bound is 0.014 (n=40, 10,000 iterations,
patient-cluster resampling), well above zero. Consensus also improves the
tail: `CVaR_0.1` rises from 0.075 (B1) to 0.092 (B2, k=6); CFR drops
from 0.063 (B1) to 0.058 (B2, k=6); and prediction consistency `J_pred`
rises from 0.649 (B1, k=1) to 0.719 (B2, k=6). The catastrophic-failure
rate at k=6 ranges 0.050–0.073 across configs, all below the
corresponding k=1 values.

**Predictability.** The four test-time-safe disagreement features
correlate with the best-worst gap. The strongest is
`prediction_jaccard_k1_within` at Spearman |ρ|=0.724 (gate threshold
0.25); the others are weaker (`matching_map_disagreement` 0.143,
`candidate_jaccard_mean_k1` 0.103, `prompt_overlap_mean_k1` 0.103). The
instability is not purely random: a within-reference disagreement feature
predicts it strongly, which is the empirical basis for M2.

**Data-leakage audit.** 0 violations across 2,400 reference assignments
over 10 seeds; every (query, reference) pair satisfies the
patient-disjoint and train-only constraints. The holdout hash matches the
pre-registered value.

**Gate.** All 10 conditions pass. The state is
`PHASET_WP1_MONUSAC_HOLDOUT_CONFIRMED`. The phenomenon replicates on an
unobserved, patient-disjoint holdout with the pre-registered magnitude.

## 2. WP2 — Cross-dataset replication

**MoNuSeg — STRONG_SUPPORT (9/9).** 30 queries, 5 seeds. `V_ref_frac` at
k=1 ranges 0.149–0.160 across configs, dropping to 0.070–0.087 at k=6.
The mean within-query reference std at k=1 is 0.039. 80.0% of queries
have best-worst gap ≥ 0.05. Mean consensus (k=6) gains +0.015 Dice over
random single (0.510 versus 0.495), with paired bootstrap CI lower bound
0.007. CFR is zero throughout (MoNuSeg has no catastrophic-failure
episodes under the frozen threshold). `J_pred` rises from 0.733 (k=1) to
0.830 (k=6). The best disagreement feature is again
`prediction_jaccard_k1_within` at Spearman |ρ|=0.827. All five Tier 1
directional conditions and all four Tier 2 strong-support conditions
pass.

**IDRiD — FAIL (4/5 directional, 2/4 strong).** 30 queries, 5 seeds.
The instability phenomenon is present at the P20-N1 config
(`V_ref_frac` 0.684 at k=1; mean reference std 0.164; 63.3% of queries
with best-worst gap ≥ 0.05), but mean consensus *hurts* Dice. Consensus
(k=6) Dice is 0.090 versus single (k=1) Dice of 0.102, a gain of −0.012
with paired bootstrap CI lower bound −0.032. d3 fails (consensus below
single), s1 fails (gain below +0.01), and s2 fails (CI lower below 0).
d1, d2, d4, d5 pass, and s3 / s4 pass. CFR is very high (0.71–1.00 across
configs) — references are uniformly poor, and aggregating poor references
does not help. The best disagreement feature is
`prediction_jaccard_k1_within` at Spearman |ρ|=0.952, but the
predictability of the gap does not translate into a consensus benefit.

**Cross-dataset synthesis.** The phenomenon is dataset-dependent. On
MoNuSeg the protocol replicates with strong support and the consensus
benefit is directionally identical to MoNuSAC. On IDRiD the
*instability* is real and even predictable, but the *consensus
mitigation* fails: when references are uniformly poor, mean consensus
reduces Dice. This bounds the claim — we do not assert universality —
and is itself a scientific finding. We do not pool significance across
datasets because patient-cluster structure and reference-pool size
differ.

## 3. WP3 M2 — Dual-model selective prediction (PASS, holdout-confirmed)

**The anti-correlation finding.** The catastrophic-trained model's
instability AUROC on validation is 0.060 (near-random), while the
instability-trained model's AUROC is 0.944. The same four features
cannot simultaneously predict instability and catastrophic failure; the
two axes are anti-correlated. This forces the dual-model design: two
logistic regressions on the same features, trained on different labels.

**Validation gate (5/5 PASS).** Instability-trained M2-2 logistic
regression: instability AUROC 0.944 (query-level 0.939), AUPRC 0.989.
Catastrophic-trained M2-2 logistic regression: catastrophic AUPRC 0.434
against prevalence 0.117; ECE 0.043 (gate threshold 0.10); CFR reduction
39.2% at 80% coverage (CFR all 0.117, CFR@80% 0.071); Dice change +0.021
at 80% coverage (Dice all 0.414, Dice@80% 0.436). All five conditions
pass on validation.

**Holdout confirmation (6/6 PASS).** The frozen dual predictor is run
once on the 40-query holdout (1,200 episodes). Instability AUROC 0.928
(query-level 1.000); catastrophic AUPRC 0.528 against prevalence 0.050;
ECE 0.027; CFR reduction **79.2%** at 80% coverage (CFR all 0.050,
CFR@80% 0.010); Dice change +0.009 at 80% coverage (Dice all 0.476,
Dice@80% 0.485). The holdout CFR reduction is larger than the validation
reduction (79.2% vs 39.2%), and the holdout Dice change is positive
(selective prediction slightly *improves* Dice, because the abstained
episodes are disproportionately catastrophic). All six conditions pass;
the state is `PHASET_M2_SELECTIVE_PREDICTION_PASS`.

**Calibration.** The catastrophic-trained model is well calibrated on
both validation (ECE 0.043) and holdout (ECE 0.027). On the holdout,
most probability mass concentrates in the 0.9–1.0 bin (456/615 episodes
on validation), where accuracy 0.943 closely matches confidence 0.981.
The model is not over-confident on the abstention boundary.

**What M2 does and does not.** M2 does not eliminate reference
instability; it abstains on episodes likely to end in catastrophic
failure. It does not retrain the matcher or segmenter. It uses four
test-time-safe features and a logistic regression; no query ground truth
is needed at deployment. The dual-model design is structural: the
anti-correlation makes a single-model design infeasible on this feature
space.

**Cross-dataset transfer (supplementary, not a gate condition).** We
applied the frozen dual-model M2 (trained on MoNuSAC train, selected on
MoNuSAC val) to MoNuSeg and IDRiD episode data without re-tuning. The
ranking ability transfers well: instability AUROC is 0.874 on MoNuSeg
and 0.994 on IDRiD (both above the 0.70 gate threshold). However,
calibration does NOT transfer to IDRiD: ECE jumps from 0.027 (MoNuSAC
holdout) to 0.477 (IDRiD), and CFR reduction drops from 79.2% to 6.0%.
The root cause is distribution shift: IDRiD's catastrophic prevalence is
0.807 versus MoNuSAC train's 0.117, so the frozen threshold
misclassifies most IDRiD episodes. On MoNuSeg, there are zero
catastrophic failures (prevalence 0), so AUPRC and CFR reduction are
vacuously zero — there is nothing to predict or reduce, but the
instability ranking still transfers (AUROC 0.874). The finding is that
M2's *ranking* generalizes cross-dataset, but its *calibration and
threshold* do not, suggesting that target-domain recalibration on a
small labeled set would be needed to deploy M2 on a new dataset.

## 4. WP3 M1 — Dynamic reference budget (NO_GAIN)

**Policy collapse.** Cross-validation selects `λ_c=λ_r=0` for *all* five
policy variants — the cost-quality optimum is "maximize Dice", which
means "always k=6". The best policy is M1-3 (handcrafted threshold): val
Dice 0.4571, avg_k 5.63, CVaR 0.0537, CFR 0.1203. M1-2 (always k=6) has
val Dice 0.4586, CVaR 0.0490, CFR 0.1252 — it Pareto-dominates the
learned policies.

**Cost-saving mode (vs k=6).** cs1 passes (Dice diff −0.0015, within
the 0.003 tolerance); cs2 passes (CVaR diff +0.0047); cs3 passes (CFR
diff −0.0049); cs4 **fails** (avg_k 5.63, only 6.1% reduction versus the
25% gate). The policy cannot meaningfully reduce references without
hurting Dice.

**Quality-cost mode (vs k=3).** qc1 fails (avg_k 5.63 is outside
[2.5, 3.5], even though Dice beats k=3 by +0.014); qc2 fails (CVaR diff
−0.0006 vs k=3); qc3 fails (CFR diff +0.0049 vs k=3). Quality-cost mode
fails on all three conditions.

**Decision and interpretation.** The state is
`PHASET_M1_DYNAMIC_BUDGET_NO_GAIN`. The scientific finding is that
dynamic budgeting cannot meaningfully reduce references without hurting
Dice under this pipeline. This is a failure-boundary contribution: it
bounds the MAUP-style adaptive-quantity paradigm on the frozen
DINOv2-L + SAM-B pipeline and explains why M2 (abstention) succeeds
where M1 (budgeting) fails — abstention acts on the *risk* axis, while
budgeting acts on the *cost* axis, and the cost axis is uninformative
once `λ=0` is optimal.

## 5. Summary of outcomes

| Work package | Outcome | Gate |
|---|---|---|
| WP1 MoNuSAC holdout | CONFIRMED | 10/10 PASS |
| WP2 MoNuSeg | STRONG_SUPPORT | 9/9 PASS |
| WP2 IDRiD | FAIL | 4/5 directional, 2/4 strong |
| WP3 M1 dynamic budget | NO_GAIN | cost-saving FAIL, quality-cost FAIL |
| WP3 M2 selective prediction | PASS | 5/5 validation, 6/6 holdout |

The Phase T program produced one positive mitigation (M2, holdout-
confirmed), one documented negative (M1), one cross-dataset strong
replication (MoNuSeg), and one cross-dataset failure (IDRiD). The paper
is therefore a *benchmark + failure-boundary + one positive mitigation*
paper. The M2 holdout confirmation (CFR reduction 79.2%, Dice change
+0.009, ECE 0.027) is the central positive result; the M1 negative and
the IDRiD failure are reported as failure boundaries that bound the
claim and delineate where reference-conditioned mitigation works and
where it does not. No prohibited claim is made: no "SOTA", no
"instability solved", no "universal across datasets" (explicitly
contradicted by IDRiD), no "mean Dice alone proves robustness" (every
robustness statement cites variance or tail risk).
