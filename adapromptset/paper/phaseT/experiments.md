# Experiments

The experiments are organized into three work packages that are executed
sequentially under a strict pre-registration discipline. WP1 confirms the
core phenomenon on the unobserved MoNuSAC holdout. WP2 replicates the
protocol on MoNuSeg and IDRiD to bound the claim cross-dataset. WP3
studies two risk-aware mitigations — M1 dynamic reference budget and M2
dual-model selective prediction — on validation, then confirms the
passing method on the holdout exactly once. Each work package has its own
gate; a failure is a scientific finding, not a bug, and is reported as
such.

## WP1 — MoNuSAC confirmatory holdout

**Goal.** Confirm the reference-induced instability phenomenon on an
unobserved, patient-disjoint MoNuSAC holdout under the frozen protocol.

**Setup.** 40 holdout queries (8 patients), 10 resampling seeds, three
frozen prompt configs (P20-N1, P30-N0, P50-N0), k∈{1,3,6}. The k-scaling
design yields 6 single-reference subsets, 1 three-reference mean-consensus
subset, and 1 six-reference mean-consensus subset per (query, seed,
config), for 9,600 SAM calls in total. The reference pool is the 80-case /
18-patient train-only pool, patient-disjoint from the holdout queries; the
reference-query pair constraint is audited on every (query, reference)
pair across all 10 seeds (2,400 reference assignments in total).

**Method families.** The internal B1–B7 reference-aggregation families
are evaluated. B1 is the random single-reference baseline (k=1); B2_mean
is the mean-consensus baseline (k=3, k=6). The primary comparison for the
gate is B2_mean (k=6) versus B1 (k=1).

**Metrics.** `V_query`, `V_ref`, `V_ref_frac`, mean within-query
reference std, best-worst gap (mean and fraction ≥ 0.05), `MAD_ref`,
`J_pred`, mean Dice, `CVaR_0.1`, `CVaR_0.2`, CFR (Dice < 0.1),
worst-subset Dice. Paired bootstrap (10,000 iterations,
patient-cluster resampling, 95% CI from the 2.5th / 97.5th percentiles)
for the B2-vs-B1 Dice comparison. Spearman correlation between each
disagreement feature and the best-worst gap.

**Gate (10 conditions, all must pass).** (1) ≥50% holdout queries have
best-worst gap ≥ 0.05. (2) Mean within-query reference std ≥ 0.02. (3)
`V_ref_frac` ≥ 0.10. (4) Mean consensus (B2) Dice − random single (B1)
Dice ≥ 0.010. (5) Paired bootstrap 95% CI lower bound > 0. (6)
Catastrophic-failure rate of B2 not increased vs B1. (7) `CVaR_0.1` of
B2 not decreased vs B1. (8) `J_pred` of B2 improved vs B1. (9)
Disagreement-feature vs instability: best Spearman |ρ| ≥ 0.25. (10) No
data leakage (audit PASS).

**Compute.** 9,600 SAM calls, 1,526.5 seconds wall-clock on CPU; 0
GPU-hours against the 195 GPU-hour program cap.

## WP2 — Cross-dataset replication

**Goal.** Bound the claim cross-dataset by replicating the protocol on
MoNuSeg and IDRiD under a two-tier gate.

**Setup.** MoNuSeg: 30 strict-validation queries, 5 resampling seeds,
three prompt configs, k∈{1,3,6}. IDRiD: 30 validation queries under one
frozen lesion intent, 5 resampling seeds, three prompt configs, k∈{1,3,6}.
Both datasets use their own train-only reference pools (37 cases for
MoNuSeg, 383 for IDRiD) and enforce patient/case-disjoint reference-query
pairs.

**Two-tier gate.** Tier 1 (directional support, all five must pass): d1
≥30% queries with best-worst gap ≥ 0.05; d2 mean within-query reference
std ≥ 0.015; d3 mean consensus Dice ≥ random single Dice (not below); d4
CFR not increased under consensus; d5 `J_pred` not decreased under
consensus. Tier 2 (strong support, four additional): s1 consensus gain ≥
0.01; s2 paired bootstrap CI lower > 0; s3 `CVaR_0.1` improved; s4 best
disagreement-feature Spearman |ρ| ≥ 0.20. A dataset with Tier 1 pass and
Tier 2 pass is `STRONG_SUPPORT`; Tier 1 pass with Tier 2 fail is
`DIRECTIONAL_SUPPORT`; any Tier 1 failure is `FAIL`.

**Pooled-significance policy.** No pooled significance is computed across
datasets, because patient-cluster structure and reference-pool size differ.
Cross-dataset synthesis is qualitative: we report per-dataset gate
outcomes and the directional pattern of `V_ref_frac`, consensus gain, and
Spearman correlation.

## WP3 — Risk-aware mitigation

**Goal.** Study M1 dynamic reference budget and M2 dual-model selective
prediction on validation, then confirm the passing method on the holdout
exactly once.

**Data splits.** Train: 80 queries / 1,200 episodes (model fitting,
threshold selection, M1 utility labels, M2 label construction). Validation:
41 queries / 615 episodes (model selection, gate evaluation). Holdout: 40
queries / 1,200 episodes (one-shot confirmation of the frozen method).
Splits are patient-disjoint and reused from prior phase manifests; the
holdout is the same 40 queries used by WP1, observed a second time only
for the pre-registered final-method confirmation.

**M1 setup.** Six policy variants (always-k1, always-k3, always-k6,
handcrafted threshold, logistic regression, gradient boosting) optimize
`U(q,k) = (1-λ_c-λ_r)·Dice - λ_c·Cost - λ_r·Risk` over k∈{1,3,6}.
λ_c and λ_r are selected by cross-validation on train. The gate has two
modes: cost-saving (vs k=6, requires ≥25% reference reduction without
losing Dice / CVaR / CFR) and quality-cost (vs k=3, requires Dice gain
with avg_k in [2.5, 3.5]). Pass if either mode passes.

**M2 setup.** Five model variants (entropy threshold, disagreement
threshold, logistic regression, gradient boosting, small MLP) operate on
four test-time-safe disagreement features. Two labels are constructed on
train only: the instability label (best-worst gap exceeds the train-set
90th percentile) and the catastrophic label (Dice < 0.1). The
instability-trained model drives cond1 (instability AUROC); the
catastrophic-trained model drives cond2–cond5 (selective prediction).
Cross-validation on train selects the model class; validation runs the
5-condition gate; holdout runs the 6-condition gate (the sixth being
directional consistency).

**Prevalence.** Train/val prevalence of the instability label is 0.847;
of the catastrophic label, 0.117. Holdout prevalence of the
catastrophic label is 0.050. The catastrophic-trained model is evaluated
against prevalence on each split, not against a fixed threshold.

**Evaluation protocol.** For M2, the risk-coverage curve is computed at
10 coverage levels from 0.1 to 1.0; the gate uses 80% coverage. ECE is
computed over ten equal-width probability bins; calibration is reported
both raw and post-hoc (the post-hoc number is used only if the raw ECE
exceeds 0.10). The holdout run observes the frozen dual predictor once,
on the same 40 queries as WP1; no parameter is retuned.

## Training details

All learned components (M1 regressors, M2 logistic regression / gradient
boosting / MLP) are fit on the 80-query / 1,200-episode train split. M2
logistic regression uses L2 regularization with the strength selected by
5-fold cross-validation on train; the gradient-boosting variant uses 100
trees with depth 3 and a 0.1 learning rate; the small MLP has one hidden
layer of 16 units. M1 gradient boosting uses the same hyperparameters.
No model selection is performed on validation or holdout; validation is
used only for the pre-registered gate, and holdout for one-shot
confirmation. Random seeds for model fitting are recorded in the run
metadata. All metric implementations are unit-tested against analytic
references; the metric suite is the same one audited in prior phases.

## Compute and runtime

Phase T used 0 GPU-hours on WP1 (CPU inference) and a small CPU budget on
WP2 and WP3 (model fitting is lightweight). The 195 GPU-hour program cap
is not binding; the remaining budget is reserved for any authorized
external-baseline reproduction. Per-run metadata (environment, git commit,
config, input hashes, runtime, SAM calls, matching passes) is recorded in
`reports/phaseT_*_run_metadata.json` for every WP.

## External baselines

External comparator methods (GF-SAM, SynPo, ProSAM, MAUP, SAMIX, AlignSAM,
PPD) are classified into tiers (Tier 1 directly reusable; Tier 2 minimal
adaptation; Tier 3 conceptual only; Tier 4 not applicable). GF-SAM and
MAUP are identified as Tier 2 reproducible. Numerical reproduction
requires user authorization for external code download and is therefore
reported as best-effort and pending; no Tier 3 numerical results are
fabricated, and no external method is claimed to be "fully outperformed".
The internal B1–B7 families cover at least three distinct aggregation
families, satisfying the strong-internal-baselines provision of the
external-comparator minimum.
