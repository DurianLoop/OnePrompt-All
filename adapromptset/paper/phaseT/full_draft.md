# Reference-Induced Instability in Visual-Reference Segmentation: A Cross-Reference Benchmark and Failure-Boundary Analysis

*Phase T paper draft. Candidate title (one of three from directive §19);
see `title_candidates.md` for the full candidate list and selection
rationale. This draft is a paper candidate, not a completed publication.
The formal test set is locked and unused; no SOTA, "instability solved",
or "universal across datasets" claim is made.*

---

## Abstract

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

---

## 1. Introduction

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
measurements on the same query rather than independent samples? And *can*
a simple, risk-aware strategy mitigate the worst outcomes without
retraining the frozen matcher (DINOv2-L) or the frozen segmenter (SAM-B)?

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

**Contributions.** (i) A holdout-confirmed reference-induced instability
benchmark with a corrected variance decomposition and a tail-risk metric
suite. (ii) A cross-dataset failure boundary (MoNuSeg strong support,
IDRiD failure). (iii) A dual-model selective predictor (M2) that is
holdout-confirmed and deploys four test-time-safe features. (iv) A
documented negative result for dynamic reference budgeting (M1). (v) A
reproducible, pre-registered benchmark protocol, metric suite, and gate
structure.

---

## 2. Related Work

We organize prior work into five clusters and state, for each cluster, what
the cluster does, what we do differently, and the Phase T evidence that
supports the differentiation. We do not rank external methods as "fully
outperformed" (a prohibited claim); we report head-to-head metrics under a
matched protocol where reproduction is possible, and document mechanisms
where it is not.

**Prompt-robustness methods (GF-SAM, SynPo, AlignSAM, PPD).** GF-SAM builds
a graph over candidate references or prompts and uses graph grouping to
select a coherent reference set; its object is *selection* ("which
references are best?"). Our object is *distribution*: we do not select
references — every query receives the full per-query pool of six
references, resampled across ten seeds — and we measure the distribution of
predictions induced by resampling. The distinction matters empirically:
on the WP1 holdout the reference fraction of total variance is 0.061–0.263,
and M1 shows that an optimized dynamic budget over references collapses to
"always k=6", i.e. there is no selection gain to be had beyond using the
full pool. SynPo synthesizes positive and negative point prompts to
augment the prompt space; we use only *real* references and a frozen
candidate pool, and our negative results concern natural reference-induced
instability, not adversarial or synthetic perturbations. The WP2 IDRiD
failure (consensus gain −0.012) is attributable to the dataset and
reference pool, not to prompt construction, since prompt construction is
identical to MoNuSeg where consensus helps (+0.015). AlignSAM uses
reinforcement learning to iteratively refine prompts; we do no RL, no
iterative refinement, and no SAM fine-tuning. M2's holdout confirmation
(6/6, CFR reduction 79.2%) is achieved with a static logistic regression,
with no policy gradient or environment interaction. Point Prompt Defender
filters *adversarial* point prompts; we study *natural* reference-induced
instability where every point prompt is a legitimate DINOv2-matched
candidate, and M1 "filtering" over reference subsets fails (NO_GAIN) — a
negative result that bounds the filtering paradigm for natural instability.

**Reference-aware methods (ProSAM, MAUP, SAMIX).** ProSAM models prompt
robustness probabilistically, learning prompt distributions that are stable
under perturbation. We do not learn prompts; we treat the prompt
distribution induced by reference resampling as the *object of
measurement* and predict failure risk from test-time-safe features. Our
probability model is over *failure*, not over *prompts*. M2's
catastrophic-trained model produces a calibrated probability of
catastrophic failure per episode (AUPRC 0.434 on validation against
prevalence 0.117; 0.528 on holdout against prevalence 0.050; ECE 0.043 /
0.027). MAUP uses multi-scale anchors and uncertainty to decide *how many*
prompts to use; we share the question "how many references?" (M1) but
report a *negative result*: the cost-quality optimum is `lambda=0`, i.e.
always k=6. All five M1 policy variants select `lambda_c=lambda_r=0`;
`M1-2_always_k6` Pareto-dominates the learned policies on validation. The
cost-quality trade-off is unfavorable under our pipeline, the opposite of
MAUP's positive prompt-quantity result. SAMIX selects or mixes references
to produce a stronger composite prompt; we do not select or mix references
to build a stronger prompt, and we show that on IDRiD mixing/consensus
*hurts* Dice (−0.012), contradicting the SAMIX premise that reference
aggregation is beneficial.

**Few-shot segmentation.** Few-shot segmentation (PANet, PPNet, and
related) trains a segmenter with a support set of labeled images and
generalizes to a query; the support set is a *training* signal. We do not
train a segmenter. SAM-B is frozen; the references are used only to
condition point prompts via DINOv2 matching. Our "support set" is a
prompt-conditioning set, not a learning set, and we benchmark the
*instability* of this paradigm. WP1 (10/10) and WP2 MoNuSeg (9/9) replicate
the instability phenomenon with zero segmenter training, isolating
reference-induced variance from any learning-induced variance.

**Selective prediction and abstention.** M2 overlaps with this cluster, and
the overlap is the largest among the ten novelty-audit items. The
differentiation is fourfold. First, the *object of abstention* is a
reference-conditioned episode, not an input in general; the risk we model
is reference-induced catastrophic failure, not generic uncertainty. Second,
the *dual-model design is forced by an empirically measured
anti-correlation* between instability (high best-worst gap) and
catastrophic failure (low Dice): a single predictor cannot both flag
instability and flag catastrophic risk, so we train two logistic
regressions on the same four features with different labels. The
catastrophic-trained model's instability AUROC is 0.060 (near-random)
while the instability-trained model's AUROC is 0.944 — an empirical
demonstration that the two axes are not the same. Third, the predictor
uses *test-time-safe features only*, enabling deployment. Fourth, the
result is *pre-registered holdout confirmation* (6/6, CFR reduction
79.2%), not a validation-split claim.

**Set aggregation and consensus.** Set aggregation (DeepSets, Set
Transformer, mean/median/trimmed ensembles) combines a variable-size set
into a fixed representation. Prior work in this pipeline already tested
six deterministic aggregators (B1–B6) plus learned weighting and
set-conditioned gates, all failing to beat B2_mean. Phase T's contribution
here is *not* a new aggregator; it is (i) holdout confirmation that
B2_mean remains the strongest automatic baseline (WP1 cond4/cond5:
consensus gain +0.027, CI lower 0.014), and (ii) the M1 result that
*dynamic* set sizing also fails. The aggregation question is closed
negatively under this pipeline.

**Summary.** Phase T does not select references (vs. GF-SAM, SAMIX), does
not synthesize prompts (vs. SynPo), does not refine prompts iteratively
(vs. AlignSAM), does not defend against adversarial points (vs. PPD), does
not learn a prompt distribution (vs. ProSAM), does not train a segmenter
(vs. few-shot segmentation), and does not propose a new set aggregator.
What it does is *measure* the prediction distribution, tail risk, and
failure probability under reference resampling, *bound* the phenomenon
cross-dataset, and *mitigate* its worst outcomes with a dual-model
selective predictor that is holdout-confirmed and a dynamic-budget method
that is a documented negative.

---

## 3. Benchmark Protocol

**Datasets and roles.** MoNuSAC (primary): 80 train / 41 validation / 40
holdout / 101 formal test (locked). Reference pool: 80 cases / 18 patients,
train-only, patient-disjoint from queries. MoNuSeg (secondary): 37 train
cases serve as the reference pool; 30 strict-validation queries are used
for WP2; 14 test cases locked. IDRiD (tertiary): 383 train cases serve as
the reference pool for one frozen lesion intent; 30 validation queries are
used for WP2; 225 test cases locked. No patient/case leakage is permitted
between splits within any dataset.

**Reference construction.** Each query receives six references drawn from
the train-only pool, patient-disjoint from the query. k∈{1,3,6} subset
sizes: k=1 uses the six individual references; k=3 uses mean consensus
over the first three; k=6 uses mean consensus over all six. All methods
evaluated on the same query share the exact same six-reference pool; methods
differ only in aggregation or selection. Ten resampling seeds on MoNuSAC
train; five on MoNuSeg and IDRiD. Holdout and validation reference
allocations are fixed at pre-registration and hash-verified.

**Frozen pipeline.** Matcher: DINOv2-L features with z-score
foreground/background normalization, reused from the per-case feature
cache. Candidate selection: P_TOPK positive, N_BOTTOMK or N_NONE negative.
Three prompt configs: P20-N1 (K+=20, K-=1, N_BOTTOMK), P30-N0 (K+=30,
K-=0, N_NONE), P50-N0 (K+=50, K-=0, N_NONE). Segmenter: SAM-B, single call
per consensus aggregation, multimask output rule frozen. B2_mean fuses
reference-conditioned prompts before the single SAM call, making consensus
a single-SAM-call baseline that controls cost.

**Metrics.** Mean performance: Dice, IoU, AJI, component recall,
precision/recall. Reference-instability: `V_query`, `V_ref`, `V_ref_frac`,
`MAD_ref`, `J_pred`. Tail risk: `CVaR_0.1`, `CVaR_0.2`, CFR (Dice < 0.1),
worst-subset Dice, best-worst gap. Cost: reference count, DINOv2 matching
passes (cached / uncached), SAM calls, runtime, GPU-hours. The statistical
unit is the query/case; references are repeated measurements and are not
treated as independent samples. Paired bootstrap CIs use B=10,000
iterations and patient-cluster resampling for MoNuSAC; Holm's step-down
correction at familywise α=0.05 for multiple comparisons. Reporting mean
Dice without variance or tail-risk evidence is a prohibited over-claim.

**Holdout design.** The MoNuSAC holdout may be observed exactly twice:
(1) WP1 confirmatory benchmark; (2) WP3 final-method one-shot
confirmation. Each run is independent and preceded by hash verification of
the frozen manifests and configs. The WP1 run produces 9,600 SAM calls. No
parameter on the freeze list may be retuned between the two runs based on
holdout observations; the catastrophic threshold (0.1), bootstrap iteration
count (10,000), and qualitative-sampling rules (worst-5, best-5, median-5
by `V_ref`) are all frozen.

---

## 4. Method

The method has four parts: the reference-resampling protocol (Section 3),
the corrected variance decomposition and tail-risk metric suite, the M2
dual-model selective predictor, and the M1 dynamic-budget policy (reported
as a negative result).

**Corrected variance decomposition.** We decompose total variance into a
query-attributable component and a reference-attributable component:
`V_ref = (1/N) Σ_q Var_S[D(q,S)]`, `V_query = (1/N) Σ_q Var_S[D(q,·)]`,
`V_ref_frac = V_ref / (V_query + V_ref)`, where S indexes reference
subsets and D(q,S) is the Dice of query q under reference subset S.
`V_ref_frac` is the fraction of total variance attributable to reference
choice. `MAD_ref` is the mean absolute pairwise reference deviation. `J_pred`
is the prediction-consistency Jaccard: `J_pred = E_{S_i,S_j}
IoU(M̂(q,S_i),M̂(q,S_j))`. Tail-risk metrics: `CVaR_0.1` is the mean Dice
of the worst 10% of episodes; CFR is the fraction of episodes with Dice <
0.1 (frozen threshold); the worst-subset Dice is the minimum per-query Dice
across references, averaged over queries; the best-worst gap is the
per-query difference between best and worst reference-subset Dice.

**Disagreement features.** Four test-time-safe features (computable
without query ground truth) are computed per episode:
`matching_map_disagreement`, `candidate_jaccard_mean_k1`,
`prompt_overlap_mean_k1`, and `prediction_jaccard_k1_within`. The last is
the within-query version of `J_pred` at k=1 and is the strongest
instability predictor on the WP1 holdout (Spearman |ρ|=0.724).

**M2 dual-model selective prediction.** The goal is a deployable selective
predictor that flags reference-conditioned episodes likely to end in
catastrophic failure. The design is forced by an empirical
anti-correlation: instability (high best-worst gap) and catastrophic
failure (low Dice) are anti-correlated on the same feature space. A
predictor trained on the instability label achieves AUROC 0.944 on
validation but a catastrophic AUPRC of only 0.071 (below prevalence
0.117); a predictor trained on the catastrophic-failure label achieves
AUPRC 0.434 (above prevalence) but an instability AUROC of 0.060
(near-random). The two axes are not the same, and a single model cannot
satisfy both. This forces the dual-model design: two logistic regressions
on the same four test-time-safe features, trained with different labels.
The instability label is the binary indicator that the episode's
best-worst gap exceeds the train-set 90th percentile; the catastrophic
label is the binary indicator that the episode's Dice is below 0.1. Both
labels are constructed on train only. Five model variants are evaluated
(M2-0 entropy, M2-1 disagreement, M2-2 logistic regression, M2-3 gradient
boosting, M2-4 small MLP); the frozen dual predictor selects M2-2 logistic
regression on both axes for calibration. Gate conditions (5 on validation,
6 on holdout): cond1 instability AUROC ≥ 0.70; cond2 catastrophic AUPRC >
prevalence; cond3 ECE ≤ 0.10; cond4 CFR reduction ≥ 30% at 80% coverage;
cond5 mean Dice not decrease at 80% coverage; cond6 (holdout) cond1–5 all
pass on holdout. At deployment the catastrophic-trained model produces a
calibrated probability of catastrophic failure per episode; the system
abstains when the probability exceeds a coverage-derived threshold.

**M1 dynamic reference budget (negative result).** Six policy variants
(always-k1, always-k3, always-k6, handcrafted threshold, logistic
regression, gradient boosting) optimize `U(q,k) = (1-λ_c-λ_r)·Dice -
λ_c·Cost - λ_r·Risk` over k∈{1,3,6}. λ_c and λ_r are selected by
cross-validation on train. Cost-saving mode (vs k=6) requires ≥25%
reference reduction without losing Dice / CVaR / CFR; quality-cost mode
(vs k=3) requires Dice gain with avg_k in [2.5, 3.5]. Pass if either mode
passes.

**Pre-registration.** All of the above is frozen at pre-registration. The
freeze list includes the reference pool, k∈{1,3,6}, the resampling seeds,
the three prompt configs, matching normalization, candidate selection, the
SAM config, all metric definitions, the catastrophic threshold (0.1), the
bootstrap procedure (10,000 iterations, patient-cluster for MoNuSAC), and
the qualitative-sampling rules. The claim-boundary audit scans every
holdout report for prohibited tokens before any state transition.

---

## 5. Experiments

The experiments are organized into three sequential work packages under
strict pre-registration. WP1 confirms the core phenomenon on the unobserved
MoNuSAC holdout. WP2 replicates the protocol on MoNuSeg and IDRiD. WP3
studies M1 and M2 on validation, then confirms the passing method on the
holdout exactly once.

**WP1 — MoNuSAC confirmatory holdout.** 40 holdout queries (8 patients), 10
resampling seeds, three frozen prompt configs, k∈{1,3,6}, yielding 9,600
SAM calls. The reference pool is the 80-case / 18-patient train-only pool.
Method families: B1 random single (k=1), B2_mean (k=3, k=6), and the rest
of the B1–B7 internal families. The primary gate comparison is B2_mean
(k=6) vs B1 (k=1). Gate: 10 conditions, all must pass (best-worst gap,
within-query reference std, `V_ref_frac`, consensus Dice gain, paired
bootstrap CI lower, CFR, `CVaR_0.1`, `J_pred`, disagreement-feature
Spearman, no data leakage). Compute: 9,600 SAM calls, 1,526.5 s wall-clock
on CPU, 0 GPU-hours.

**WP2 — Cross-dataset replication.** MoNuSeg: 30 strict-validation
queries, 5 seeds, three configs, k∈{1,3,6}. IDRiD: 30 validation queries
under one frozen lesion intent, 5 seeds, three configs, k∈{1,3,6}. Two-tier
gate: Tier 1 directional (d1 best-worst gap fraction, d2 reference std, d3
consensus Dice ≥ single, d4 CFR not increased, d5 `J_pred` not decreased);
Tier 2 strong (s1 gain ≥ 0.01, s2 CI lower > 0, s3 `CVaR_0.1` improved, s4
Spearman ≥ 0.20). No pooled significance across datasets.

**WP3 — Risk-aware mitigation.** Train: 80 queries / 1,200 episodes.
Validation: 41 queries / 615 episodes. Holdout: 40 queries / 1,200
episodes (one-shot confirmation). M1: six policy variants, λ_c, λ_r
selected by CV on train; gate has cost-saving and quality-cost modes. M2:
five model variants on four test-time-safe features, two labels
constructed on train only; validation runs the 5-condition gate, holdout
runs the 6-condition gate. Train/val prevalence of the catastrophic label:
0.117; holdout prevalence: 0.050. The risk-coverage curve is computed at
10 coverage levels; the gate uses 80% coverage.

**Training details.** M2 logistic regression: L2 regularization, strength
selected by 5-fold CV on train. M2 gradient boosting: 100 trees, depth 3,
0.1 learning rate. M2 small MLP: one hidden layer of 16 units. M1 gradient
boosting: same hyperparameters. No model selection on validation or
holdout; validation is used only for the pre-registered gate. Random seeds
for model fitting are recorded in run metadata. All metric implementations
are unit-tested against analytic references.

**External baselines.** GF-SAM, SynPo, ProSAM, MAUP, SAMIX, AlignSAM, PPD
are classified by tier. GF-SAM and MAUP are identified as Tier 2
reproducible; numerical reproduction requires user authorization for
external code download and is therefore pending. No Tier 3 numerical
results fabricated; no external method claimed "fully outperformed". The
internal B1–B7 families cover at least three distinct aggregation
families, satisfying the strong-internal-baselines provision.

---

## 6. Results

### 6.1 WP1 — MoNuSAC holdout (10/10 PASS)

**Within-query reference variance.** `V_ref_frac` ranges 0.061–0.263
across the three prompt configs and k∈{1,3,6}. At k=1, `V_ref_frac` is
0.263 (P20-N1), 0.211 (P30-N0), 0.126 (P50-N0); at k=6 it drops to 0.189,
0.159, 0.061. The mean within-query reference std at k=1 is 0.074 (gate
threshold 0.02). Reference-induced variance is a substantial fraction of
total variance and shrinks — but does not vanish — under consensus.

**Best-worst gap.** 92.5% of holdout queries (37/40) have a mean
best-worst gap ≥ 0.05 (gate threshold 50%). The mean best-worst gap at
k=1 is 0.239 (P20-N1), 0.214 (P30-N0), 0.167 (P50-N0).

**Mean consensus vs random single.** Mean consensus over six references
gains +0.027 Dice over random single (0.506 vs 0.479 averaged across
configs). Paired bootstrap 95% CI lower bound 0.014 (n=40, 10,000
iterations, patient-cluster resampling). Consensus improves the tail:
`CVaR_0.1` rises 0.075 → 0.092; CFR drops 0.063 → 0.058; `J_pred` rises
0.649 → 0.719. CFR at k=6 ranges 0.050–0.073 across configs.

**Predictability.** `prediction_jaccard_k1_within` correlates with the
best-worst gap at Spearman |ρ|=0.724 (gate threshold 0.25); the other
three features are weaker (0.143, 0.103, 0.103). The instability is not
purely random.

**Data-leakage audit.** 0 violations across 2,400 reference assignments
over 10 seeds. The holdout hash matches the pre-registered value. All 10
gate conditions pass; the state is
`PHASET_WP1_MONUSAC_HOLDOUT_CONFIRMED`.

### 6.2 WP2 — Cross-dataset replication

**MoNuSeg — STRONG_SUPPORT (9/9).** `V_ref_frac` at k=1 ranges
0.149–0.160, dropping to 0.070–0.087 at k=6. Mean within-query reference
std at k=1 is 0.039. 80.0% of queries have best-worst gap ≥ 0.05. Consensus
(k=6) gains +0.015 Dice over single (0.510 vs 0.495), CI lower 0.007. CFR
is zero throughout. `J_pred` rises 0.733 → 0.830. Best Spearman 0.827.
All five Tier 1 and all four Tier 2 conditions pass.

**IDRiD — FAIL (4/5 directional, 2/4 strong).** The instability is present
at P20-N1 (`V_ref_frac` 0.684 at k=1; mean reference std 0.164; 63.3% of
queries with best-worst gap ≥ 0.05), but mean consensus *hurts* Dice.
Consensus (k=6) Dice 0.090 vs single (k=1) 0.102, gain −0.012, CI lower
−0.032. d3, s1, s2 fail; d1, d2, d4, d5, s3, s4 pass. CFR is 0.71–1.00
across configs — references are uniformly poor. Best Spearman 0.952, but
predictability does not translate into a consensus benefit.

**Cross-dataset synthesis.** The phenomenon is dataset-dependent. On
MoNuSeg the protocol replicates with strong support; on IDRiD the
instability is real and predictable, but consensus *hurts* Dice because
references are uniformly poor. This bounds the claim — we do not assert
universality — and is itself a scientific finding.

### 6.3 WP3 M2 — Dual-model selective prediction (PASS, holdout-confirmed)

**Anti-correlation.** The catastrophic-trained model's instability AUROC
on validation is 0.060 (near-random), while the instability-trained
model's AUROC is 0.944. The same four features cannot simultaneously
predict both axes; the dual-model design is forced.

**Validation gate (5/5 PASS).** Instability-trained M2-2 logistic
regression: AUROC 0.944 (query-level 0.939), AUPRC 0.989.
Catastrophic-trained M2-2: AUPRC 0.434 (prevalence 0.117); ECE 0.043;
CFR reduction 39.2% at 80% coverage (CFR all 0.117, CFR@80% 0.071); Dice
change +0.021 (Dice all 0.414, Dice@80% 0.436). All five conditions pass.

**Holdout confirmation (6/6 PASS).** Frozen dual predictor run once on
the 40-query holdout (1,200 episodes). Instability AUROC 0.928
(query-level 1.000); catastrophic AUPRC 0.528 (prevalence 0.050); ECE
0.027; CFR reduction **79.2%** at 80% coverage (CFR all 0.050, CFR@80%
0.010); Dice change +0.009 (Dice all 0.476, Dice@80% 0.485). Holdout CFR
reduction exceeds validation (79.2% vs 39.2%); holdout Dice change is
positive. All six conditions pass; the state is
`PHASET_M2_SELECTIVE_PREDICTION_PASS`.

**Calibration.** ECE 0.043 (val) / 0.027 (holdout). On the holdout most
probability mass concentrates in the 0.9–1.0 bin, where accuracy 0.943
closely matches confidence 0.981. The model is not over-confident on the
abstention boundary.

**Cross-dataset transfer (supplementary).** The frozen M2 is applied to
MoNuSeg and IDRiD without re-tuning. Ranking transfers well: instability
AUROC 0.874 (MoNuSeg) and 0.994 (IDRiD), both above the 0.70 gate
threshold. Calibration does NOT transfer to IDRiD: ECE jumps to 0.477
and CFR reduction drops to 6.0%, because IDRiD's catastrophic prevalence
(0.807) differs sharply from MoNuSAC train's (0.117). On MoNuSeg there
are zero catastrophic failures (prevalence 0), so AUPRC and CFR
reduction are vacuously zero. The finding: M2's *ranking* generalizes
cross-dataset, but its *calibration and threshold* do not, suggesting
that target-domain recalibration on a small labeled set would be needed
to deploy M2 on a new dataset.

### 6.4 WP3 M1 — Dynamic reference budget (NO_GAIN)

**Policy collapse.** CV selects `λ_c=λ_r=0` for all five policy variants —
the optimum is "maximize Dice", i.e. "always k=6". Best policy M1-3
(handcrafted threshold): val Dice 0.4571, avg_k 5.63, CVaR 0.0537, CFR
0.1203. M1-2 (always k=6) Pareto-dominates: Dice 0.4586, CVaR 0.0490, CFR
0.1252.

**Cost-saving mode (vs k=6).** cs1 passes (Dice diff −0.0015); cs2 passes
(CVaR diff +0.0047); cs3 passes (CFR diff −0.0049); cs4 **fails** (avg_k
5.63, 6.1% reduction vs 25% gate).

**Quality-cost mode (vs k=3).** qc1 fails (avg_k 5.63 outside [2.5, 3.5]);
qc2 fails (CVaR diff −0.0006); qc3 fails (CFR diff +0.0049). The state is
`PHASET_M1_DYNAMIC_BUDGET_NO_GAIN`. The finding: dynamic budgeting cannot
meaningfully reduce references without hurting Dice under this pipeline.
M2 (abstention) succeeds where M1 (budgeting) fails because abstention
acts on the risk axis while budgeting acts on the cost axis, and the cost
axis is uninformative once `λ=0` is optimal.

### 6.5 Summary

| Work package | Outcome | Gate |
|---|---|---|
| WP1 MoNuSAC holdout | CONFIRMED | 10/10 PASS |
| WP2 MoNuSeg | STRONG_SUPPORT | 9/9 PASS |
| WP2 IDRiD | FAIL | 4/5 directional, 2/4 strong |
| WP3 M1 dynamic budget | NO_GAIN | cost-saving FAIL, quality-cost FAIL |
| WP3 M2 selective prediction | PASS | 5/5 validation, 6/6 holdout |

---

## 7. Limitations

**Single primary dataset.** The holdout-confirmed benchmark is on MoNuSAC
alone (40 queries, 8 patients). Cross-dataset replications (MoNuSeg 30,
IDRiD 30) are smaller and use five seeds rather than ten. The
pre-registered gate structure makes small-sample claims falsifiable, but
a larger holdout would tighten the CIs.

**Frozen matcher and segmenter.** All results are under DINOv2-L + SAM-B.
We do not know whether `V_ref`, the anti-correlation, or the M2 gains
transfer to other matchers or segmenters. We make no "model-agnostic"
claim.

**Cross-dataset failure is part of the finding.** IDRiD fails the
consensus-benefit gate. We hypothesize uniformly poor references
(CFR 0.71–1.00) but have not isolated the cause (lesion intent,
reference-pool size, matcher failure on retinal imagery). A follow-up
ablation is needed.

**M2 is calibrated but not risk-free.** CFR at 80% coverage is 0.010, not
zero. The remaining failures are the hardest cases; a deployed system
needs a human-in-the-loop policy for the abstained 20%. M2 is a
risk-aware abstention layer, not a fix.

**M1 negative is bounded to this pipeline.** The NO_GAIN reflects the
cost-quality utility and k∈{1,3,6}. A richer action space or a different
utility (e.g., one pricing risk rather than reference count) might
behave differently.

**External comparators best-effort.** GF-SAM and MAUP identified as Tier 2
reproducible; numerical reproduction pending authorization. No Tier 3
numerical results fabricated; no external method claimed "fully
outperformed".

**No formal test access.** The formal test set (101 MoNuSAC cases) is
locked and unused. All claims are bounded by holdout evidence. No SOTA or
leaderboard claim.

**Medical imaging focus.** All three datasets are medical. Generalization
to natural images not tested. We report where the protocol replicates
and where it does not.

**Single-instability-axis assumption.** M2's dual-model design assumes two
axes suffice. Other axes (boundary-quality, instance-count instability)
may exist and may not be captured by the four features. We claim
sufficiency to pass the pre-registered gate, not completeness.

---

## 8. Conclusion

We asked three questions: how much does a valid reference affect
visual-reference segmentation, how should the resulting instability be
measured, and can a simple risk-aware strategy mitigate it without
retraining the frozen matcher and segmenter?

**On measurement.** Reference-induced instability is real, substantial,
and predictable. On a pre-registered, patient-disjoint MoNuSAC holdout,
`V_ref_frac` is 0.061–0.263, 92.5% of queries show a best-worst gap ≥
0.05, and a within-reference disagreement feature predicts the gap at
Spearman |ρ|=0.724. Mean consensus gains +0.027 Dice (CI lower 0.014)
while reducing CFR and improving `J_pred`. The corrected variance
decomposition, the tail-risk metric suite, and the patient-cluster paired
bootstrap convert a hidden degree of freedom into a measured, falsifiable
phenomenon. Mean Dice alone is not a robustness claim.

**On the boundary.** The phenomenon is dataset-dependent. MoNuSeg shows
strong support (9/9, gain +0.015, CI lower 0.007); IDRiD fails (gain
−0.012, CI lower −0.032) because references are uniformly poor. We do not
claim universality; the failure is part of the contribution.

**On mitigation.** M1 collapses to "always k=6" (`λ_c=λ_r=0` for all
variants) and fails its gate (avg_k 5.63, 6.1% reduction vs 25% target).
M2 passes its pre-registered holdout gate (6/6): instability AUROC 0.928,
catastrophic AUPRC 0.528 (prevalence 0.050), ECE 0.027, CFR reduction
79.2% at 80% coverage, Dice change +0.009. The dual-model design is
structural: the anti-correlation between instability and catastrophic
failure makes a single-model design infeasible. M2 uses four
test-time-safe features, requires no query ground truth at deployment,
and does not retrain the matcher or segmenter.

**What this paper is not.** Not a state-of-the-art segmentation paper.
Matcher and segmenter frozen; formal test locked and unused; no
leaderboard claim. We do not claim to have solved reference instability,
to be universal across datasets, or to have fully outperformed any
external method.

**Recommended next step.** The M2 holdout confirmation (6/6) is a
candidate for a Phase U formal-test freeze under a separate protocol and
explicit user approval. Until then, the result is a holdout-confirmed,
pre-registered mitigation that reduces catastrophic failure by 79.2% at
80% coverage on the frozen pipeline, plus a documented negative result
for dynamic budgeting and a cross-dataset failure boundary. The benchmark
protocol, metric suite, and gate structure are released for reuse.

---

## Appendix

### A. Frozen hyperparameters

Prompt configs: P20-N1 (K+=20, K-=1, N_BOTTOMK), P30-N0 (K+=30, K-=0,
N_NONE), P50-N0 (K+=50, K-=0, N_NONE). Six references per query,
patient-disjoint, k∈{1,3,6}. Matcher: DINOv2-L, z-score normalization.
Segmenter: SAM-B, single call per consensus, multimask rule frozen.
Catastrophic threshold: Dice < 0.1. Bootstrap: B=10,000, patient-cluster
for MoNuSAC. M2: logistic regression, L2, 5-fold CV on train; four
test-time-safe features; two labels (instability: gap > train p90;
catastrophic: Dice < 0.1); 80% coverage gate. M1: `U(q,k) =
(1-λ_c-λ_r)·Dice - λ_c·Cost - λ_r·Risk`, λ_c, λ_r ∈ {0, 0.25, 0.5,
0.75, 1.0}.

### B. Per-config WP1 holdout variance decomposition

| config | k | V_ref | V_query | V_ref_frac | mean_ref_std | best-worst gap |
|---|---|---|---|---|---|---|
| P20-N1 | 1 | 0.011088 | 0.031122 | 0.2627 | 0.085208 | 0.239294 |
| P20-N1 | 6 | 0.008304 | 0.035611 | 0.1891 | 0.069922 | 0.225234 |
| P30-N0 | 1 | 0.008818 | 0.032969 | 0.2110 | 0.077195 | 0.214120 |
| P30-N0 | 6 | 0.006790 | 0.035913 | 0.1590 | 0.061878 | 0.196472 |
| P50-N0 | 1 | 0.005186 | 0.035860 | 0.1264 | 0.060222 | 0.166836 |
| P50-N0 | 6 | 0.002579 | 0.039515 | 0.0613 | 0.040395 | 0.132370 |

### C. Per-config WP1 holdout mean Dice and tail risk

| config | k | mean_dice | CVaR_0.1 | CFR |
|---|---|---|---|---|
| P20-N1 | 1 | 0.461742 | 0.056059 | 0.0729 |
| P20-N1 | 6 | 0.488852 | 0.076556 | 0.0650 |
| P30-N0 | 1 | 0.487949 | 0.075086 | 0.0621 |
| P30-N0 | 6 | 0.516846 | 0.102650 | 0.0575 |
| P50-N0 | 1 | 0.478970 | 0.083139 | 0.0550 |
| P50-N0 | 6 | 0.511686 | 0.098674 | 0.0500 |

### D. WP1 gate condition detail

| # | Condition | Result | Threshold |
|---|---|---|---|
| 1 | fraction best-worst gap ≥ 0.05 | 0.9250 | ≥ 0.50 |
| 2 | mean within-query reference std | 0.074208 | ≥ 0.02 |
| 3 | V_ref_frac per config (k=1) | 0.1264–0.2627 | ≥ 0.10 |
| 4 | consensus gain (k=6 vs k=1) | +0.026631 | ≥ 0.010 |
| 5 | paired bootstrap 95% CI lower | 0.014009 | > 0 |
| 6 | consensus CFR vs single CFR | 0.0575 vs 0.0633 | ≤ |
| 7 | consensus CVaR vs single CVaR | 0.091945 vs 0.075133 | ≥ single − 0.005 |
| 8 | J_pred consensus vs single | 0.718951 vs 0.648965 | ≥ |
| 9 | best |Spearman(feature, gap)| | 0.7241 | ≥ 0.25 |
| 10 | patient-disjoint violations / 2400 | 0 | 0 |

### E. M2 validation vs holdout

| metric | validation | holdout |
|---|---|---|
| instability AUROC | 0.9443 | 0.9283 |
| catastrophic AUPRC | 0.4338 | 0.5277 |
| ECE | 0.0425 | 0.0270 |
| CFR reduction @80% | 0.3924 | 0.7917 |
| Dice change @80% | +0.0214 | +0.0090 |
| prevalence (catastrophic) | 0.1171 | 0.0500 |

### F. M1 model comparison (validation)

| model | λ_c | λ_r | mean_dice | avg_k | CVaR_0.1 | CFR |
|---|---|---|---|---|---|---|
| M1-0 always_k1 | 0 | 0 | 0.414136 | 1.0000 | 0.059762 | 0.117073 |
| M1-1 always_k3 | 0 | 0 | 0.443245 | 3.0000 | 0.054352 | 0.115447 |
| M1-2 always_k6 | 0 | 0 | 0.458553 | 6.0000 | 0.048999 | 0.125203 |
| M1-3 handcrafted | 0 | 0 | 0.457056 | 5.6341 | 0.053718 | 0.120325 |
| M1-4 logistic | 0 | 0 | 0.455240 | 5.8780 | 0.048977 | 0.125203 |
| M1-5 gradient_boosting | 0 | 0 | 0.451626 | 4.9805 | 0.052773 | 0.117073 |

### G. Disagreement-feature Spearman |ρ| (WP1 holdout, k=1)

| feature | Spearman |ρ| |
|---|---|
| prediction_jaccard_k1_within | 0.7241 |
| matching_map_disagreement | 0.1434 |
| candidate_jaccard_mean_k1 | 0.1030 |
| prompt_overlap_mean_k1 | 0.1030 |

### H. Prohibited-claim audit

The following tokens do not appear in any Phase T paper section except
inside the claim registry as documentation of the prohibition: "SOTA",
"state-of-the-art", "instability solved", "instability eliminated",
"universal across datasets", "mean Dice alone proves robustness",
"submission-ready", "publication-completed". Negated occurrences (e.g.,
"we do not claim SOTA") are permitted and used.

---

*End of draft. This is a paper candidate, not a completed publication.
The formal test set is locked and unused for the entire Phase T program.
The claim-boundary audit scans every Phase T artifact for prohibited
tokens before any final state transition.*
