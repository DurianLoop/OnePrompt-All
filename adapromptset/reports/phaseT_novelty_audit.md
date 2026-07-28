# Phase T — Novelty Audit (Directive §17)

Date: 2026-07-23
Branch: `phaseT-publication-completion`
Directive: `D:\SAM\adapromptset\reports\AdaPromptSet_PhaseT_Publication_Completion.md` (§17)
Reference: `reports/phaseS_novelty_audit.md` (Phase S, §13)

## 0. Scope of this Audit

This audit updates the novelty analysis from Phase S in light of the Phase T
results (WP1 holdout, WP2 cross-dataset replication, WP3 M1 dynamic budget,
WP3 M2 selective prediction). It is written to support §17 of the Phase T
directive, which requires explicit differentiation from ten prior
works/fields and articulation of the core novelty candidate:

> "This work does not merely select references or generate stronger prompts.
> It systematically measures the prediction distribution, tail risk, and
> failure probability under reference resampling, and studies dynamic
> reference budget and selective prediction."

The audit is evidence-based: every differentiating claim cites a Phase T
result file. Prohibited claims from `docs/phaseT_claim_registry.md` are
respected throughout (no "SOTA", no "instability solved", no "universal
across datasets").

---

## 1. Executive Summary

### What is novel (Phase T increments over Phase S)

1. **Holdout-confirmed instability benchmark.** Phase S established the
   phenomenon on the validation split (n=41). Phase T confirms it on a
   patient-disjoint, pre-registered MoNuSAC holdout (n=40, 10 seeds, 9600
   SAM calls) with 10/10 gate conditions passing
   (`reports/phaseT_wp1_holdout/gateWP1_summary.md`): `V_ref_frac` in
   0.0613–0.2627 across configs, 92.5% of queries with best-worst gap ≥ 0.05,
   consensus gain +0.0266 Dice (paired bootstrap 95% CI lower = 0.0140), and
   zero leakage across 2400 reference assignments. This converts a
   validation-split observation into a held-out, pre-registered finding.

2. **Cross-dataset boundary of the phenomenon.** Phase T shows the
   instability is **dataset-dependent**, not universal. MoNuSeg replicates
   with strong support (9/9 conditions, `reports/phaseT_wp2_monuseg/`),
   while IDRiD **fails** (4/5 directional, 2/4 strong,
   `reports/phaseT_wp2_idrid/`): on IDRiD, mean consensus *hurts* Dice
   (gain = −0.0118, bootstrap CI lower = −0.0318). This negative result is
   itself novel and bounds the claim.

3. **A new scientific finding: instability and catastrophic failure are
   anti-correlated.** Episodes where all references fail similarly have low
   best-worst gap but low Dice. This is documented in
   `reports/phaseT_wp3_m2/m2_gate_summary.md` and forces a **dual-model**
   design: one model trained on the instability label (drives AUROC), a
   separate model trained on the catastrophic-failure label (drives
   selective prediction). No prior audited work makes this distinction.

4. **Holdout-confirmed selective prediction (M2 PASS, 6/6).** A
   logistic-regression selective predictor on four test-time-safe features
   passes on validation (AUROC 0.9443, catastrophic AUPRC 0.4338 vs
   prevalence 0.1171, CFR reduction 39.2% at 80% coverage) and **confirms
   on holdout** (AUROC 0.9283, AUPRC 0.5277 vs prevalence 0.0500, CFR
   reduction **79.2%**, Dice change +0.0090;
   `reports/phaseT_wp3_m2_holdout/m2_holdout_summary.md`).

5. **A documented negative result for dynamic reference budget (M1
   NO_GAIN).** Cost–quality optimization selects `lambda_c=lambda_r=0`,
   i.e. the policy collapses to "always k=6"; cost-saving mode fails
   (avg_k=5.63, only 6.1% reduction vs the 25% gate) and quality-cost mode
   fails (`reports/phaseT_wp3_m1/m1_gate_summary.md`). This is a
   failure-boundary contribution: dynamic budgeting cannot mitigate
   instability under this pipeline.

### What is NOT novel (carried from Phase S or general practice)

- The matching-and-prompting pipeline (DINOv2-L + frozen SAM-B) is reused
  from Phase X/R/S without architectural change.
- Mean consensus (B2_mean) as an aggregator is a standard baseline; we do
  not claim it as our method, only as the strongest automatic baseline.
- The four test-time-safe features used by M2 are the disagreement features
  already introduced in Phase S; M2's contribution is the *selective
  prediction framing and dual-model design*, not new feature engineering.
- No new segmenter, no new prompt generator, no fine-tuning of SAM.

### Net positioning

Phase T's core contribution is a **risk-aware characterization** of
reference-induced instability: a held-out benchmark of prediction
distribution + tail risk + failure probability under reference resampling,
plus a dual-model selective-prediction method that is holdout-confirmed and
a dynamic-budget method that is a documented negative result. This matches
the §17 core-novelty candidate and the allowed paper type
(benchmark + failure-boundary + one positive mitigation).

---

## 2. Method-by-Method Differentiation

For each of the ten items required by §17, we state (a) what they do,
(b) what we do differently, and (c) the Phase T evidence supporting the
difference.

### 2.1 GF-SAM — graph grouping for reference selection

(a) GF-SAM builds a graph over candidate references/prompts and uses graph
grouping to select a coherent reference set, with the goal of improving
prompt quality.

(b) We do **not** select references. The reference set is the full
per-query pool of 6 references, resampled across 10 seeds. Our object of
study is the *distribution of predictions induced by resampling*, not the
quality of a selected subset. Where GF-SAM asks "which references are
best?", we ask "how much does the answer change when the reference set
changes, and what is the tail risk?"

(c) WP1 holdout: `V_ref_frac` 0.0613–0.2627 and 92.5% of queries with
best-worst gap ≥ 0.05 show that reference *resampling*, not reference
*selection*, is the dominant measurable effect. M1 further shows that
even an optimized dynamic budget over references collapses to k=6 (no
selection gain), reinforcing that the phenomenon is distributional, not
selectional.

### 2.2 SynPo — positive/negative point construction

(a) SynPo synthesizes positive/negative point prompts (including synthetic
negatives) to augment the prompt space and improve SAM's robustness to
point placement.

(b) We use **only real references** and the frozen Phase X candidate pool
(DINOv2-L patch-grid centers). We do not synthesize prompts or construct
synthetic negatives. Our negative results concern natural
reference-induced instability, not adversarial or synthetic prompt
perturbations.

(c) The candidate pool is frozen across all Phase T runs
(`docs/phaseT_reference_protocol.md`); the only varying factor is the
reference subset drawn per seed. WP2 IDRiD's failure (consensus gain
−0.0118) is attributable to the dataset and reference pool, not to prompt
construction, since the prompt construction is identical to MoNuSeg where
consensus helps (+0.0149).

### 2.3 AlignSAM — RL iterative prompting

(a) AlignSAM uses reinforcement learning to iteratively refine prompts
during SAM inference, treating prompt improvement as a sequential
decision problem.

(b) We do **no RL**, no iterative refinement, and no fine-tuning of SAM.
The pipeline is single-call SAM-B with a frozen matcher. Our
"mitigation" is not prompt refinement but *risk-aware abstention* (M2)
and *budget allocation* (M1), both operating on the reference axis, not
the prompt axis.

(c) M2's selective predictor is a logistic regression on four
test-time-safe features — no policy gradient, no reward, no environment
interaction. Its holdout confirmation (6/6, CFR reduction 79.2%) is
achieved without any iterative prompting, demonstrating that the
predictable component of failure risk can be captured statically.

### 2.4 Point Prompt Defender — point filtering

(a) Point Prompt Defender filters or defends against *adversarial* point
prompts, removing points that are hostile perturbations.

(b) We study **natural** reference-induced instability: every point
prompt is a legitimate DINOv2-matched candidate, none are adversarial.
Our M1 "filtering" is over *reference subsets*, not over points, and it
fails (NO_GAIN). The failure is itself the contribution: filtering cannot
fix a distributional problem.

(c) M1 gate: the best policy (`M1-3_handcrafted_threshold`) selects
`lambda_c=lambda_r=0`, i.e. no filtering of references is beneficial
beyond using all 6. Cost-saving mode fails cs4 (avg_k=5.63, 6.1%
reduction vs 25% required). This is a negative result that bounds the
filtering paradigm for *natural* instability.

### 2.5 ProSAM — probabilistic prompt robustness

(a) ProSAM models prompt robustness probabilistically, learning prompt
distributions that are stable under perturbation, with the goal of
producing prompts that SAM answers consistently.

(b) We do not learn prompts and do not model a prompt distribution to be
*made* robust. We treat the prompt distribution induced by reference
resampling as the **object of measurement**: we report its variance
(`V_ref`), tail (`CVaR_0.1`), and failure mass (CFR), and we predict
failure risk from test-time-safe features. Our probability model is over
*failure*, not over *prompts*.

(c) M2 catastrophic model: AUPRC 0.4338 on validation (prevalence 0.1171)
and 0.5277 on holdout (prevalence 0.0500) — a calibrated probability of
*catastrophic failure* per episode, not a probability over prompts. ECE
0.0425 (val) / 0.0270 (holdout) confirms the probability is calibrated.

### 2.6 MAUP — uncertainty-aware prompt quantity

(a) MAUP uses multi-scale anchors and uncertainty to decide *how many*
prompts to use, framing prompt quantity as an uncertainty-driven choice.

(b) We share the question "how many references?" (M1 dynamic budget) but
   (i) we use single-scale DINOv2-L features, not multi-scale anchors,
   (ii) our budget decision is cost–quality-optimized, not
   uncertainty-quantile-driven, and (iii) we report a **negative result**:
   the cost–quality optimum is `lambda=0`, i.e. always k=6. MAUP-style
   uncertainty-aware budgeting is therefore not supported under our
   pipeline.

(c) M1 model comparison (`m1_gate_summary.md`): all five policy variants
   (always-k1, always-k3, always-k6, handcrafted threshold, logistic
   regression, gradient boosting) select `lambda_c=lambda_r=0`.
   `M1-2_always_k6` (Dice 0.4586, CVaR 0.0490) Pareto-dominates the
   learned policies on val. The cost–quality trade-off is unfavorable,
   which is the *opposite* of MAUP's positive prompt-quantity result.

### 2.7 SAMIX — reference selection

(a) SAMIX selects/mixes references to produce a stronger composite
prompt for SAM.

(b) We do **not** select or mix references to build a stronger prompt.
   We use mean consensus (B2_mean) over the full reference set as a
   *baseline*, and our method contributions (M1, M2) operate on the
   *risk* of the reference set, not on improving it. Crucially, we show
   that on IDRiD, mixing/consensus **hurts** Dice (−0.0118), contradicting
   the SAMIX premise that reference aggregation is beneficial.

(c) WP2 IDRiD `gateWP2_summary.md`: d3 FAIL, s1 FAIL, s2 FAIL — consensus
   Dice (k=6) = 0.0904 vs single (k=1) = 0.1022. This is direct evidence
   that reference aggregation is not universally beneficial, a finding
   SAMIX does not report.

### 2.8 Few-shot segmentation (general field)

(a) Few-shot segmentation (PANet, PPNet, etc.) trains a segmenter with a
   support set of labeled images and generalizes to a query; the support
   set is a *training* signal.

(b) We do **not** train a segmenter. SAM-B is frozen; the references are
   used only to condition point prompts via DINOv2 matching. Our
   "support set" is a prompt-conditioning set, not a learning set.
   Furthermore, we benchmark the *instability* of this paradigm, which
   few-shot segmentation works typically do not measure.

(c) All Phase T runs use frozen SAM-B with no gradient updates
   (`docs/phaseT_dataset_registry.md`). WP1 holdout (10/10) and WP2
   MoNuSeg (9/9) replicate the instability phenomenon with zero segmenter
   training, isolating reference-induced variance from any
   learning-induced variance.

### 2.9 Selective prediction (general field)

(a) Selective prediction / abstention methods learn a confidence function
   and abstain on low-confidence inputs to improve risk-coverage.

(b) We **do** perform selective prediction (M2), and here the overlap is
   the largest among the ten items. The differentiation is fourfold:
   1. **Object of abstention**: we abstain on *reference-conditioned
      episodes*, not on inputs in general; the risk we model is
      reference-induced catastrophic failure, not generic uncertainty.
   2. **Dual-model design forced by anti-correlation**: because
      instability (high gap) and catastrophic failure (low Dice) are
      anti-correlated, a single selective predictor cannot both flag
      instability and flag catastrophic risk; we train two models on the
      same features with different labels. This is a structural
      contribution absent from generic selective prediction.
   3. **Test-time-safe features only**: the predictor uses four
      disagreement features computable without query GT, enabling
      deployment.
   4. **Pre-registered holdout confirmation**: the 6/6 holdout pass
      (CFR reduction 79.2%) is a confirmatory result, not a
      validation-split claim.

(c) `m2_holdout_summary.md`: holdout AUROC 0.9283, AUPRC 0.5277 vs
   prevalence 0.0500, ECE 0.0270, CFR reduction 79.2% at 80% coverage,
   Dice change +0.0090. The dual-model rationale and anti-correlation
   are documented in `m2_gate_summary.md` §"Dual-model approach".

### 2.10 Set aggregation (general field)

(a) Set aggregation (DeepSets, Set Transformer, mean/median/trimmed
   ensembles) combines a variable-size set into a fixed representation.

(b) Phase S already tested six deterministic aggregators (B1–B6) plus
   learned weighting and set-conditioned gates, all failing to beat
   B2_mean. Phase T's contribution here is **not** a new aggregator; it
   is (i) holdout confirmation that B2_mean remains strongest
   (WP1 cond4/cond5), and (ii) the M1 result that *dynamic* set sizing
   (a form of adaptive aggregation) also fails. The aggregation question
   is closed negatively under this pipeline.

(c) WP1: consensus (k=6) Dice 0.5058 vs single (k=1) 0.4792, gain
   +0.0266, CI lower 0.0140 — B2_mean confirmed on holdout. M1: all
   dynamic-budget policies collapse to k=6; no adaptive aggregator
   beats static mean consensus.

---

## 3. Core Novelty Claims

The following five claims are the Phase T novelty contributions. Each is
tied to an allowed claim in `docs/phaseT_claim_registry.md` and to a
specific evidence file.

### Claim N1 — Holdout-confirmed reference-induced instability benchmark

On a pre-registered, patient-disjoint MoNuSAC holdout (n=40, 10 seeds,
2400 reference assignments, zero leakage), reference resampling induces
substantial, measurable, predictable segmentation variability: `V_ref_frac`
0.0613–0.2627, 92.5% of queries with best-worst gap ≥ 0.05, disagreement
feature Spearman |ρ| = 0.7241.
Evidence: `reports/phaseT_wp1_holdout/gateWP1_summary.md` (10/10 PASS).
Allowed claim: (d) Holdout-Confirmed Instability.

### Claim N2 — Cross-dataset boundary: instability is dataset-dependent

The phenomenon replicates with strong support on MoNuSeg (9/9, consensus
gain +0.0149, CI lower 0.0069, best Spearman 0.8266) but **fails** on
IDRiD (consensus gain −0.0118, CI lower −0.0318). Instability is real but
not universal; mean consensus can hurt Dice on datasets where references
are uniformly poor.
Evidence: `reports/phaseT_wp2_monuseg/gateWP2_summary.md`,
`reports/phaseT_wp2_idrid/gateWP2_summary.md`.
This bounds claim (d) and is consistent with the prohibition on
"universal across datasets".

### Claim N3 — Anti-correlation between instability and catastrophic failure

Episodes where all references fail similarly exhibit **low** best-worst
gap but **low** Dice. Instability (high gap) and catastrophic failure
(low Dice) are anti-correlated, so a single predictor cannot satisfy
both an instability-AUROC gate and a catastrophic-risk selective gate.
This finding forces the dual-model design and is, to our knowledge, not
reported in any audited prior work.
Evidence: `reports/phaseT_wp3_m2/m2_gate_summary.md` §"Dual-model
approach"; the catastrophic-trained model's instability AUROC is 0.0600
(near-random) while the instability-trained model's AUROC is 0.9443,
empirically demonstrating the anti-correlation.

### Claim N4 — Holdout-confirmed dual-model selective prediction

A dual-model selective predictor (logistic regression on four
test-time-safe disagreement features) is holdout-confirmed: instability
AUROC 0.9283, catastrophic AUPRC 0.5277 (prevalence 0.0500), ECE 0.0270,
CFR reduction **79.2%** at 80% coverage, Dice change +0.0090. This is a
deployable, pre-registered, holdout-confirmed mitigation that does not
retrain the matcher or segmenter.
Evidence: `reports/phaseT_wp3_m2_holdout/m2_holdout_summary.md` (6/6 PASS).
Allowed claim: (e) Risk-Aware Mitigation (M2 route).

### Claim N5 — Documented negative result for dynamic reference budget

Cost–quality-optimized dynamic reference budgeting (M1) collapses to
`lambda_c=lambda_r=0` (always k=6) across all five policy variants.
Cost-saving mode fails (avg_k=5.63, 6.1% reduction vs 25% gate) and
quality-cost mode fails. Dynamic budgeting cannot mitigate instability
under this pipeline — a failure-boundary contribution.
Evidence: `reports/phaseT_wp3_m1/m1_gate_summary.md`
(`PHASET_M1_DYNAMIC_BUDGET_NO_GAIN`).

---

## 4. Non-Novel Aspects (Honest Statement)

To pre-empt reviewer concerns, we explicitly list what is **not** new in
Phase T:

1. **Pipeline architecture.** DINOv2-L matching + frozen SAM-B + point
   prompts is inherited from Phase X/R/S unchanged. Phase T adds no new
   matcher, segmenter, or prompt generator.
2. **Mean consensus aggregator.** B2_mean is a standard baseline. We
   claim it as the *strongest automatic baseline*, not as a method
   contribution.
3. **Disagreement features.** The four test-time-safe features used by
   M2 (`prediction_jaccard_k1_within`, `candidate_jaccard_mean_k1`,
   `prompt_overlap_mean_k1`, `matching_map_disagreement`) were
   introduced in Phase S. M2's novelty is the selective-prediction
   framing and dual-model design, not the features.
4. **V_ref decomposition and tail-risk metrics (CVaR, CFR, worst-subset).
   These are Phase S contributions; Phase T applies them on holdout and
   cross-dataset, but does not invent new metrics.
5. **No SAM fine-tuning, no prompt learning, no new training of the
   segmenter.** The only learned components in Phase T are a logistic
   regression (M2) and small regressors (M1), all on test-time-safe
   features.
6. **Datasets.** MoNuSAC, MoNuSeg, IDRiD are public medical datasets;
   Phase T adds no new data.

These are stated plainly because the paper's value rests on the
*characterization* and *holdout-confirmed mitigation*, not on
architectural novelty.

---

## 5. Risk of Overlap and Rebuttals

This section identifies areas where reviewers may claim insufficient
novelty and prepares evidence-based rebuttals.

### Risk R1 — "This is just reference selection / SAMIX"

**Concern.** The reference-resampling setup may be conflated with
reference selection (SAMIX, §2.7).
**Rebuttal.** We do not select references; we *resample* them and
measure the induced distribution. M1 shows dynamic selection collapses
to k=6 (NO_GAIN), and WP2 IDRiD shows consensus can *hurt* Dice
(−0.0118) — the opposite of a selection-benefit claim. The contribution
is distributional measurement + selective abstention, not selection.

### Risk R2 — "This is just selective prediction"

**Concern.** M2 overlaps with the generic selective-prediction field
(§2.9).
**Rebuttal.** Three differentiators: (i) the abstention object is
*reference-conditioned episodes*, not generic inputs; (ii) the
**dual-model design is forced by an empirically measured
anti-correlation** (instability-trained AUROC 0.9443 vs
catastrophic-trained AUROC 0.0600 on the same features) — a structural
contribution; (iii) pre-registered holdout confirmation (6/6, CFR
reduction 79.2%), not a validation-split claim.

### Risk R3 — "This is just MAUP-style prompt-quantity selection"

**Concern.** M1 dynamic budget may resemble MAUP (§2.6).
**Rebuttal.** We report a **negative result**: all M1 policies collapse
to `lambda=0` (always k=6); cost-saving fails (6.1% reduction vs 25%
gate). This contradicts MAUP's positive prompt-quantity thesis under our
pipeline and is presented as a failure boundary, not a competing
positive method.

### Risk R4 — "Instability is obvious / not a contribution"

**Concern.** Reviewers may argue that reference-induced variance is
expected and thus not novel.
**Rebuttal.** Phase T converts the "expected" intuition into a
*pre-registered, holdout-confirmed, quantified* phenomenon with tail
risk (CVaR, CFR) and predictability (Spearman 0.7241 on holdout). The
cross-dataset failure on IDRiD further shows the phenomenon is
non-trivial: it does not appear wherever references are uniformly poor,
which is itself a measurable scientific boundary (Claim N2 + N3).

### Risk R5 — "Overlap with ProSAM (probabilistic prompt robustness)"

**Concern.** M2's probability output may resemble ProSAM (§2.5).
**Rebuttal.** ProSAM models a distribution *over prompts* to make
prompts robust. M2 models a distribution *over failure* to abstain. The
two are orthogonal: ProSAM changes the prompt; M2 changes the *decision
to deploy*. M2 uses frozen prompts and a calibrated failure probability
(ECE 0.0270 on holdout), with no prompt distribution learning.

### Risk R6 — "Only medical data / limited generality"

**Concern.** All three datasets (MoNuSAC, MoNuSeg, IDRiD) are medical.
**Rebuttal.** This is a real limitation and is acknowledged in the
limitations section; the prohibition on "universal across datasets" is
respected. The cross-dataset result (MoNuSeg PASS, IDRiD FAIL) already
demonstrates that the phenomenon is dataset-dependent, so we do not
over-claim. The benchmark protocol is dataset-agnostic and transferable;
we report where it replicates and where it does not.

### Risk R7 — "Dual-model is just two classifiers"

**Concern.** The M2 dual-model design may be seen as trivially training
two logistic regressions.
**Rebuttal.** The contribution is not the classifier choice (logistic
regression is deliberately simple for calibration) but the **scientific
finding that forces it**: instability and catastrophic failure are
anti-correlated on the same feature space. A single model achieves
AUROC 0.0600 on the axis it was not trained for; the dual-model
separation is empirically necessary, not an engineering preference. The
holdout confirmation (6/6) under a pre-registered gate rules out
validation-split overfitting.

### Risk R8 — "Negative results (M1, IDRiD) are not contributions"

**Concern.** Reviewers may discount M1 NO_GAIN and IDRiD FAIL as
non-contributions.
**Rebuttal.** Both are *failure-boundary* contributions under a
pre-registered protocol: M1 bounds the dynamic-budget paradigm, and
IDRiD bounds the universality of consensus benefit. Combined with the
positive M2 result, they delineate *where* reference-conditioned
mitigation works (selective prediction on catastrophic risk) and *where*
it does not (budget allocation, or datasets with uniformly poor
references). This is the paper's failure-boundary value per the allowed
paper type.

---

## 6. Consistency Check Against the Claim Registry

This audit respects `docs/phaseT_claim_registry.md`:

- Allowed claims (a)–(c) are Phase S unconditional claims, restated
  only as context.
- Allowed claim (d) is supported by WP1 10/10 (Claim N1).
- Allowed claim (e) is supported by M2 6/6 holdout (Claim N4); the M1
  route is a documented negative (Claim N5), not a positive mitigation.
- No prohibited claim appears: no "instability solved", no "universal
  across datasets" (explicitly contradicted by IDRiD), no "SOTA", no
  "formal test success", no "ProSAM/MAUP/SAMIX fully outperformed"
  (only differentiated, not ranked), no "mean Dice alone proves
  robustness" (all robustness claims cite V_ref/CVaR/CFR), no
  "instability is purely random" (predictability is a core finding).

---

## 7. Bottom Line

Phase T's novelty is **not** a new matcher, segmenter, or prompt
generator. It is a **risk-aware characterization** of
reference-induced instability, comprising: (1) a holdout-confirmed
benchmark of prediction distribution, tail risk, and failure
probability under reference resampling; (2) a cross-dataset boundary
(MoNuSeg strong support, IDRiD failure); (3) the anti-correlation
finding between instability and catastrophic failure; (4) a
holdout-confirmed dual-model selective predictor (M2 PASS, 6/6); and
(5) a documented negative result for dynamic reference budget (M1
NO_GAIN). This matches the §17 core-novelty candidate and differentiates
cleanly from all ten required prior works/fields.
