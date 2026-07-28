# Phase T — Rebuttal Plan

Date: 2026-07-23
Reference: `phaseT_mock_review.md` (this document addresses every criticism `Cx.y` and question `Qx.y` raised there)
Directive: §18 (mock review + rebuttal plan)

## How to read this plan

For each criticism from each reviewer, four fields are provided:
- **Current evidence**: artifacts / results we already have.
- **Needed experiments**: feasible additions (within the 195 GPU-hour budget unless noted).
- **Unresolvable limitations**: what we honestly cannot fix without scope creep or rule violation.
- **Impact on submission**: Fatal / Major / Minor.

Experiments are consolidated and prioritized in the final "Consolidated Action Plan."

---

## Reviewer 1 (Supportive)

### C1.1 — Narrative leans on negatives; lead with M2

- **Current evidence**: M2 holdout 6/6 PASS, CFR reduction 79.2%, Dice +0.0090 (`phaseT_wp3_m2_holdout/m2_holdout_summary.md`). M1 NO_GAIN and IDRiD FAIL are supporting boundary evidence, not the headline.
- **Needed experiments**: None. Restructure the paper outline so M2 is Section 4 (primary contribution) and M1 / IDRiD are Section 5 (boundaries).
- **Unresolvable limitations**: None.
- **Impact**: **Minor.** Presentation fix.

### C1.2 — Anti-correlation needs more than one number

- **Current evidence**: Catastrophic-trained model's instability AUROC = 0.0600 (val); instability-trained model's catastrophic AUPRC = 0.0710 vs prevalence 0.1171 (i.e. below random) (`phaseT_wp3_m2/m2_gate_summary.md`). Two-directional failure supports the anti-correlation.
- **Needed experiments**: (a) scatter plot of per-episode instability-gap vs Dice; (b) Pearson/Spearman correlation between the two labels across the 1200 train episodes; (c) permutation test (shuffle catastrophic labels, retrain, report instability-AUROC distribution). Cost: <1 GPU-hour.
- **Unresolvable limitations**: None significant.
- **Impact**: **Major → recoverable to Minor.** This is the central scientific claim and must be statistically defensible.

### C1.3 — Missing qualitative figures

- **Current evidence**: 18 figures are scaffolded in `phaseT_figures/` (fig_14 high-instability examples, fig_15/16 success/failure cases). The PNGs exist.
- **Needed experiments**: Ensure figures 14–18 are included in the submission draft; add a per-query side-by-side of high-gap vs low-gap episodes.
- **Unresolvable limitations**: None.
- **Impact**: **Minor.** Presentation.

### C1.4 — Why not a shared multi-task head?

- **Current evidence**: The dual-model was chosen because a single model cannot satisfy both cond1 (instability AUROC ≥ 0.70) and cond2–5 (catastrophic selective prediction). The catastrophic model's instability AUROC 0.0600 is the empirical justification.
- **Needed experiments**: Train a shared-backbone multi-task logistic regression / small MLP with two output heads on the same four features; report whether it passes both cond1 and cond4. Cost: <1 GPU-hour.
- **Unresolvable limitations**: None.
- **Impact**: **Minor → strengthens novelty defense** against C2.4 and C4.4.

### Q1.1 / Q1.2 — AUROC drop and Dice-change drop val→holdout

- **Current evidence**: AUROC 0.9443 → 0.9283 (drop 0.016); Dice change +0.0214 → +0.0090. Both directions consistent (cond6 PASS).
- **Needed experiments**: DeLong / cluster-bootstrap CI on both AUROCs; report whether they overlap. Prevalence-matched Dice-change evaluation (subsample val to prevalence 0.05).
- **Unresolvable limitations**: Prevalence shift (0.1171 → 0.0500) is intrinsic to the holdout split.
- **Impact**: **Minor** (consistency holds; magnitude needs CI).

---

## Reviewer 2 (Novelty-Skeptic)

### C2.1 — V_ref decomposition is standard ANOVA

- **Current evidence**: The correction fixed a code-level axis-mixing bug in Phase R (`phaseS_trackA_metric_audit/variance_code_trace.md`). The contribution is not the decomposition itself but its application to separate case difficulty from reference-induced variance — a distinction prior visual-reference work does not report.
- **Needed experiments**: None. Reframe in paper as "application of variance decomposition to the reference axis," not "novel decomposition."
- **Unresolvable limitations**: The decomposition is indeed standard; we cannot claim it as a method contribution.
- **Impact**: **Minor.** Repositioning.

### C2.2 — Mean consensus is ensemble averaging

- **Current evidence**: B2_mean is a standard baseline; we claim it as the *strongest automatic baseline*, not a contribution (`phaseT_novelty_audit.md` §4). The contribution is the *holdout confirmation* that it remains strongest and the M1 result that *dynamic* sizing fails.
- **Needed experiments**: None.
- **Unresolvable limitations**: Cannot make averaging novel.
- **Impact**: **Minor.** Already acknowledged in §4 of the novelty audit.

### C2.3 — "We don't select" is thin vs SAMIX

- **Current evidence**: SAMIX is Tier 4 (requires RL training of SPNet + SAM2 adaptation; wrong paradigm). On IDRiD, consensus hurts Dice (−0.0118), contradicting the SAMIX premise.
- **Needed experiments**: If SAMIX code were available, run its selector on our reference pool. It is not available. Alternative: implement a simple greedy reference selector (maximize DINOv2 matching score, or maximize leave-one-out consensus contribution) and show it does not beat B2_mean — this is a proxy for the selection paradigm.
- **Unresolvable limitations**: SAMIX itself cannot be run (no code).
- **Impact**: **Major → Minor** with the proxy experiment.

### C2.4 — M2 is generic selective prediction; "dual-model" is a rebrand

- **Current evidence**: The dual-model is forced by anti-correlation (catastrophic model's instability AUROC 0.0600). Multi-task with shared parameters has not been tested (see C1.4 needed experiment).
- **Needed experiments**: Shared-head multi-task baseline (C1.4). If it fails both gates, the dual-model is empirically necessary, not a rebrand.
- **Unresolvable limitations**: The classifier choice is deliberately simple (logistic regression for calibration); we will not claim method novelty in the classifier itself.
- **Impact**: **Major → Minor** with C1.4 experiment.

### C2.5 — ProSAM/MAUP differentiation is conceptual, not empirical

- **Current evidence**: GF-SAM and MAUP are Tier 2 reproducible; protocol alignment is documented (`phaseT_external_baselines/protocol_alignment.md`). They are **not yet reproduced** — reproducing them requires user authorization to download and run external code, which has not been granted.
- **Needed experiments**: With user authorization, reproduce GF-SAM or MAUP on the MoNuSAC holdout under the unified protocol; compute their `V_ref_frac`, best-worst gap, and CFR. Estimated 20–40 GPU-hours each.
- **Unresolvable limitations**: ProSAM, SAMIX, AlignSAM, Point Prompt Defender have no public code/weights (Tier 3/4) — cannot be reproduced regardless of budget.
- **Impact**: **Major.** This is the highest-leverage addition. Publication readiness condition 3 explicitly allows "≥1 external method reproduced OR documented justification + strong internal baselines." We currently rely on the fallback; reproducing GF-SAM or MAUP converts this from Major to Minor.

### C2.6 — "Prediction agreement predicts quality" is tautological

- **Current evidence**: The best feature (prediction_jaccard_k1_within) has |ρ|=0.7241 on holdout; the other three features have |ρ| ≤ 0.14. The predictability is not the novelty — the *holdout-confirmed selective deployment* is.
- **Needed experiments**: Ablate the feature set: run M2 with only the non-tautological features (matching_map_disagreement, candidate_jaccard, prompt_overlap) and report AUROC / CFR. If it still passes, the tautology objection weakens.
- **Unresolvable limitations**: prediction_jaccard is inherently correlated with quality.
- **Impact**: **Minor.** Reframe predictability as a means, not the contribution.

### Q2.1 / Q2.2 / Q2.3

- Q2.1: GF-SAM/MAUP not reproduced due to a user-authorization requirement for external-code download; the protocol alignment is ready. **Needed**: user authorization.
- Q2.2: See C1.4 needed experiment.
- Q2.3: Sensitivity analysis over catastrophic thresholds (p05, p10, p15, fixed 0.1). Cost: <1 GPU-hour.

---

## Reviewer 3 (Data-Scale and Statistics)

### C3.1 — Holdout n=40 too small for AUROC; report DeLong CI

- **Current evidence**: AUROC 0.9283 (episode-level) and 1.0000 (query-level) on 40 queries (`phaseT_wp3_m2_holdout/m2_holdout_summary.md`).
- **Needed experiments**: DeLong CI on episode-level AUROC accounting for within-query clustering (cluster-bootstrap by query). Report query-level CI via leave-one-out.
- **Unresolvable limitations**: n=40 is fixed by the patient-disjoint holdout split; we cannot enlarge it without violating the formal-test lock.
- **Impact**: **Major.** The AUROC claim must carry a CI; if the CI excludes 0.7 the claim survives.

### C3.2 — Asymmetric seed counts (10 vs 5)

- **Current evidence**: WP1 uses 10 seeds (holdout, confirmatory); WP2 uses 5 seeds (exploratory cross-dataset). This is documented but not justified in the reviewed material.
- **Needed experiments**: Power analysis showing 5 seeds gives ±X on the `V_ref_frac` estimate. Optionally rerun WP2 with 10 seeds if budget allows (~30 GPU-hours).
- **Unresolvable limitations**: None material; 5 seeds was a resource choice.
- **Impact**: **Minor → Major** if the IDRiD/MoNuSeg difference is seed-sensitive. Run the sensitivity check first.

### C3.3 — IDRiD "consensus hurts" is statistically null

- **Current evidence**: IDRiD consensus gain −0.0118, bootstrap CI lower −0.0318 (upper not reported but likely positive) (`phaseT_wp2_idrid/gateWP2_summary.md`).
- **Needed experiments**: Report the full CI (upper bound). Reframe IDRiD as "consensus does not significantly help" rather than "consensus significantly hurts." Run a non-inferiority test against a −0.01 margin.
- **Unresolvable limitations**: n=30 limits power.
- **Impact**: **Major → Minor** with reframing. The directional FAIL (d3) is a gate failure, not a significance claim; we must not overstate it.

### C3.4 — Anti-correlation is one number without a test

- **Current evidence**: Catastrophic model's instability AUROC 0.0600; instability model's catastrophic AUPRC 0.0710 < prevalence 0.1171.
- **Needed experiments**: (a) Pearson/Spearman ρ between instability-gap and Dice across episodes (expect negative); (b) permutation test: shuffle catastrophic labels, retrain, report the AUROC distribution; (c) report CI on the 0.0600 AUROC.
- **Unresolvable limitations**: None.
- **Impact**: **Major.** Pairs with C1.2. Must be done before submission.

### C3.5 — Multiple comparisons without correction

- **Current evidence**: ~40 gate tests across WP1 / WP2 / M2, uncorrected.
- **Needed experiments**: Report Bonferroni- and Holm-corrected p-values / CIs for all gate conditions near threshold. Specifically MoNuSeg s2 (CI lower 0.0069) and WP1 cond5 (0.0140).
- **Unresolvable limitations**: The gate thresholds are pre-registered; correction is post-hoc reporting, not re-thresholding.
- **Impact**: **Minor.** Reporting, not re-running.

### C3.6 — M2 holdout absolute counts are small

- **Current evidence**: ~60 catastrophic episodes → ~10 at 80% coverage; Dice +0.0090 ≈ one good episode per query.
- **Needed experiments**: Report absolute episode counts in the paper, not just rates. Add a "practical significance" paragraph.
- **Unresolvable limitations**: n=40 queries × 30 episodes = 1200 episodes is the maximum without test access.
- **Impact**: **Minor.** Transparency.

### C3.7 — AUPRC not comparable across prevalences

- **Current evidence**: Val AUPRC 0.4338 (prevalence 0.1171); holdout AUPRC 0.5277 (prevalence 0.0500).
- **Needed experiments**: Prevalence-matched evaluation: subsample val to prevalence 0.05 and recompute AUPRC; or report AUPRC relative to prevalence (lift = AUPRC / prevalence).
- **Unresolvable limitations**: Prevalence shift is intrinsic to the split.
- **Impact**: **Minor.** Reframing.

### Q3.1 / Q3.2 / Q3.3

- Q3.1: Addressed in C3.1 and C3.4.
- Q3.2: Addressed in C3.4 needed experiments.
- Q3.3: Addressed in C3.2.

---

## Reviewer 4 (Differentiation from ProSAM/MAUP/SAMIX)

### C4.1 — "We don't select" too thin vs SAMIX

- **Current evidence**: IDRiD consensus hurts (−0.0118). SAMIX is Tier 4 (no code).
- **Needed experiments**: Implement a greedy reference selector (top-k by DINOv2 matching score, or by leave-one-out consensus contribution) as a SAMIX proxy. Show it does not beat B2_mean on MoNuSAC or IDRiD.
- **Unresolvable limitations**: SAMIX itself cannot be run.
- **Impact**: **Major → Minor** with the proxy experiment.

### C4.2 — MAUP differentiation contradicted by M1

- **Current evidence**: M1 collapses to k=6 across all five policy variants (`phaseT_wp3_m1/m1_gate_summary.md`).
- **Needed experiments**: Rephrase M1 as "a generalization of MAUP's boundary to single-scale DINOv2 + single-call SAM-B," not a differentiator. Optionally run MAUP (Tier 2) to show its multi-center uncertainty also collapses under this pipeline.
- **Unresolvable limitations**: Without MAUP reproduction, the generalization claim is conceptual.
- **Impact**: **Major → Minor** with reframing + optional MAUP run.

### C4.3 — ProSAM differentiation unverifiable

- **Current evidence**: ProSAM is Tier 3 (no code). Differentiation is conceptual only.
- **Needed experiments**: None feasible without a code release.
- **Unresolvable limitations**: ProSAM cannot be reproduced.
- **Impact**: **Minor** (acknowledged limitation) — but contributes to the C4.5 aggregate.

### C4.4 — Dual-model is methodologically lightweight

- **Current evidence**: Logistic regression is deliberately simple for calibration (ECE 0.0270 holdout). The gradient-boosting variant (`m2_gate_summary.md`) achieves AUROC 0.9265 — *worse* than logistic regression's 0.9443, supporting the choice.
- **Needed experiments**: Reposition the paper as a *finding* paper (anti-correlation) with a *minimal deployable method*, not a method paper. Cite the GBM ablation as evidence that more capacity does not help.
- **Unresolvable limitations**: Cannot make logistic regression "novel" as a method.
- **Impact**: **Major → Minor** with repositioning.

### C4.5 — No head-to-head with any external method

- **Current evidence**: Zero external methods reproduced. Internal baselines B1–B7 are all variants of our own pipeline.
- **Needed experiments**: Reproduce GF-SAM or MAUP (Tier 2) on the MoNuSAC holdout. Protocol alignment is ready (`phaseT_external_baselines/protocol_alignment.md`). Estimated 20–40 GPU-hours. **Requires user authorization to download external code.**
- **Unresolvable limitations**: ProSAM, SAMIX, AlignSAM, Point Prompt Defender (Tier 3/4) cannot be reproduced.
- **Impact**: **Major (highest leverage).** Same root cause as C2.5. This is the single most important addition for acceptance.

### C4.6 — "Frozen backbones" is field norm, not differentiator

- **Current evidence**: Acknowledged in novelty audit §4 ("not new").
- **Needed experiments**: None. Remove "frozen backbones" from the novelty claim list.
- **Unresolvable limitations**: None.
- **Impact**: **Minor.** Framing.

### Q4.1 / Q4.2 / Q4.3

- Q4.1: GF-SAM on MoNuSAC holdout, k ∈ {1,3,6}, 10 seeds — estimated 20–30 GPU-hours. MAUP similar.
- Q4.2: Needs SAMIX code (unavailable); proxy in C4.1.
- Q4.3: Would require mask-prompt SAM runs; feasible at ~10 GPU-hours. Lower priority.

---

## Reviewer 5 (Medical-Only / Generalizability)

### C5.1 — Medical-only evaluation

- **Current evidence**: Three datasets, all medical (MoNuSAC, MoNuSeg, IDRiD). The cross-dataset boundary (MoNuSeg PASS, IDRiD FAIL) is within-medical.
- **Needed experiments**: Run the pipeline on a non-medical few-shot segmentation benchmark (PASCAL-5i or COCO-20i) with the same DINOv2-L + SAM-B + reference-resampling protocol. Estimated 30–50 GPU-hours.
- **Unresolvable limitations**: If non-medical data is out of scope for this submission, reposition as a medical-imaging paper (see C5.5).
- **Impact**: **Major.** Either add ≥1 non-medical dataset or explicitly reposition.

### C5.2 — IDRiD failure may be domain-specific (low base Dice)

- **Current evidence**: IDRiD base Dice ~0.09–0.19; CFR 0.89–1.00. The "consensus hurts" finding is on a near-failing segmenter.
- **Needed experiments**: Control for base Dice: stratify datasets by mean-Dice bucket and re-compare consensus gain. Report whether the sign of consensus gain correlates with base Dice across the three datasets.
- **Unresolvable limitations**: IDRiD's difficulty is intrinsic to the dataset.
- **Impact**: **Major → Minor** with stratified analysis. Reframe IDRiD as "when base Dice is very low, consensus does not rescue."

### C5.3 — "3 datasets, 1 domain"

- **Current evidence**: All medical.
- **Needed experiments**: See C5.1.
- **Unresolvable limitations**: If no non-medical dataset is added, this stands.
- **Impact**: **Major** (same root cause as C5.1).

### C5.4 — M2 features unvalidated cross-domain

- **Current evidence**: Four features validated only on medical data.
- **Needed experiments**: If a non-medical dataset is added (C5.1), retrain M2 on it and report AUROC / CFR. Alternatively, report feature-distribution shift statistics between MoNuSAC and a natural-image proxy.
- **Unresolvable limitations**: Without cross-domain data, the "test-time-safe features" claim is medical-only.
- **Impact**: **Major** (depends on C5.1).

### C5.5 — Position as medical contribution

- **Current evidence**: Claim registry already prohibits "universal across datasets."
- **Needed experiments**: Explicitly retitle / reposition: "Reference-Induced Instability in *Medical* Visual-Reference Segmentation." Narrow the abstract claims.
- **Unresolvable limitations**: If we reposition, the contribution's breadth shrinks but its defensibility rises.
- **Impact**: **Minor** (positioning choice; a fallback if C5.1 is not done).

### Q5.1 / Q5.2 / Q5.3

- Q5.1: See C5.1 needed experiments.
- Q5.2: IDRiD `matching_map_disagreement` |ρ| = 0.1017 is the weakest of all datasets; this supports the DINOv2-matching-quality explanation. Add to paper.
- Q5.3: Catastrophic threshold is dataset-specific (train p10 Dice); sensitivity analysis in Q2.3.

---

## Consolidated Action Plan

Prioritized list of experiments to address the Major issues, within the 195 GPU-hour budget:

| Priority | Experiment | Addresses | Est. GPU-h | Requires |
|---|---|---|---|---|
| 1 | Reproduce GF-SAM or MAUP on MoNuSAC holdout | C2.5, C4.5 | 20–40 | User authorization |
| 2 | Anti-correlation permutation test + CIs + scatter | C1.2, C3.4 | <1 | None |
| 3 | Shared-head multi-task M2 ablation | C1.4, C2.4, C4.4 | <1 | None |
| 4 | Non-medical dataset (PASCAL-5i / COCO-20i) | C5.1, C5.3, C5.4 | 30–50 | Data license check |
| 5 | DeLong / cluster-bootstrap CIs on all AUROCs | C3.1, C3.7 | <1 | None |
| 6 | WP2 rerun with 10 seeds | C3.2 | ~30 | None |
| 7 | SAMIX-proxy greedy selector | C2.3, C4.1 | <1 | None |
| 8 | Base-Dice-stratified consensus analysis | C5.2 | <1 | None |
| 9 | Catastrophic-threshold sensitivity (p05/p10/p15/0.1) | Q2.3, Q5.3 | <1 | None |
| 10 | Bonferroni / Holm-corrected gate reporting | C3.5 | 0 (analysis) | None |

Total estimated: 80–125 GPU-hours, within the 195 GPU-hour cap.

Experiments 2, 3, 5, 7, 8, 9, 10 are essentially free (CPU-feasible, <5 GPU-hours total) and should be done regardless of route. Experiments 1 and 4 are the gating decisions: they require external authorization / data and together consume 50–90 GPU-hours.

## Unresolvable Limitations (Aggregate)

1. **ProSAM, SAMIX, AlignSAM, Point Prompt Defender** cannot be reproduced (no public code/weights). Differentiation remains conceptual.
2. **n=40 holdout** is fixed by the patient-disjoint split; the formal test set is locked and cannot be accessed.
3. **IDRiD base difficulty** is intrinsic to the dataset; we cannot make the segmenter succeed there.
4. **SAMIX code absence** blocks a direct selection-vs-resampling comparison; only a proxy is possible.
5. **Prevalence shift** val → holdout (0.1171 → 0.0500) is intrinsic to the data split and affects AUPRC comparability.
6. **Medical-only scope** is a real boundary unless a non-medical dataset is added.

## Impact Summary

- **Fatal**: 0 (no criticism is unaddressable).
- **Major (addressable)**: C2.5 / C4.5 (external methods — same root), C3.1 / C3.4 (statistics of AUROC and anti-correlation), C4.1 / C4.4 (differentiation + repositioning), C5.1 / C5.3 / C5.4 (non-medical or repositioning), C3.3 (IDRiD framing), C4.2 (M1 framing), C3.2 (seed-count sensitivity), C5.2 (base-Dice confound).
- **Minor (presentation / analysis)**: C1.1, C1.3, C2.1, C2.2, C2.6, C3.5, C3.6, C3.7, C4.3, C4.6, C5.5.

## Submission Route Implication

- **Hybrid route (all 12 publication-readiness conditions)**: requires GF-SAM or MAUP reproduction (condition 3) plus the statistical additions (experiments 2, 3, 5). Feasible if the user authorizes external code. The non-medical dataset (experiment 4) is not strictly required by condition 3 but is required to neutralize R5.
- **Benchmark-only route**: conditions 1, 2, 4, 5, 6, 8, 9, 10, 11, 12 pass; condition 3 via the "documented justification + strong internal baselines" fallback; condition 7 (M2 PASS) is already satisfied. This route is currently viable and survives most reviewer criticisms *except* C4.5 / C2.5 (external methods) and C5.1 (non-medical), which would remain Major.
- **Recommended**: pursue the Hybrid route by completing experiments 1–5 in the action plan. If external-code authorization is denied, fall back to Benchmark-only with explicit medical-imaging repositioning (C5.5) and accept that C4.5 / C2.5 / C5.1 remain Major-but-defensible limitations disclosed in the paper.

## Bottom Line

No criticism is fatal. The paper's core (holdout-confirmed benchmark + dual-model selective prediction + cross-dataset boundary) survives if the anti-correlation claim is statistically shored up (experiments 2, 3, 5) and if at least one of {external method reproduction, non-medical dataset} is added. The mock review's aggregate Weak-Reject/Borderline is recoverable to a Weak-Accept by executing priorities 1–5 of the action plan; it is recoverable to a defensible Benchmark-only submission by executing priorities 2, 3, 5, 7, 8, 9, 10 plus repositioning.
