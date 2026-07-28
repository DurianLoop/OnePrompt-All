# Phase T — Mock Peer Review

Date: 2026-07-23
Paper type: Benchmark + failure-boundary + one positive mitigation (per directive §17, §21)
Suggested title: "Reference-Induced Instability in Visual-Reference Segmentation: A Holdout-Confirmed Benchmark, Cross-Dataset Boundary, and Dual-Model Selective Prediction"
Reviewers: 5 simulated (R1–R5) per directive §18
Companion document: `phaseT_rebuttal_plan.md`

## Meta-context

This mock review simulates five reviewers with deliberately distinct stances (supportive, novelty-skeptic, statistics-skeptic, differentiation-skeptic, generalizability-skeptic). Each criticism is numbered `Cx.y` and is addressed individually in `phaseT_rebuttal_plan.md`. Scores follow the CVPR/MICCAI convention (1–10). The reviewers are intentionally adversarial; the goal is to surface every plausible objection before submission.

## Shared Summary

The paper extends a prior Phase S benchmark on reference-induced instability in visual-reference segmentation. Using a frozen DINOv2-L matcher and SAM-B segmenter, the authors: (WP1) confirm on a pre-registered MoNuSAC holdout (n=40, 10 seeds, 9600 SAM calls, 10/10 gate conditions) that reference resampling induces substantial, predictable segmentation variability (`V_ref_frac` 0.0613–0.2627, 92.5% of queries with best-worst gap ≥ 0.05, best disagreement-feature Spearman 0.7241); (WP2) replicate on MoNuSeg (9/9 conditions, strong support) but fail on IDRiD (4/5 directional, consensus hurts Dice by −0.0118, CI lower −0.0318); (WP3 M1) report a documented negative result for dynamic reference budget (all five policies collapse to `lambda_c=lambda_r=0`, i.e. always k=6); (WP3 M2) present a dual-model selective predictor that passes on validation (5/5) and holdout (6/6), reducing catastrophic failure rate by 79.2% at 80% coverage on holdout (Dice +0.0090). The dual-model design is motivated by an empirical anti-correlation between instability and catastrophic failure.

---

## Reviewer 1 (Supportive)

### Reading

This is a carefully executed, pre-registered study that converts a Phase S validation-split observation into a held-out, cross-dataset characterization with a deployable mitigation. The honesty of the negative results (M1, IDRiD) is commendable and rare.

### Strengths acknowledged

1. The 10/10 holdout pass with zero leakage across 2400 reference assignments is a model of protocol discipline.
2. The cross-dataset boundary (MoNuSeg PASS, IDRiD FAIL) is more scientifically informative than a uniform "it works everywhere" claim.
3. The dual-model selective predictor is holdout-confirmed with a 79.2% CFR reduction — a concrete, deployable gain.
4. The claim registry (prohibited tokens) is a useful self-regulation device.

### Criticisms

**C1.1** The narrative leans heavily on negative results (M1 NO_GAIN, IDRiD FAIL). While honest, the paper should lead with the positive M2 contribution to avoid the appearance of a "benchmark-only" paper.

**C1.2** The anti-correlation finding (the central scientific novelty) is supported by a single number: the catastrophic-trained model's instability AUROC = 0.0600. This needs more than one number — a scatter plot, a per-feature breakdown, or a permutation test.

**C1.3** Several figures listed in the protocol (18 figures) are referenced but not shown in the reviewed material; the qualitative examples (high/low instability cases) are essential for reviewer trust.

**C1.4** The M2 dual-model uses two logistic regressions on the same four features. The justification for not sharing representations (e.g., a multi-task head) is not provided.

### Questions

Q1.1 Is the val-to-holdout AUROC drop (0.9443 → 0.9283) within bootstrap noise?
Q1.2 Why does the catastrophic model's Dice change at 80% coverage drop from +0.0214 (val) to +0.0090 (holdout)? Is the gain degrading with lower prevalence?

### Scores
Soundness: 7/10 | Presentation: 7/10 | Contribution: 7/10

### Recommendation: Weak Accept (borderline)

---

## Reviewer 2 (Novelty-Skeptic)

### Reading

The paper positions itself as a "risk-aware characterization" of reference-induced instability. I am not convinced the novelty is as clean as claimed; several "differentiators" are framing choices rather than substantive methodological differences.

### Strengths acknowledged

The pre-registered holdout protocol and the honest IDRiD negative result are above the bar for this venue's reproducibility norms.

### Criticisms

**C2.1** *The V_ref decomposition is variance decomposition.* Separating within-query variance from reference-induced variance is a standard ANOVA-style decomposition. The "Phase R cond3 axis-mixing correction" is a bug fix in prior code, not a methodological contribution. Calling it a "genuine methodological contribution" overstates it.

**C2.2** *Mean consensus is a standard ensemble baseline.* B2_mean is averaging. Demonstrating that averaging beats single-reference inference is the bias–variance trade-off, not a new finding. The +0.0266 Dice gain on holdout is consistent with ordinary ensemble behavior.

**C2.3** *The "we don't select references" differentiator from SAMIX is semantic.* Resampling references and measuring the induced distribution is operationally a study over the reference-selection distribution. SAMIX selects; you marginalize. The distinction is real but thin, and the paper does not run SAMIX to show the distinction matters empirically.

**C2.4** *M2 is generic selective prediction with a "dual-model" rebrand.* Training two logistic regressions on the same features with different labels is multi-task learning with non-shared parameters. The "anti-correlation forces dual-model" argument would be stronger if the authors showed a shared-head multi-task model fails; they do not. The claim that "no prior audited work makes this distinction" is a non-falsifiable absence claim.

**C2.5** *The ProSAM/MAUP differentiation is conceptual, not empirical.* The novelty audit asserts ProSAM models "prompts" while M2 models "failure." Without running ProSAM or MAUP, the authors cannot show their approach yields different outcomes. The paper lists GF-SAM and MAUP as "Tier 2 reproducible but not yet reproduced" — this is the weakest link. A reviewer will ask: if you cannot run them, how is this a comparative paper?

**C2.6** *The "instability is predictable" finding* (Spearman 0.7241 on holdout) uses a single best feature (prediction_jaccard_k1_within). The other three features have |ρ| ≤ 0.14. This is essentially "prediction agreement predicts prediction quality" — a near-tautology.

### Questions

Q2.1 Why were GF-SAM and MAUP not reproduced despite being identified as Tier 2 reproducible? What would it take?
Q2.2 Can you show a shared-parameter multi-task model fails the dual-model claim?
Q2.3 Is the anti-correlation robust to the choice of catastrophic threshold (currently train mean_dice p10 = 0.0676)?

### Scores
Soundness: 5/10 | Presentation: 6/10 | Contribution: 4/10

### Recommendation: Reject (novelty insufficient without external baselines)

---

## Reviewer 3 (Data-Scale and Statistics Skeptic)

### Reading

The claims rest on small samples with multiple uncorrected comparisons. The headline numbers (79.2% CFR reduction, AUROC 0.9283) are impressive in relative terms but tiny in absolute counts. I am concerned about statistical power and the anti-correlation claim's robustness.

### Criticisms

**C3.1** *Holdout n=40 is small for AUROC claims.* The M2 holdout AUROC of 0.9283 is computed on 1200 episodes from 40 queries. But episodes within a query are not independent (same image, same reference pool). The effective n is closer to 40. A DeLong CI on n=40 AUROC is wide; the paper does not report it. The query-level AUROC of 1.0000 on holdout is computed on 40 queries — this is almost certainly overfit to a small set and should be reported with a CI.

**C3.2** *Asymmetric seed counts.* WP1 uses 10 seeds; WP2 uses 5 seeds. The paper does not justify this. With 5 seeds on WP2, the per-query best-worst gap estimate is noisier, which may explain some of the IDRiD/MoNuSeg differences.

**C3.3** *The IDRiD "consensus hurts" claim is not statistically significant.* The bootstrap CI lower bound is −0.0318 and the upper bound is presumably positive. The point estimate −0.0118 is within noise. The paper phrases IDRiD as a "FAIL" and a "boundary finding," but statistically it is a null result that could go either way. Calling it a "novel negative result" overstates the evidence.

**C3.4** *The anti-correlation rests on one number without a test.* The catastrophic-trained model's instability AUROC = 0.0600. Is 0.0600 statistically distinguishable from 0.5 (random)? With n=615 val episodes (effective n≈41 queries), the CI on AUROC 0.06 likely excludes 0.5, but the paper does not report it. More importantly, "anti-correlation" implies a negative correlation between the two labels across episodes; the paper reports a model-transfer failure, not a correlation coefficient. These are different claims.

**C3.5** *Multiple comparisons without correction.* WP1 has 10 gate conditions, WP2 has 9 per dataset, M2 has 6. Across WP1 + WP2 (×2) + M2 (×2) that is ~40 tests. With Bonferroni, some "PASS" conditions near threshold would fail. The bootstrap CI lower on MoNuSeg s2 (0.0069) is close to zero; with correction it may not survive.

**C3.6** *M2 holdout absolute counts.* Holdout prevalence (catastrophic) = 0.0500 × 1200 = 60 catastrophic episodes. At 80% coverage, CFR drops to 0.0104 × 960 ≈ 10 episodes. The 79.2% reduction is thus moving from ~60 to ~10 catastrophic episodes — a difference of ~50 episodes across 40 queries. The Dice change +0.0090 is roughly one good episode per query. These are small absolute effects and the paper should report them as such.

**C3.7** *Val-to-holdout prevalence shift.* Catastrophic prevalence drops from 0.1171 (val) to 0.0500 (holdout). AUPRC is prevalence-sensitive; the holdout AUPRC of 0.5277 vs prevalence 0.0500 looks strong, but AUPRC comparisons across prevalences are not directly comparable. The "holdout confirms val" claim needs prevalence-matched evaluation.

### Questions

Q3.1 Report DeLong or bootstrap CIs for all AUROCs and query-level metrics.
Q3.2 Permutation test for the anti-correlation: shuffle the catastrophic labels and retrain — does the instability AUROC of the catastrophic model remain near 0.06?
Q3.3 Why 5 seeds for WP2 but 10 for WP1? Power analysis?

### Scores
Soundness: 4/10 | Presentation: 6/10 | Contribution: 5/10

### Recommendation: Reject (statistical evidence insufficient; major revision possible)

---

## Reviewer 4 (Differentiation from ProSAM/MAUP/SAMIX)

### Reading

The paper's novelty audit (§2) is thorough on paper but evades the core question: do the claimed methodological differences produce different empirical outcomes? Without running the comparators, the differentiations are assertions.

### Criticisms

**C4.1** *"We don't select references" is too thin against SAMIX.* SAMIX selects/mixes references to build a stronger prompt. You resample and aggregate by mean. The distinction (selection vs. resampling) is a framing of the same operation over the reference set. The IDRiD result (consensus hurts) is genuinely interesting, but it does not differentiate you from SAMIX — it contradicts SAMIX's premise, which is a different (and stronger) claim. The paper should run SAMIX's selection on the same reference pool to show selection differs from resampling.

**C4.2** *MAUP differentiation is contradicted by your own M1 result.* You argue MAUP does uncertainty-aware prompt quantity and you do cost–quality budget. But your M1 result (all policies collapse to k=6) could equally be read as "MAUP-style dynamic quantity doesn't help under this pipeline." If MAUP's positive result is dataset/pipeline-specific, your negative result doesn't differentiate — it generalizes a boundary. The paper should either run MAUP or reframe M1 as a confirmation of MAUP's boundary, not a differentiator.

**C4.3** *ProSAM differentiation is unverifiable.* "ProSAM models prompts; M2 models failure" is a clean conceptual split, but without ProSAM's code (Tier 3), you cannot show ProSAM wouldn't also reduce CFR if applied. The paper's claim that M2 is "orthogonal" to ProSAM is an assertion about an unrun method.

**C4.4** *The dual-model is methodologically lightweight.* Two logistic regressions on four hand-crafted features is a 2010-era selective prediction design. The "novelty" is the anti-correlation finding, but the anti-correlation is a property of the data/pipeline, not a method contribution. If the contribution is the finding, the method should be the simplest possible — which it is — but then the paper is a finding paper, not a method paper, and should be positioned as such.

**C4.5** *No head-to-head with any external method.* This is the most serious issue. The publication readiness checklist (condition 3) explicitly allows "OR documented justification + strong internal baselines," but the internal baselines (B1–B7) are all variants of your own pipeline. A reviewer cannot judge whether the instability phenomenon is specific to your DINOv2-L + SAM-B + point-prompt design or general to reference-guided SAM segmentation. GF-SAM uses graph grouping; MAUP uses multi-center uncertainty — these are different prompt constructions that might show different instability profiles.

**C4.6** *The "frozen backbones" argument cuts both ways.* You claim novelty in not fine-tuning SAM, but ProSAM, MAUP, GF-SAM are also training-free or use frozen SAM. This is not a differentiator; it is the field norm.

### Questions

Q4.1 What is the minimal experiment with GF-SAM or MAUP that would be feasible within the 195 GPU-hour budget?
Q4.2 If SAMIX selection were run on your IDRiD reference pool, would it also hurt Dice?
Q4.3 Is the anti-correlation a property of point-prompt SAM specifically, or would it appear with mask-prompt SAM (e.g., GF-SAM's mask-conditioned prompting)?

### Scores
Soundness: 5/10 | Presentation: 6/10 | Contribution: 4/10

### Recommendation: Reject (insufficient empirical differentiation; resubmit with ≥1 external method)

---

## Reviewer 5 (Medical-Only Data / Generalizability)

### Reading

All three datasets are medical histology/retinal. SAM is a general-purpose segmenter. The paper's claims about "reference-induced instability" are drawn entirely from a narrow domain and may not generalize.

### Criticisms

**C5.1** *Medical-only evaluation.* MoNuSAC (nuclear), MoNuSeg (nuclear), IDRiD (retinal lesions) are all medical, all 2D, all H&E/retinal. The instability phenomenon could be an artifact of (a) small medical targets, (b) high class imbalance, (c) DINOv2's weaker matching on medical textures, or (d) SAM-B's known weakness on medical imagery. Without a non-medical dataset, the paper cannot attribute the instability to the reference paradigm rather than to the domain.

**C5.2** *The IDRiD failure may be domain-specific, not pipeline-specific.* IDRiD lesions (microaneurysms, hemorrhages) are tiny, sparse, and low-contrast. The CFR on IDRiD is ~0.89–1.00 — the segmenter essentially fails on everything. A "consensus hurts" finding on a dataset where the base Dice is 0.09 is not informative about consensus; it is informative about IDRiD's difficulty. The paper should control for base Dice when comparing datasets.

**C5.3** *No natural-image, no remote-sensing, no RGB-D validation.* The directive (§21) requires ≥2–3 datasets for a benchmark; the paper has 3, but all medical. A reviewer could argue this is "3 datasets, 1 domain" and does not meet the spirit of the multi-dataset requirement.

**C5.4** *The dual-model selective predictor is validated only on medical data.* The 79.2% CFR reduction is on MoNuSAC holdout. Whether the four disagreement features generalize to natural images (where DINOv2 is stronger) is unknown. The paper claims "test-time-safe features" but does not test cross-domain safety.

**C5.5** *The claim registry prohibits "universal across datasets,"* which the paper respects — but then the contribution is bounded to medical reference-guided SAM segmentation, a narrow niche. The paper should explicitly position itself as a medical-imaging contribution, not a general SAM contribution.

### Questions

Q5.1 Would the instability phenomenon appear on a natural-image few-shot segmentation benchmark (e.g., PASCAL-5i, COCO-20i) with the same pipeline?
Q5.2 Is the IDRiD failure attributable to DINOv2 matching quality (matching_map_disagreement on IDRiD has |ρ|=0.1017, the weakest of all datasets)?
Q5.3 Does the catastrophic threshold (p10 Dice) generalize, or is it dataset-specific?

### Scores
Soundness: 6/10 | Presentation: 6/10 | Contribution: 5/10

### Recommendation: Weak Reject (generalizability unverified; resubmit with ≥1 non-medical dataset)

---

## Meta-Review

Consensus on weaknesses: (1) no external method reproduced (R2, R4); (2) small samples without CIs / correction (R3); (3) medical-only (R5); (4) anti-correlation under-tested (R1, R2, R3). Consensus on strengths: pre-registered holdout protocol, honest negatives, deployable M2.

Aggregate recommendation: **Weak Reject / Borderline** (1 Weak Accept, 2 Weak Reject, 2 Reject). The paper is not acceptable in its current form but is recoverable. The single most impactful addition is reproducing ≥1 external method (GF-SAM or MAUP) under the unified protocol; the second is adding CIs and a permutation test for the anti-correlation; the third is adding ≥1 non-medical dataset or explicitly repositioning as a medical-imaging contribution.

Fatal issues: none (all criticisms are addressable with additional experiments or repositioning).
Major issues: C2.5, C3.1, C3.4, C4.5, C5.1 (plus their correlates C4.1, C4.4, C5.3).
Minor issues: C1.1–C1.4, C2.1, C2.2, C2.6, C3.5, C3.6, C3.7, C4.6, C5.5.

The detailed per-criticism rebuttal, with current evidence / needed experiments / unresolvable limitations / impact, is in `phaseT_rebuttal_plan.md`.
