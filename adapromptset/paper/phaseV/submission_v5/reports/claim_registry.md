# Claim Registry — Phase V Consistency Audit

**Audit Date:** 2026-07-27  
**Source:** `todo.md` Section 14 + `contradiction_log.md` + `corrected_master_numbers.json` + `query_level_statistics.json` + `final_validity_report.json`

---

## Status Legend

| Status | Meaning |
|--------|---------|
| **SUPPORTED** | Claim is fully supported by current evidence |
| **PARTIALLY_SUPPORTED** | Claim has some supporting evidence but requires qualification or caveats |
| **UNSUPPORTED** | Claim is not supported by current evidence; overstatement |
| **REQUIRES_NEW_EXPERIMENT** | Claim cannot be validated without additional experiments |
| **PROHIBITED** | Claim is factually incorrect; must be removed or fundamentally rewritten |

---

## Claim-by-Claim Audit

### Claim 1: "first RRE problem"

| Field | Detail |
|-------|--------|
| **Status** | ✅ SUPPORTED |
| **Current Evidence** | Literature review confirms no prior work formally defines or addresses the Reference Reliability Estimation (RRE) problem in the context of frozen visual prompting pipelines. |
| **What the paper can actually claim** | "We introduce the Reference Reliability Estimation (RRE) problem: predicting which reference images will lead to catastrophic or high-instability segmentations in visual prompting pipelines." |
| **Required Changes for v5** | None. Retain as-is. Ensure the claim is scoped to "visual prompting for medical/multi-reference segmentation" to avoid overreach. |

---

### Claim 2: "universal property"

| Field | Detail |
|-------|--------|
| **Status** | ⚠️ PARTIALLY_SUPPORTED |
| **Current Evidence** | Query-level Pearson r: L+SAM-B r=-0.512 (p<0.001), B+SAM-B r=-0.549 (p<0.001), L+SAM-L r=-0.221 (p=0.049, not Holm-significant), L+SAM2-T r=-0.111 (p=0.325, ns), L+SAM2-S r=-0.179 (p=0.113, ns). The anti-correlation holds strongly only for SAM-B pipelines. SAM2-T partial correlation is +0.129 (positive direction, ns). The direction is inconsistent across segmenters. |
| **What the paper can actually claim** | "The instability-CFR association is a reproducible empirical pattern for SAM-B based pipelines (r=-0.51 to -0.55, p<0.001). For SAM-L and SAM2 variants, the association is weaker or absent, suggesting segmenter architecture dependence." |
| **Required Changes for v5** | Replace "universal property" with "reproducible empirical pattern for SAM-B pipelines." Qualify that the effect is segmenter-dependent. Cite query-level statistics with Holm-corrected p-values. |

---

### Claim 3: "structural anti-correlation"

| Field | Detail |
|-------|--------|
| **Status** | ⚠️ PARTIALLY_SUPPORTED |
| **Current Evidence** | Partial correlation (controlling mean Dice): L+SAM-B partial r=-0.381 (p<0.001), B+SAM-B partial r=-0.500 (p<0.001). Both significant. However: L+SAM-L partial r=-0.064 (p=0.575, ns), L+SAM2-T partial r=+0.129 (p=0.255, ns). After Holm correction, only L+SAM-B and B+SAM-B remain significant. Null 1 (random beta) reproduces 94% of observed correlation (r=-0.583 vs observed -0.619), suggesting a substantial non-structural component. |
| **What the paper can actually claim** | "A genuine structural component exists for SAM-B pipelines (partial r=-0.38 to -0.50, p<0.001 after controlling for mean Dice). For other segmenters, the structural component is not detectable. The structural component coexists with a mechanical component (Null 1 reproduces ~94% of the correlation)." |
| **Required Changes for v5** | Scope to SAM-B pipelines only. Report partial r alongside raw Pearson r. Acknowledge the null experiment results. Avoid implying a purely structural causal mechanism. |

---

### Claim 4: "training-free"

| Field | Detail |
|-------|--------|
| **Status** | 🚫 PROHIBITED |
| **Current Evidence** | RAMP uses M2-3 gradient boosting, a supervised learning algorithm that trains a risk predictor on labeled multi-reference data. The matcher and segmenter are frozen, but the risk estimator is trained. |
| **What the paper can actually claim** | "Supervised risk estimator with frozen matcher and segmenter" or "RAMP employs a lightweight gradient-boosted risk predictor trained on multi-reference labels." |
| **Required Changes for v5** | Remove all instances of "training-free" from title, abstract, and body. Replace with "supervised risk estimator with frozen foundation models." The matcher (DINOv2) and segmenter (SAM/SAM2) are frozen; the risk predictor (gradient boosting) is trained. |

---

### Claim 5: "calibration-free"

| Field | Detail |
|-------|--------|
| **Status** | 🚫 PROHIBITED |
| **Current Evidence** | IDRID ECE = 0.477 (severe miscalibration). RAMP uses fixed thresholds (0.3/0.7) for risk category assignment. The selective prediction framework does not require calibrated probabilities, but it does not provide them either. |
| **What the paper can actually claim** | "Coverage-based ranking does not require a fixed probability threshold" or "RAMP operates on relative risk scores rather than calibrated probabilities." |
| **Required Changes for v5** | Remove all instances of "calibration-free." Replace with honest reporting: "RAMP uses rank-based risk scores (ECE=0.477 on IDRID); the selective prediction framework operates on coverage rather than probability thresholds." |

---

### Claim 6: "reference reliability dominates prompt sophistication"

| Field | Detail |
|-------|--------|
| **Status** | ❌ UNSUPPORTED |
| **Current Evidence** | The claim conflates an orthogonal dimension comparison. DINOv2-L and DINOv2-B produce different mean Dice levels (L+SAM-B: 0.446 vs B+SAM-B: 0.426), but the anti-correlation strength is similar. No experiment directly tests whether matcher quality dominates or is independent of prompt sophistication. The two are orthogonal dimensions, not competing ones. |
| **What the paper can actually claim** | "Reference reliability and prompt sophistication are orthogonal dimensions of visual prompting performance. Both contribute independently to segmentation quality." |
| **Required Changes for v5** | Replace "dominates" with "is an orthogonal dimension to." Frame as complementary rather than competitive. If a dominance claim is made, a factorial experiment (matcher quality × prompt sophistication) is required. |

---

### Claim 7: "six segmenters and four matchers"

| Field | Detail |
|-------|--------|
| **Status** | 🚫 PROHIBITED |
| **Current Evidence** | Actually evaluated: 4 segmenters (SAM-B, SAM-L, SAM2-Tiny, SAM2-Small) and 2 matchers (DINOv2-L, DINOv2-B). SAM-H and MedSAM are available in the codebase but no anti-correlation data was collected for them. |
| **What the paper can actually claim** | "Four segmenters (SAM-B, SAM-L, SAM2-Tiny, SAM2-Small) and two matchers (DINOv2-L, DINOv2-B)." |
| **Required Changes for v5** | Correct all counts throughout the paper. If SAM-H or MedSAM experiments are added, update counts accordingly. Otherwise, report exact evaluated models. |

---

### Claim 8: "cross-domain generality"

| Field | Detail |
|-------|--------|
| **Status** | ⚠️ PARTIALLY_SUPPORTED |
| **Current Evidence** | Only fold-0 of Pascal-5i and COCO-20i was evaluated. The standard evaluation protocol requires 4-fold cross-validation. Single-fold results are insufficient to establish generality. |
| **What the paper can actually claim** | "Initial cross-domain evidence on fold-0 of Pascal-5i and COCO-20i" or "Preliminary results suggest cross-domain transferability." |
| **Required Changes for v5** | Either (a) complete 4-fold evaluation for Pascal-5i and COCO-20i, or (b) downgrade claim to "initial cross-domain evidence (fold-0 only)." The latter requires acknowledging the limitation explicitly. |

---

### Claim 9: "formal theorem"

| Field | Detail |
|-------|--------|
| **Status** | 🚫 PROHIBITED |
| **Current Evidence** | Three Lemmas and one Theorem were replaced with Geometric Hypotheses (H1-H3) + Empirical Mechanism Analysis + Testable Predictions. The original formalization did not constitute a valid mathematical proof. |
| **What the paper can actually claim** | "We propose three geometric hypotheses (H1-H3) that predict an anti-correlation between reference instability and segmentation quality. We provide empirical mechanism analysis and testable predictions derived from these hypotheses." |
| **Required Changes for v5** | Remove all references to "theorem" and "lemma." Replace with "Geometric Hypotheses" and "Empirical Mechanism Analysis." The theoretical contribution is hypothesis-driven, not theorem-proven. |

---

### Claim 10: "equivalence"

| Field | Detail |
|-------|--------|
| **Status** | 🚫 PROHIBITED |
| **Current Evidence** | p>0.05 does not imply equivalence. To claim equivalence, a Two One-Sided Test (TOST) procedure with a pre-specified equivalence margin is required. The current analysis only shows "failure to reject the null hypothesis of no difference." |
| **What the paper can actually claim** | "No statistically significant difference detected between DINOv2-L and DINOv2-B matchers (p>0.05)." |
| **Required Changes for v5** | Replace "equivalence" with "no statistically significant difference detected." If equivalence is genuinely desired, run TOST with a pre-specified margin (e.g., Δ=0.02 Dice). |

---

### Claim 11: "56-84% reduction"

| Field | Detail |
|-------|--------|
| **Status** | 🚫 PROHIBITED |
| **Current Evidence** | Full CFR reduction range: L+SAM-B=55.9%, L+SAM-L=84.4%, **L+SAM2-T=37.5%**, B+SAM-B=55.9%. The 37.5% value (worst case) was selectively omitted from the range. |
| **What the paper can actually claim** | "37.5-84.4% CFR reduction at 80% coverage" or report per-pipeline: "L+SAM-B: 55.9%, L+SAM-L: 84.4%, L+SAM2-T: 37.5%, B+SAM-B: 55.9%." |
| **Required Changes for v5** | Correct the range to 37.5-84.4%. Report all four pipeline results individually. Do not cherry-pick the best range. |

---

### Claim 12: "identical conditions"

| Field | Detail |
|-------|--------|
| **Status** | 🚫 PROHIBITED |
| **Current Evidence** | GF-SAM uses SAM-H, MAUP uses SAM-H, H-SAM uses SAM-L, while RAMP uses SAM-B. Different segmenter backbones have different baseline Dice levels, making direct comparison unfair. |
| **What the paper can actually claim** | "All methods share the same protocol (DINOv2 matcher, multi-reference visual prompting). Segmenter backbones differ: RAMP uses SAM-B, GF-SAM/MAUP use SAM-H, H-SAM uses SAM-L." |
| **Required Changes for v5** | Remove "identical conditions." Create separate comparison tables grouped by segmenter backbone. Clearly mark backbone differences. Report fair comparisons (same backbone) and cross-backbone comparisons separately. |

---

### Claim 13: "RRN narrows oracle gap"

| Field | Detail |
|-------|--------|
| **Status** | ⚠️ PARTIALLY_SUPPORTED |
| **Current Evidence** | AUROC=0.612 (<0.65 threshold for reliable prediction). Dice@Top-3=0.525 vs similarity=0.516, oracle=0.588 (advantage +0.009, below +0.010 threshold). Test set: only 12 queries with 0 catastrophic cases. No catastrophic prediction success to demonstrate. |
| **What the paper can actually claim** | "RRN shows modest improvement over similarity-based selection (Dice@Top-3: +0.009). However, AUROC=0.612 and limited test set (12 queries, 0 catastrophic) prevent strong conclusions." |
| **Required Changes for v5** | Move RRN to appendix only. Run grouped nested 5-fold CV for reliable evaluation. If results improve, consider returning to main text. Current evidence is insufficient for a main-text claim. |

---

### Claim 14: "5-60% deployment envelope"

| Field | Detail |
|-------|--------|
| **Status** | ⚠️ PARTIALLY_SUPPORTED |
| **Current Evidence** | The 5-60% deployment envelope is derived from theoretical analysis of the coverage-abstention trade-off. It has not been empirically validated across deployment scenarios. The 80% coverage point is the only empirically tested operating point. |
| **What the paper can actually claim** | "Theoretical analysis suggests a 5-60% deployment envelope based on the coverage-abstention trade-off. Empirical validation at 80% coverage confirms effectiveness within this range." |
| **Required Changes for v5** | Clarify that the envelope is theoretically derived, not empirically validated at all points. Either (a) add experiments at multiple coverage levels (50%, 60%, 70%, 80%, 90%) or (b) present as theoretical projection with 80% coverage as the single empirical validation point. |

---

## Summary Statistics

| Status | Count | Claims |
|--------|-------|--------|
| SUPPORTED | 1 | #1 (first RRE problem) |
| PARTIALLY_SUPPORTED | 5 | #2, #3, #8, #13, #14 |
| UNSUPPORTED | 1 | #6 (dominates claim) |
| PROHIBITED | 7 | #4, #5, #7, #9, #10, #11, #12 |
| REQUIRES_NEW_EXPERIMENT | 0 | — |

**V5 Minimum Viable Claims**: 1 fully supported + 5 partially supported (with qualification) = 6 claims that can be made in v5. The 7 prohibited claims must be removed or fundamentally rewritten.

**Gate Status**: `PHASEV_P0_NUMERICAL_CONSISTENCY_FAIL` — 7 of 14 claims are P0-CRITICAL prohibited items. All prohibited claims must be resolved before v5 submission.

---

## Cross-Reference: Contradiction Log Mapping

| Claim # | Contradiction Log # | Contradiction |
|---------|---------------------|---------------|
| #2, #3 | Contradiction 1 | Anti-correlation Pearson r range overstated |
| #7 | Contradiction 2 | Model counts inflated |
| #11 | Contradiction 3 | CFR reduction range omitted worst case |
| #4 | Contradiction 4 | "training-free" falsely claimed |
| #5 | Contradiction 5 | "calibration-free" falsely claimed |
| #10 | Contradiction 9 | Equivalence incorrectly claimed |
| #12 | Contradiction 11 | "identical conditions" misrepresented |
| #13 | Contradiction 12 | RRN test set insufficient |
| #8 | Contradiction 14 | Single fold only |
| #9 | (final_validity_report Q3) | Theorem replaced with hypotheses |
| #6 | (final_validity_report Q16) | "dominates" claim withdrawn |
| #14 | — | Theoretical analysis, not empirically validated |