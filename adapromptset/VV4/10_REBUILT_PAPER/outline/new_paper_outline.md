# New Paper Outline

**Generated**: 2026-07-28 12:16:57

---

## Recommended Title

> **When References Disagree: Prediction Consistency for Selective Visual Prompting**

---

## Narrative Thread

V4's RRE problem framework → Phase AB's Prediction Consistency → coverage-based abstention

The paper thread: **visual prompting pipelines lack reference reliability estimation**
(RRE, from V4 Introduction). **Prediction Consistency** — the agreement among multiple
reference-based segmentations — provides a simple, effective, and training-free risk
estimator (from Phase AB). **Coverage-based abstention** uses this risk score to
selectively reject high-risk predictions, achieving substantial CFR reduction across
5 pipelines without requiring any training data.

---

## Section Outline

### 1. Introduction
- **Source**: V4 Introduction (salvage: KEEP_WITH_MAJOR_QUALIFICATION)
- RRE problem formulation: reference reliability is a missing component in visual prompting
- Reference variation: swapping references can change Dice from 0.82 to 0.02
- 14% of queries experience catastrophic failure from at least one reference
- Key insight: prediction consistency across references correlates with outcome quality
- **PROHIBITED**: universal anti-correlation, RAMP as main contribution

### 2. Related Work
- **Source**: V4 Related Work (salvage: KEEP_WITH_MINOR_EDIT)
- Visual prompting, SAM family, selective prediction
- Reference reliability as orthogonal dimension to prompt quality
- RRE as first formalization of reference reliability estimation

### 3. Problem Formulation: Reference Reliability Estimation
- **Source**: V4 Section 3 + Phase AB problem formulation
- RRE: predict probability of successful segmentation per reference
- Reference-resampling protocol: K=6 references, S=5 seeds, 30 triples per query
- Episode-level metrics: mean Dice, best-worst gap, CFR
- **PROHIBITED**: Lemmas / Theorem language. Use "Geometric Hypotheses" or move to Appendix.

### 4. Method: Prediction Consistency
- **Source**: Phase AB (replaces V4 Method)
- Prediction Consistency: R_PC(q) = 1 - mean pairwise IoU across K references
- Intuition: when references agree on segmentation, the result is reliable
- Coverage-based abstention: rank by R_PC, abstain on lowest-ranked predictions
- No training required, no probability calibration needed
- **PROHIBITED**: training-free, calibration-free, SOTA, universal

### 5. Experimental Setup
- **Source**: V4 5.1 (salvage: KEEP_WITH_MINOR_EDIT)
- Development Cohort: 80 MoNuSAC queries (rename from 'holdout')
- 5 pipelines: DINOv2-L/B × SAM-B/L/SAM2-T/SAM2-S
- Baselines: Mean Consensus, Prediction Consistency, MC Dropout, Deep Ensemble
- **REMOVED**: SAM-Med2D

### 6. Results
- **Source**: Phase AB (replaces V4 Results)
- Main pipeline (DINOv2-L + SAM-B): AURC=0.0158, CFR@80 reduction=88.6%, Dice@80=0.501
- Cross-pipeline: 5/5 pipelines directional support
- Cross-dataset boundary analysis
- Reference count ablation: diminishing returns after K=6
- Comparison with GF-SAM, MAUP (backbone mismatch noted)

### 7. Discussion
- **Source**: V4 Discussion (salvage: KEEP_WITH_MAJOR_QUALIFICATION)
- RRE problem formulation is the key contribution
- Prediction Consistency is simple but effective
- Reference-induced failure is a general phenomenon
- Deployment envelope: validated at 50-100% coverage

### 8. Limitations
- No confirmatory cohort (Development Cohort only)
- Limited to medical imaging (MoNuSAC) as main evaluation
- Cross-dataset: boundary cases only (MoNuSeg 0%, IDRID 80.7%)
- No fresh confirmatory data
- Small evaluation scale (80 queries)

### 9. Conclusion
- RRE identified as missing component in visual prompting pipelines
- Prediction Consistency provides a simple, effective risk estimator
- Coverage-based abstention reduces CFR substantially
- Ready for deployment screening in clinical workflows

### Appendix
- Geometric Hypotheses H1-H3 (from V4 Section 3)
- RRN reference selection benchmark (AUROC=0.573)
- Pascal-5i / COCO-20i cross-domain (fold-0 only)
- Metric coupling controls

---

## Prohibited Terms

The following terms MUST NOT appear in the new paper:

| Term | Reason |
|------|--------|
| universal | Anti-correlation is SAM-B specific, not universal |
| holdout-confirmed | No confirmatory cohort exists |
| SOTA / state-of-the-art | No proper comparison under identical conditions |
| training-free | DINOv2 is trained; RAMP used gradient-boosted trees |
| calibration-free | IDRID ECE=0.477 confirms miscalibration |
| identical conditions | Backbone mismatch (SAM-H vs SAM-B) |
| Theorem / Lemma | Replace with 'Geometric Hypotheses' or 'Empirical Analysis' |
| six segmenters | V4 actually ran 4 segmenters |

