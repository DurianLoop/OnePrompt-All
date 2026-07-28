# New Paper Contributions

**Generated**: 2026-07-28 12:16:57

---

## Threefold Contribution

### C1: RRE Problem Definition

- **What**: Formalize Reference Reliability Estimation (RRE) as a standalone problem
  in visual prompting pipelines.
- **Why novel**: No prior work explicitly defines whether a reference is reliable
  for segmentation. Reference-based few-shot learning focuses on prompt quality,
  not reference reliability.
- **Evidence**: Phase AB lists 'Clear problem formulation (RRE)' as key strength.
  V4 hallucination audit: Class A (VERIFIED).
- **V4 source**: Introduction, Abstract, Related Work (salvage: KEEP)

### C2: Prediction Consistency Risk Score

- **What**: R_PC(q) = 1 - mean pairwise IoU across K references — a simple,
  training-free risk estimator that captures reference-induced uncertainty.
- **Why novel**: Unlike RAMP (supervised, zero effect) or RRN (learned, AUROC=0.573),
  Prediction Consistency requires no training data and achieves AURC=0.0158.
- **Evidence**: Phase W: B4 Prediction Consistency is strongest baseline. Phase AB:
  5/5 pipelines directional support. AURC=0.0158, CFR@80 reduction=88.6%.
- **Phase AB source**: WP3, WP4, main results

### C3: 5-Pipeline Cross-Architecture Evaluation

- **What**: Systematic evaluation of Prediction Consistency across 5 visual
  prompting pipelines (DINOv2-L/B × SAM-B/L/SAM2-T/SAM2-S).
- **Why novel**: Demonstrates that the risk signal is robust to matcher and
  segmenter architecture choices. 5/5 pipelines show directional support.
- **Evidence**: Phase AB cross-pipeline: all 5 pipelines positive. Cross-dataset
  boundary analysis on MoNuSeg and IDRID.
- **Phase AB source**: WP3 cross-pipeline, WP3b cross-dataset

---

## Contribution Mapping: V4 → New Paper

| V4 Claim | V4 Status | New Paper Status |
|----------|-----------|-----------------|
| RRE problem formulation | SUPPORTED | C1 (retained) |
| Anti-correlation (universal) | PROHIBITED | Removed from main text |
| Geometric Hypotheses H1-H3 | SUPPORTED (as hypotheses) | Appendix |
| RAMP weighting/fusion/consensus | PROHIBITED | Removed |
| Prediction Consistency | N/A (new) | C2 (main contribution) |
| RRN (AUROC=0.612) | APPENDIX_ONLY | Appendix |
| External baselines | PARTIALLY_SUPPORTED | C3 (with backbone note) |
| SAM-Med2D | REMOVED | Removed |
| Pascal/COCO cross-domain | PARTIALLY_SUPPORTED | Appendix |
| Component ablation | PROHIBITED | Removed |
| Deployment envelope | PARTIALLY_SUPPORTED | C3 (50-100% range) |

---

## Prohibited Words

The following terms must NOT appear in the new paper text:

- `universal` (anti-correlation is SAM-B specific)
- `holdout-confirmed` (no confirmatory cohort)
- `SOTA` / `state-of-the-art` (no proper comparison)
- `training-free` (DINOv2 is trained)
- `calibration-free` (IDRID ECE=0.477)
- `identical conditions` (backbone mismatch)
- `Theorem` / `Lemma` (use 'Geometric Hypotheses')

