# V4 Reconstruction Decision

**Generated**: 2026-07-28 12:16:57
**Canonical V4**: `paper/phaseU/full_draft.tex`

---

## Decision

### `VV4_PARTIALLY_RECONSTRUCTABLE`

V4 cannot be reconstructed as-is, but a substantial portion of its content —
particularly the RRE problem formulation, experimental setup, and key baseline
numbers — can serve as scaffolding for a new paper built around Prediction Consistency.

---

## Evidence Assessment

### What CAN be retained from V4

| Asset | Status | Rationale |
|-------|--------|-----------|
| Introduction / RRE formulation | **KEEP** | Strongest surviving contribution. Phase AB lists as key strength. |
| Protocol / reference-resampling | **KEEP** | K=6, S=5, 30 triples — reproducible from code. |
| Top-K baseline Dice = 0.446 | **REPRODUCED** | WP8-001: MATCH, Δ=0.00024 |
| Reference variation 0.82→0.02 | **REPRODUCED** | WP8-002: MATCH_QUALITATIVE |
| Catastrophic prevalence 14% | **REPRODUCED** | WP8-003: MATCH, 14.25% |
| Related Work | **KEEP** | Literature review is thorough and well-organized. |
| 5.3 Failure Regime Characterization | **KEEP** | Mid-entropy non-monotonicity is novel and interesting. |
| 5.6 Feature Dissociation | **KEEP** | f1 vs f4 dissociation confirmed. |
| Appendix (metric coupling controls) | **KEEP** | Rigorous supplementary material. |

### What CANNOT be retained from V4

| Asset | Status | Rationale |
|-------|--------|-----------|
| RAMP full method (weighting + fusion + consensus) | **PROHIBITED** | Phase W: zero added value. |
| Anti-correlation as universal property | **PROHIBITED** | Significant only for SAM-B, not SAM2. Null model reproduces 94%. |
| RRN as main-text method | **APPENDIX_ONLY** | AUROC=0.573 < 0.65 threshold. |
| SAM-Med2D results | **REMOVED** | Checkpoint unloadable. Phase AB removed from main table. |
| Component ablation data | **REMOVED** | Phase V: all variants identical. Phase W: zero effect. |
| External baselines under 'identical conditions' | **PROHIBITED** | Backbone mismatch (SAM-H vs SAM-B). |
| Pascal/COCO cross-domain results | **APPENDIX_ONLY** | Single fold only. 4-fold needed. |

### What is PARTIALLY salvageable

| Asset | Action | Rationale |
|-------|--------|-----------|
| Geometric Hypotheses H1-H3 | **KEEP_WITH_MAJOR_QUALIFICATION** | As empirical association, not structural theorem. |
| Factorial analysis (2×2) | **KEEP_WITH_MINOR_EDIT** | Valid. Rename 'holdout' to 'Development Cohort'. |
| GF-SAM / MAUP comparison | **KEEP_WITH_MAJOR_QUALIFICATION** | Numbers verified. Must note backbone mismatch. |
| Deployment envelope | **KEEP_WITH_MAJOR_QUALIFICATION** | Validated 50-100%, not 5-60%. |
| Discussion | **KEEP_WITH_MAJOR_QUALIFICATION** | Rewrite to focus on Prediction Consistency. |

---

## Decision Matrix

| Decision Level | Definition | V4 Status |
|---------------|------------|-----------|
| VV4_RECONSTRUCTABLE_FROM_V4_SKELETON | V4 narrative + methods + results all valid | ❌ |
| **VV4_PARTIALLY_RECONSTRUCTABLE** | V4 skeleton valid, methods need replacement, some results removed | ✅ |
| VV4_SUBSTANTIALLY_UNSUPPORTED | Most claims unsupported or contradicted | ❌ |
| VV4_AUDIT_BLOCKED | Cannot determine due to missing evidence | ❌ |

---

## Salvage Ratio

- **Total sections**: 19
- **KEEP / KEEP_WITH_MINOR_EDIT**: 5 (26%)
- **KEEP_WITH_MAJOR_QUALIFICATION**: 6 (32%)
- **REWRITE_FROM_PHASE_AB**: 5 (26%)
- **REMOVE**: 1 (5%)
- **MOVE_TO_APPENDIX**: 2 (11%)
- **Effective salvage ratio**: 58%

---

## Phase AB Alignment

- **Phase AB Status**: PHASEAB_PAPER_CANDIDATE_READY (12/15 gates)
- **Core method**: Prediction Consistency (R_PC(q) = 1 - mean pairwise IoU)
- **Main pipeline**: DINOv2-L + SAM-B
- **Key results**: AURC=0.0158, CFR@80 reduction=88.6%, Dice@80=0.501
- **5/5 pipelines directional support**: All cross-pipeline transfers positive
- **No fresh confirmatory cohort**: Recommend workshop/benchmark venue

---

## Recommendation

1. **Use V4 narrative skeleton** (RRE problem, protocol, Related Work)
2. **Replace Method section** with Phase AB Prediction Consistency + coverage-based abstention
3. **Replace Results section** with Phase AB main results
4. **Remove** anti-correlation universal claim, RAMP weighting/fusion, SAM-Med2D
5. **Move** RRN and Pascal/COCO to Appendix
6. **Rebuild** under new title: 'When References Disagree: Prediction Consistency for Selective Visual Prompting'
7. **Target venue**: Workshop / Benchmark track

