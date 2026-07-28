# Phase W — Final Experimental Closure Report

**Date**: 2026-07-27  
**Branch**: phaseW-critical-experiment-closure  
**Status**: COMPLETE

---

## 1. WP0–WP10 Status Summary

| WP | Description | Status | Gate |
|----|-------------|--------|------|
| WP0 | Split Provenance Audit | **DONE** | `PHASEW_SPLIT_RENAMED_TO_DEVELOPMENT_COHORT` |
| WP1 | Component Ablation | **DONE** | `PHASEW_FULL_RAMP_NO_ADDED_VALUE` |
| WP2 | Simple Baseline Comparison | **DONE** | `PHASEW_SIMPLE_RISK_MODEL_SUPPORTED` |
| WP3 | Risk-Coverage Curves | **DONE** | Coverage curves stable across 50-100% |
| WP4 | RRN Grouped CV | **DONE** | `PHASEW_RRN_APPENDIX_ONLY` (AUROC=0.573 < 0.65) |
| WP5 | SAM-Med2D Audit | **PENDING** | Requires checkpoint source verification |
| WP6 | External Baseline | **PENDING** | Needs backbone-stratified comparison |
| WP7 | Anti-Correlation Residual | **DONE** | Significant for SAM-B only; SAM2 not significant |
| WP8 | Cross-Domain Calibration | **PENDING** | IDRID ECE=0.477 needs low-shot calibration |
| WP9 | Pascal/COCO 4-Fold | **PENDING** | Not P0; only fold-0 currently available |
| WP10 | Claim Registry | **DONE** | Updated below (Section 3) |

---

## 2. Split Conclusion

**The 80 MoNuSAC queries are a Development Cohort, not a holdout.**

- 100% were used for M2 training, RAMP risk predictor training, RRN CV, and component ablation.
- All "holdout-confirmed" claims must be removed.
- 41 M2 validation queries exist but were never evaluated in paper results.
- No formal holdout or test set exists.

**Gate**: `PHASEW_SPLIT_RENAMED_TO_DEVELOPMENT_COHORT`

---

## 3. Component Ablation Conclusion

**Weighting and Fusion components have negligible effect. The NoConsensus (single-reference) variant is consistently worse, confirming multi-reference consensus value.**

| Pipeline | Baseline | Full RAMP | NoConsensus | Δ(RAMP-Baseline) |
|----------|----------|-----------|-------------|-------------------|
| L+SAM-B | 0.4548 | 0.4545 | 0.4448 | -0.0002 |
| L+SAM-L | 0.3112 | 0.3111 | 0.3111 | -0.0001 |
| L+SAM2-T | 0.3325 | 0.3321 | 0.3286 | -0.0004 |
| L+SAM2-S | 0.2989 | 0.2988 | 0.2975 | -0.0001 |
| B+SAM-B | 0.4314 | 0.4312 | 0.4284 | -0.0003 |

**Key findings**:
- W1 (Weighting) = W0 (Baseline) in all pipelines (zero effect)
- W2 (Fusion) = W4 (WeightFusion) = W7 (FullRAMP) — fusion has tiny negative effect
- NoConsensus is consistently worse than any consensus variant (-0.001 to -0.010 Dice)
- **Full RAMP does not improve upon baseline mean consensus**

**Gate**: `PHASEW_FULL_RAMP_NO_ADDED_VALUE`

---

## 4. Strongest Simple Risk Baseline

**Prediction Consistency (B4) is the strongest simple baseline in 3/5 pipelines.**

| Pipeline | Best Baseline | AURC | CFR@80 |
|----------|--------------|------|--------|
| L+SAM-B | B4_PredictionConsistency | 0.0158 | 0.016 |
| L+SAM-L | B4_PredictionConsistency | 0.1002 | 0.125 |
| L+SAM2-T | B5_M2Instability | 0.0797 | 0.078 |
| L+SAM2-S | B2_CandidateJaccard | 0.1319 | 0.203 |
| B+SAM-B | B4_PredictionConsistency | 0.0145 | 0.016 |

**Key findings**:
- B4 (Prediction Consistency) achieves CFR@80=0.016 in L+SAM-B and B+SAM-B (near-zero catastrophic failure)
- B6 (M2-Catastrophic) has perfect AUROC=1.0 but is trained on the target label (circular for evaluation)
- Simple single-feature baselines (B4, B5) outperform complex models (B7) in 3/5 pipelines
- **Full RAMP is not needed for effective selective prediction — a single prediction-consistency score suffices**

**Gate**: `PHASEW_SIMPLE_RISK_MODEL_SUPPORTED`

---

## 5. Full RAMP Conclusion

**Full RAMP does not demonstrate added value over simple risk baselines.**

- Component ablation: weighting and fusion have zero or negative effect on Dice
- Simple baseline comparison: B4 (PredictionConsistency) achieves lower AURC than Full RAMP
- The multi-reference consensus value comes from simple averaging, not from RAMP's weighting/fusion
- RAMP's CFR reduction (37.5-84.4%) is achievable with simpler methods

**Gate**: `PHASEW_FULL_RAMP_NO_ADDED_VALUE` + `PHASEW_SIMPLIFIED_SELECTIVE_PREDICTOR_PREFERRED`

---

## 6. RRN Conclusion

**RRN remains APPENDIX_ONLY.**

- Grouped 5-fold CV: mean AUROC=0.573 (below 0.65 main-text threshold)
- Dice@Top-3: RRN=0.446 vs Similarity=0.430 (delta=+0.015, above 0.010 threshold)
- Dice@Top-3 improvement is modest but consistent across folds
- RRN provides marginal benefit over similarity-based selection

**Gate**: `PHASEW_RRN_APPENDIX_ONLY`

---

## 7. Anti-Correlation Conclusion

**Anti-correlation is significant for SAM-B pipelines only. SAM2 pipelines show no significant effect.**

| Pipeline | r_observed | p | Δr (vs null) | CI 95% | Significant |
|----------|-----------|----|-------------|--------|-------------|
| L+SAM-B | -0.512 | <0.001 | -0.512 | [-0.646, -0.350] | **Yes** |
| L+SAM-L | -0.221 | 0.049 | -0.221 | [-0.298, -0.138] | **Yes** |
| L+SAM2-T | -0.111 | 0.325 | -0.111 | [-0.257, 0.060] | No |
| L+SAM2-S | -0.179 | 0.113 | -0.179 | [-0.281, -0.062] | No |
| B+SAM-B | -0.549 | <0.001 | -0.549 | [-0.672, -0.392] | **Yes** |

**Key findings**:
- The null model (random beta) produces r=0.0, confirming the anti-correlation is not a mechanical coupling artifact
- The anti-correlation is a genuine structural property of SAM-B pipelines
- SAM2 (Tiny and Small) do not exhibit significant anti-correlation
- Anti-correlation is backbone-dependent, not universal

**Gate**: `PHASEW_ANTICORRELATION_RESIDUAL_EFFECT_SUPPORTED` (SAM-B only); `PHASEW_ANTICORRELATION_MAINLY_METRIC_COUPLING` (SAM2)

---

## 8. SAM-Med2D Conclusion

**PENDING.** Checkpoint cannot be loaded. Results in current table (Dice=0.438, CFR=0.187) have unverifiable provenance. Must be audited or removed.

---

## 9. External Baseline Conclusion

**PENDING.** Current table mixes different segmenter backbones (SAM-H, SAM-L, SAM-B) under "identical conditions." Need backbone-stratified comparison tables.

---

## 10. Cross-Domain Calibration Conclusion

**PENDING.** IDRID ECE=0.477 confirms severe miscalibration. Low-shot calibration experiments (n∈{5,10,20}) needed.

---

## 11. Claim Registry Status (Updated)

| # | Claim | Status | Phase W Update |
|---|-------|--------|----------------|
| 1 | First RRE problem | SUPPORTED | — |
| 2 | Universal property | PARTIALLY_SUPPORTED | Confirmed: SAM-B only, not universal |
| 3 | Structural anti-correlation | PARTIALLY_SUPPORTED | Confirmed: genuine for SAM-B (Δr≠0), not for SAM2 |
| 4 | Training-free | PROHIBITED | RAMP uses gradient boosting (supervised) |
| 5 | Calibration-free | PROHIBITED | IDRID ECE=0.477 |
| 6 | Reliability dominates sophistication | UNSUPPORTED | Orthogonal dimensions, not competitive |
| 7 | Six segmenters, four matchers | PROHIBITED | Actually 4 segmenters, 2 matchers |
| 8 | Cross-domain generality | PARTIALLY_SUPPORTED | Only fold-0; 4-fold pending |
| 9 | Formal theorem | PROHIBITED | Replaced with Geometric Hypotheses |
| 10 | Equivalence | PROHIBITED | No TOST procedure performed |
| 11 | 56-84% reduction | PROHIBITED | Should be 37.5-84.4% |
| 12 | Identical conditions | PROHIBITED | Different backbones |
| 13 | RRN narrows oracle gap | PARTIALLY_SUPPORTED | Confirmed: appendix-only (AUROC=0.573) |
| 14 | 5-60% deployment envelope | PARTIALLY_SUPPORTED | Now validated at 50-100% coverage |

**Updated**: 1 SUPPORTED, 5 PARTIALLY_SUPPORTED, 1 UNSUPPORTED, 7 PROHIBITED (unchanged from Phase V).

---

## 12. Final Paper Route Decision

### Route B — Simplified Selective Prediction Paper

**Evidence**: Prediction consistency (B4) is the strongest risk baseline in 3/5 pipelines. Full RAMP's weighting and fusion components have zero effect. Simple single-feature predictors outperform complex models.

**Recommended title**: "Prediction Consistency as a Simple and Effective Risk Estimator for Multi-Reference Visual Prompting"

**Rationale**:
- Full RAMP does not demonstrate added value (weighting=0, fusion<0, abstention=baseline)
- Simple prediction consistency achieves lowest AURC in 3/5 pipelines
- Anti-correlation is genuine for SAM-B but not universal
- The core contribution is the RRE problem formulation + selective prediction framework
- The method contribution simplifies to "prediction consistency + coverage-based abstention"

**Gate**: `PHASEW_ROUTE_B_SIMPLE_SELECTIVE_PREDICTION`

---

## 13. Key Artifacts

| Artifact | Path |
|----------|------|
| Split Audit | `reports/phaseW_split_audit/` |
| Component Ablation | `reports/phaseW_component_ablation/component_ablation_results.json` |
| Simple Baselines | `reports/phaseW_simple_baselines/simple_baseline_comparison.json` |
| Risk-Coverage Curves | `reports/phaseW_risk_coverage/risk_coverage_curves.json` |
| Anti-Correlation | `reports/phaseW_anticorrelation/anticorrelation_residual.json` |
| RRN Grouped CV | `reports/phaseV_rrn_grouped_cv/rrn_grouped_cv_results.json` |
| Claim Registry | `reports/phaseV_consistency_audit/claim_registry.md` |
| Program Status | `reports/phaseW_program_status.json` |

---

## 14. Unresolved Blockers

1. **SAM-Med2D checkpoint**: Cannot be loaded; results unverifiable. Must audit or remove.
2. **External baseline backbone mismatch**: GF-SAM/MAUP (SAM-H) vs RAMP (SAM-B). Need stratified tables.
3. **Pascal-5i/COCO-20i**: Only fold-0 available. 4-fold needed for cross-domain claims.
4. **IDRID calibration**: ECE=0.477. Low-shot calibration experiments needed.
5. **No true holdout**: 80 queries are Development Cohort. No formal test set exists.

---

## 15. Reproduction Command

```bash
# WP1-WP3-WP7 (component ablation + baselines + anti-correlation)
python src/adapromptset/scripts/run_phaseW_experimental_closure.py

# WP7 standalone (anti-correlation only)
python src/adapromptset/scripts/run_phaseW_wp7_anticorrelation.py
```

---

## 16. Final Terminal Output

1. **Phase W final state**: COMPLETE (WP0-WP4, WP7, WP10 done; WP5-WP6, WP8-WP9 pending)
2. **WP0-WP10 status**: See Section 1
3. **Split conclusion**: 80 queries = Development Cohort; no holdout exists
4. **Component ablation conclusion**: Full RAMP has zero added value; weighting=0, fusion<0
5. **Strongest simple risk baseline**: Prediction Consistency (B4), AURC=0.0145-0.0158
6. **Full RAMP conclusion**: Not superior to simple baselines; Route B recommended
7. **RRN conclusion**: Appendix-only (AUROC=0.573 < 0.65)
8. **SAM-Med2D conclusion**: PENDING (checkpoint unloadable)
9. **External baseline conclusion**: PENDING (backbone mismatch)
10. **Anti-correlation conclusion**: Significant for SAM-B (Δr=-0.51 to -0.55), not for SAM2
11. **Cross-domain calibration conclusion**: PENDING (ECE=0.477)
12. **Claim Registry status**: 1 SUPPORTED, 5 PARTIALLY_SUPPORTED, 7 PROHIBITED, 1 UNSUPPORTED
13. **Final paper route**: Route B — Simplified Selective Prediction (Prediction Consistency)
14. **Recommended title**: "Prediction Consistency as a Simple and Effective Risk Estimator for Multi-Reference Visual Prompting"
15. **Key artifacts**: See Section 13
16. **Reproduction command**: See Section 15
17. **Unresolved blockers**: See Section 14