# V4 High-Risk Claim Audit Report
**Generated**: 2026-07-28 12:04:25
**Canonical V4**: `paper/phaseU/full_draft.tex`

---

## HRA-01: "Four segmenters, two matchers"

**V4 Text**: Abstract: 'four segmenters and two matchers'

**Finding**: **PARTIALLY SUPPORTED**. V4 actually ran 4 segmenters (SAM-B, SAM-L, SAM2-T, SAM2-S) and 2 matchers (DINOv2-L, DINOv2-B). Phase U WP1-WP2 confirms execution. However, the claim 'six segmenters, four matchers' in earlier versions was PROHIBITED (Phase W Claim Registry).

**Verdict**: PARTIALLY_SUPPORTED

**Evidence**: Phase U WP1: SAM-B, SAM-L, SAM2-T, SAM2-S (4 segmenters). Phase U WP2: DINOv2-L, DINOv2-B (2 matchers). run_phaseU_wp1_segmenter_audit.py, run_phaseU_wp2_matcher_audit.py

---

## HRA-02: Anti-correlation numbers (r = -0.43 to -0.71)

**V4 Text**: Abstract r=-0.49; Section 3 various r values; Figures

**Finding**: **PARTIALLY SUPPORTED — SIGNIFICANTLY OVERSTATED**. Phase W WP7 anti-correlation audit (run_phaseW_wp7_anticorrelation.py) found: L+SAM-B r=-0.512 (significant), B+SAM-B r=-0.549 (significant), L+SAM-L r=-0.221 (p=0.049, fails Holm), L+SAM2-T r=-0.111 (p=0.325, NS), L+SAM2-S r=-0.179 (p=0.113, NS). The anti-correlation is significant ONLY for SAM-B pipelines, not SAM2. Phase V Null 1 (random beta) reproduces 94% of correlation (Δr=-0.036). Phase W null model produces r=0.0, confirming genuine residual for SAM-B. The claim of 'universal' anti-correlation across all pipelines is PROHIBITED.

**Verdict**: PARTIALLY_SUPPORTED

**Evidence**: Phase W: run_phaseW_wp7_anticorrelation.py, Phase V: run_phaseV_anticorrelation_controls.py, anticorrelation_residual.json

---

## HRA-03: Three Lemmas/Hypotheses (H1-H3)

**V4 Text**: Section 3: Geometric Hypotheses H1-H3

**Finding**: **SUPPORTED (as hypotheses, not theorems)**. Phase V Gate V1 (Theory Repair) confirmed: V4's 'Lemmas' and 'Theorem' language was replaced with 'Geometric Hypotheses (H1-H3)' + 'Empirical Mechanism Analysis' + 'Testable Predictions'. No formal theorem language remains in v4 (the current version already uses hypothesis language). The hypotheses are empirically testable propositions, not formal proofs.

**Verdict**: SUPPORTED

**Evidence**: Phase V: Gate_V1_Theory_Repair = PASS. V4 uses 'Geometric Hypotheses' not 'Theorems'. theory_verification_results.json

---

## HRA-04: RAMP components (weighting, fusion, prompt adaptation)

**V4 Text**: Section 4.4: three components; Section 5.5.1: component ablation

**Finding**: **PROHIBITED — COMPONENTS HAVE ZERO EFFECT**. Phase W WP1 component ablation (run_phaseW_experimental_closure.py) conclusively found: weighting has zero effect in all pipelines, fusion has tiny negative effect, Full RAMP does not improve upon baseline mean consensus. Gate: PHASEW_FULL_RAMP_NO_ADDED_VALUE. The component ablation data in V4 is unreliable (Phase V notes all variants produced identical Dice). Phase AB confirms: Prediction Consistency (B4) is the only effective method.

**Verdict**: PROHIBITED

**Evidence**: Phase W: component_ablation_results.json, Gate PHASEW_FULL_RAMP_NO_ADDED_VALUE. Phase AB: prediction consistency is sole effective method.

---

## HRA-05: "Training-free"

**V4 Text**: V4 does NOT explicitly claim training-free (uses 'supervised risk estimator' for RAMP). However, earlier versions did.

**Finding**: **PROHIBITED (for earlier versions) / NOT CLAIMED (in V4)**. V4 correctly describes RAMP as having a 'supervised risk estimator' (gradient-boosted trees). However, Phase W Claim Registry lists 'training-free' as PROHIBITED. The GB trees are trained on labeled multi-reference data. Matcher (DINOv2) is also trained. Phase AB confirms this is prohibited.

**Verdict**: V4_AVOIDED / PROHIBITED

**Evidence**: Phase W: Claim #4 PROHIBITED. Phase V: 'RAMP training-free' = NO. RAMP uses gradient-boosted trees.

---

## HRA-06: "Calibration-free"

**V4 Text**: Abstract: 'coverage-based ranking that does not require a fixed probability threshold'

**Finding**: **PROHIBITED (as 'calibration-free') / PARTIALLY SUPPORTED (as 'no fixed threshold')**. Phase W: IDRID ECE=0.477 confirms severe miscalibration. Phase V: 'calibration-free' = PROHIBITED. The claim 'no fixed probability threshold' is true, but 'calibration-free' is false. Phase AB: prediction consistency does not output probabilities, so calibration is not applicable.

**Verdict**: PARTIALLY_SUPPORTED

**Evidence**: Phase W: IDRID ECE=0.477. Phase V: 'calibration-free' = PROHIBITED.

---

## HRA-07: CFR reduction 56-84% (actual 37.5-84.4%)

**V4 Text**: Abstract: '37.5-84.4%'

**Finding**: **SUPPORTED for V4 (corrected range)**. V4 uses the correct 37.5-84.4% range. Earlier versions used 56-84% which was incorrect. Phase V corrected this. Phase W confirms the range but notes: RAMP's CFR reduction is achievable with simpler methods (B4 Prediction Consistency). The CFR reduction is not unique to RAMP.

**Verdict**: SUPPORTED (V4 corrected)

**Evidence**: Phase V: corrected from 56-84% to 37.5-84.4%. Phase W: simple baselines achieve similar CFR reduction.

---

## HRA-08: RRN (AUROC=0.612, Dice@Top-3=0.525)

**V4 Text**: Abstract and Section 5.3

**Finding**: **PARTIALLY SUPPORTED — APPENDIX ONLY**. Phase V Gate V6: RRN AUROC=0.612 < 0.65 threshold, Dice@Top-3 advantage +0.009 < +0.010 threshold. Only 12 test queries, 0 catastrophic cases. Phase W grouped 5-fold CV: mean AUROC=0.573, below 0.65. Phase AB: RRN is appendix-only. The data split (70/15/15 with 12 test queries) is too small for reliable evaluation.

**Verdict**: PARTIALLY_SUPPORTED

**Evidence**: Phase V: Gate_V6_RRN_Evaluation = APPENDIX_ONLY. Phase W: RRN grouped CV AUROC=0.573. Phase V: rrn_evaluation.md

---

## HRA-09: External Baseline table (GF-SAM 0.388, MAUP 0.279, etc.)

**V4 Text**: Table 6 (comparison)

**Finding**: **PARTIALLY SUPPORTED — BACKBONE MISMATCH**. GF-SAM and MAUP use SAM-ViT-H (stronger than RAMP's SAM-B). CycleSAM uses SAM-B (fair comparison). H-SAM uses SAM-L. SAM-Med2D: checkpoint cannot be loaded, REMOVED from main table per Phase AB. Phase W: external baselines pending backbone-stratified comparison. Phase V: Table 4 clearly marks backbone differences. The claim of 'identical conditions' is PROHIBITED.

**Verdict**: PARTIALLY_SUPPORTED

**Evidence**: Phase W: External Baseline Conclusion PENDING. Phase V: baseline_audit/comparator_registry.md. Phase AB: SAM-Med2D REMOVED.

---

## HRA-10: Pascal-5i / COCO-20i predictions

**V4 Text**: Section 5.8: Pascal-5i r=-0.51, COCO-20i r=-0.47

**Finding**: **PARTIALLY SUPPORTED — SINGLE FOLD ONLY**. Phase W WP9: Pascal/COCO only fold-0 available, 4-fold needed. Phase V: Pascal-5i/COCO-20i only fold-0. Phase AB: no confirmatory evidence. The cross-domain claim is based on incomplete evaluation. The numbers (r=-0.51, r=-0.47) are from fold-0 only.

**Verdict**: PARTIALLY_SUPPORTED

**Evidence**: Phase W: WP9 Pascal/COCO 4-Fold = PENDING. Phase V: 'Pascal-5i/COCO-20i only fold-0'.

---

## HRA-11: Deployment envelope (5-60%)

**V4 Text**: Section 5.7: 'effective when catastrophic prevalence falls in [5%, 60%]'

**Finding**: **PARTIALLY SUPPORTED**. Phase W WP3: coverage curves stable across 50-100%. The envelope is derived from 3 medical + 2 natural image datasets. Phase W notes: validated at 50-100% coverage, not 5-60%. The lower bound (5%) is not well-tested. The criterion is useful but requires more validation on diverse modalities.

**Verdict**: PARTIALLY_SUPPORTED

**Evidence**: Phase W: WP3 Risk-Coverage Curves, coverage curves stable across 50-100%.

---

## HRA-12: "First RRE problem / first benchmark"

**V4 Text**: Abstract: 'first reference selection benchmark'; Section 2.3: 'first to define and benchmark'

**Finding**: **SUPPORTED (with qualification)**. No prior work explicitly defines RRE as a standalone problem. The literature review is thorough. However, the claim of 'first' should be qualified as 'first to formalize' rather than 'first to notice'. Reference-based few-shot learning has explored related ideas (Vinyals 2016, Snell 2017). Phase V and Phase W do not challenge this novelty claim. Phase AB supports the problem formulation as a key strength.

**Verdict**: SUPPORTED

**Evidence**: Phase V: novelty claim retained. Phase AB: 'Clear problem formulation (RRE)' listed as key strength.

---

## Summary

- PARTIALLY_SUPPORTED: 7
- PROHIBITED: 1
- SUPPORTED: 2
- SUPPORTED (V4 corrected): 1
- V4_AVOIDED / PROHIBITED: 1

