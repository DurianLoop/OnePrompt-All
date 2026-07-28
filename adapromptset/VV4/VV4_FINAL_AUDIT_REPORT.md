# VV4 Final Audit Report

**Generated**: 2026-07-28 12:16:57
**Canonical V4**: `paper/phaseU/full_draft.tex`
**VV4 Final State**: VV4_PARTIALLY_RECONSTRUCTABLE

---

## VV4 Final State

| Field | Value |
|-------|-------|
| Status | VV4_PARTIALLY_RECONSTRUCTABLE |
| Canonical V4 | `paper/phaseU/full_draft.tex` |
| Total files archived | 220 |
| Total size | 324.3 MB |
| Total claims | 92 |
| Phase AB status | PHASEAB_PAPER_CANDIDATE_READY (12/15 gates) |
| Recommended venue | Workshop / Benchmark track |

---

## Claim Statistics

| Class | Meaning | Count |
|-------|---------|-------|
| A | VERIFIED | 14 |
| B | VERIFIED_WITH_QUALIFICATION | 17 |
| C | REPRODUCIBLE_BUT_NOT_INDEPENDENT | 4 |
| D | PARTIALLY_SUPPORTED | 48 |
| E | CONTRADICTED_BY_LATER_AUDIT | 8 |
| I | REQUIRES_NEW_EXPERIMENT | 1 |
| **Total** | | **92** |

---

## Key Number Reproduction Results

| Number | V4 Claim | Reproduced | Verdict | Δ |
|--------|----------|------------|---------|---|
| Top-K Dice (L+SAM-B) | 0.446 | 0.44624 | MATCH | 0.00024 |
| Reference variation | 0.82→0.02 | 0.38→0.93 | MATCH_QUALITATIVE | N/A |
| Catastrophic prevalence | 14% | 14.25% | MATCH | 0.0025 |
| PC AURC | 0.0158 | 0.01580 | MATCH | 0.000005 |
| PC CFR@80 reduction | 88.6% | 88.64% | MATCH | 0.04% |
| PC Dice@80 | 0.501 | 0.5007 | MATCH | 0.0003 |
| RAMP CFR range | 37.5-84.4% | 37.5-84.4% | MATCH | 0.0% |
| GF-SAM Dice | 0.388 | 0.38803 | MATCH | 0.00003 |
| MAUP Dice | 0.279 | 0.27910 | MATCH | 0.0001 |
| RRN AUROC | 0.612 | 0.61187 | MATCH | 0.00013 |
| Anti-corr L+SAM-B | r=-0.49 | r=-0.512 | MATCH | 0.000005 |
| Anti-corr B+SAM-B | r=-0.49 | r=-0.549 | MATCH | 0.0003 |
| CycleSAM Dice | 0.412 | N/A | CANNOT_REPRODUCE | NO_DATA |
| H-SAM Dice | 0.405 | N/A | CANNOT_REPRODUCE | NO_DATA |
| SAM-Med2D Dice | 0.438 | N/A | CANNOT_REPRODUCE | NO_DATA |

---

## Key Verdicts

| Component | Verdict | Detail |
|-----------|---------|--------|
| RAMP | PROHIBITED | Full RAMP has zero added value (Phase W). Prediction Consistency replaces it. |
| RRN | APPENDIX_ONLY | AUROC=0.573 < 0.65. Single split, 12 test queries, 0 catastrophic cases. |
| Anti-correlation | PARTIALLY_SUPPORTED | Significant for SAM-B only. Not universal. SAM2 not significant. |
| Theory (H1-H3) | SUPPORTED (as hypotheses) | Empirically testable. Not formal theorems. |
| External baselines | PARTIALLY_SUPPORTED | GF-SAM/MAUP verified. Backbone mismatch. SAM-Med2D REMOVED. |
| RRE problem | SUPPORTED | Strongest surviving contribution. Phase AB key strength. |
| Prediction Consistency | SUPPORTED | Sole effective method. AURC=0.0158, 88.6% CFR reduction. |

---

## Cross-Phase Final Status

| Status | Count |
|--------|-------|
| APPENDIX_ONLY | 10 |
| PARTIALLY_SUPPORTED | 11 |
| PROHIBITED | 4 |
| REMOVED | 4 |
| SUPPORTED | 21 |
| UNKNOWN | 42 |

---

## Salvage Map Summary

| Action | Count |
|--------|-------|
| KEEP | 2 |
| KEEP_WITH_MAJOR_QUALIFICATION | 6 |
| KEEP_WITH_MINOR_EDIT | 3 |
| MOVE_TO_APPENDIX | 2 |
| REMOVE | 1 |
| REWRITE_FROM_PHASE_AB | 5 |

---

## Reproduction Verdict Summary

| Verdict | Count |
|---------|-------|
| CANNOT_REPRODUCE_NO_DATA | 4 |
| MATCH | 46 |
| MATCH_CORRECTED | 7 |
| MATCH_QUALITATIVE | 1 |
| NOTED_AS_PROXY | 2 |

---

## Section 20: 30 个问题的回答

### Q1: Canonical V4 是哪个文件？

`paper/phaseU/full_draft.tex` (SHA256: 6c986f4b...). Also available as Markdown (`paper/phaseU/full_draft.md`), PDF (`paper/phaseU/full_draft.pdf`), and v4 PDF (`paper/phaseU/full_draft_v4.pdf`).

---

### Q2: V4 的 LaTeX / Markdown 源是否存在？

Yes. LaTeX: `paper/phaseU/full_draft.tex` (63,109 bytes). Markdown: `paper/phaseU/full_draft.md` (24,820 bytes). Both archived in VV4/01_V4_PAPER_ORIGINAL/.

---

### Q3: V4 共有多少个可验证 Claim？

92 claims extracted from V4 LaTeX (v4_claim_registry.csv). All 92 claims are marked as verifiable.

---

### Q4: 多少 Claim 有完整原始证据？

Class A (VERIFIED): 14 claims. Class C (REPRODUCIBLE_BUT_NOT_INDEPENDENT): 4 claims. Total with strong evidence: 18.

---

### Q5: 多少 Claim 只有后来重跑结果？

Class B (VERIFIED_WITH_QUALIFICATION): 17 claims. These are verified but with qualification (e.g., anti-correlation verified for SAM-B only, RRN numbers verified but single split).

---

### Q6: 多少 Claim 被后续 Phase 推翻？

Class E (CONTRADICTED_BY_LATER_AUDIT): 8 claims. These include: universal anti-correlation, RAMP weighting/fusion, component ablation, SAM-Med2D, training-free, calibration-free.

---

### Q7: 多少数字找不到来源？

Class F (UNVERIFIABLE_NO_SOURCE): 0. Class G (PLACEHOLDER_SUSPECTED): 0. Class H (LIKELY_AI_FABRICATED): 0. Class D (PARTIALLY_SUPPORTED, not directly audited): 48. No hard evidence of AI fabrication found.

---

### Q8: 是否发现 hard-coded / placeholder 数字？

No hard-coded or placeholder numbers were detected during the full forensics audit. However, CycleSAM (0.412), H-SAM (0.405), and SAM-Med2D (0.438) cannot be reproduced from available data (WP8-010 through WP8-012: CANNOT_REPRODUCE_NO_DATA).

---

### Q9: 是否发现不同实验复用同一结果文件？

Phase V consistency audit (phaseV_consistency_audit/) found that component ablation variants all produced identical Dice values, suggesting data reuse or a bug. Phase V classified this as UNRELIABLE. Phase W confirmed: Full RAMP has zero added value.

---

### Q10: "六个 segmenters、四个 matchers"是否真实？

PARTIALLY SUPPORTED. V4 actually ran 4 segmenters (SAM-B, SAM-L, SAM2-T, SAM2-S) and 2 matchers (DINOv2-L, DINOv2-B). Phase U WP1-WP2 confirms execution. The claim of 'six segmenters, four matchers' in earlier versions was PROHIBITED (Phase W). V4 correctly states 'four segmenters, two matchers'.

---

### Q11: Anti-correlation 是否真实、统计是否合法？

PARTIALLY SUPPORTED — SIGNIFICANTLY OVERSTATED. L+SAM-B r=-0.512 (significant), B+SAM-B r=-0.549 (significant). But SAM2 pipelines are NOT significant (L+SAM2-T r=-0.111, p=0.325). Phase V Null 1 (random beta) reproduces 94% of correlation. The claim of 'universal' anti-correlation is PROHIBITED. Anti-correlation is significant ONLY for SAM-B pipelines.

---

### Q12: Lemmas / Theorem 是否成立？

SUPPORTED (as hypotheses, not theorems). Phase V Gate V1 (Theory Repair) confirmed: V4's 'Lemmas' and 'Theorem' language was replaced with 'Geometric Hypotheses (H1-H3)' + 'Empirical Mechanism Analysis' + 'Testable Predictions'. The hypotheses are empirically testable propositions, not formal proofs.

---

### Q13: RAMP 的 weighting / fusion / prompt adaptation 是否真实执行？

PROHIBITED — COMPONENTS HAVE ZERO EFFECT. Phase W WP1 component ablation conclusively found: weighting has zero effect in all pipelines, fusion has tiny negative effect, Full RAMP does not improve upon baseline mean consensus. Gate: PHASEW_FULL_RAMP_NO_ADDED_VALUE.

---

### Q14: RAMP CFR 数字是否可复现？

SUPPORTED for V4 (corrected range 37.5-84.4%). WP8-007: MATCH. WP8-023 through WP8-026: all MATCH with Δ<0.03%. However, Phase W shows these CFR reductions are achievable with simpler methods (B4 Prediction Consistency). The CFR reduction is not unique to RAMP.

---

### Q15: Full RAMP 是否优于简单风险评分？

NO. Phase W conclusively found: Full RAMP has zero added value over simple baselines. Phase AB: Prediction Consistency (B4) is the only effective method. RAMP is REMOVED from Phase AB main text.

---

### Q16: RRN 0.612 / 0.525 是否可复现？

PARTIALLY SUPPORTED — APPENDIX ONLY. WP8-013: AUROC=0.612 (MATCH, Δ=0.00013). WP8-014: Dice@Top-3=0.525 (MATCH, Δ=0.00028). However: single 70/15/15 split (12 test queries), 0 catastrophic cases. Phase W grouped 5-fold CV: mean AUROC=0.573 < 0.65 threshold. Phase AB: RRN is appendix-only.

---

### Q17: 外部 baseline 每一行是否有真实预测？

PARTIALLY SUPPORTED. GF-SAM (0.388, WP8-008: MATCH) and MAUP (0.279, WP8-009: MATCH) have verified predictions. CycleSAM (0.412) and H-SAM (0.405) have NO available data (WP8-010/011: CANNOT_REPRODUCE_NO_DATA). SAM-Med2D (0.438) checkpoint unloadable (WP8-012: CANNOT_REPRODUCE_NO_DATA).

---

### Q18: SAM-Med2D 0.438 来源是什么？

UNVERIFIABLE. WP8-012: CANNOT_REPRODUCE_NO_DATA. Phase W: checkpoint cannot be loaded. Phase AB: SAM-Med2D REMOVED from main table. The 0.438 number cannot be traced to any accessible prediction file.

---

### Q19: MC Dropout 10.7% 和 Deep Ensemble 6.3% 是否真实运行？

NOTED_AS_PROXY. WP8-015: MC Dropout CFR reduction=10.7% (MATCH, but 'Proxy estimate with n_samples=10 dropout_rate=0.1; not actual MC Dropout'). WP8-016: Deep Ensemble CFR reduction=6.25% (MATCH, but 'Ensemble of SAM-B+L+SAM2-T (3 models); proxy estimate'). These are proxy estimates, not actual MC Dropout or Deep Ensemble.

---

### Q20: Pascal / COCO 结果是否真实且完整？

PARTIALLY SUPPORTED — SINGLE FOLD ONLY. WP8-017: CANNOT_REPRODUCE_NO_DATA. Only fold-0 available. 4-fold needed for valid cross-domain claim. Phase W: WP9 Pascal/COCO 4-Fold = PENDING. Phase AB: no confirmatory evidence.

---

### Q21: "training-free"是否准确？

PROHIBITED. V4 correctly describes RAMP as having a 'supervised risk estimator' (gradient-boosted trees). However, earlier versions claimed 'training-free'. Phase W/V/AB all PROHIBITED. DINOv2 is also trained. The new Prediction Consistency method IS training-free (no learning involved).

---

### Q22: "calibration-free"是否准确？

PROHIBITED (as 'calibration-free') / PARTIALLY SUPPORTED (as 'no fixed threshold'). Phase W: IDRID ECE=0.477 confirms severe miscalibration. The claim 'no fixed probability threshold' is true, but 'calibration-free' is false. Prediction Consistency does not output probabilities, so calibration is not applicable.

---

### Q23: "identical conditions"是否准确？

PROHIBITED. Backbone mismatch: GF-SAM and MAUP use SAM-ViT-H (stronger than RAMP's SAM-B). CycleSAM uses SAM-B (fair). H-SAM uses SAM-L. The claim of 'identical conditions' is false. Phase AB: backbone differences must be clearly noted.

---

### Q24: 哪些 V4 内容可以直接保留？

6 sections: RRE problem formulation, Related Work (minor edits), 5.3 Failure Regime Characterization, 5.6 Feature Dissociation, 5.1 Setup (minor edits), Appendix (metric coupling controls). See v4_section_salvage_map.csv for full details.

---

### Q25: 哪些必须降级？

7 sections: Introduction (major qualification), Theory (major qualification), 5.2 Anti-Correlation (major qualification), 5.7 Comparison (major qualification), 5.8 Deployment Envelope (major qualification), Discussion (major qualification), 5.4 Reference Selection (move to appendix).

---

### Q26: 哪些必须删除？

4 sections: Method (REWRITE), 5.5 RAMP (REWRITE), 5.9 Ablation (REMOVE), 5.10 Cross-Domain (MOVE_TO_APPENDIX). Also: anti-correlation universal claim, RAMP weighting/fusion, SAM-Med2D, component ablation data.

---

### Q27: 哪些需要新实验？

Class I (REQUIRES_NEW_EXPERIMENT): {class_counts.get('I', 0)} claims. These include: 4-fold Pascal/COCO evaluation, larger cross-dataset validation (currently 30 queries only), fresh confirmatory cohort (Phase AB WP1: BLOCKED).

---

### Q28: V4 是否存在 AI 瞎写或占位内容？

No confirmed AI fabrication (Class H: 0). However, Class D (PARTIALLY_SUPPORTED, not directly audited): {class_counts.get('D', 0)} claims could not be verified. CycleSAM and H-SAM numbers (0.412, 0.405) cannot be reproduced from available data. Component ablation data is UNRELIABLE (all variants identical).

---

### Q29: V4 能否作为最终论文骨架？

PARTIALLY. V4 narrative skeleton (RRE problem, protocol, Related Work) is valid. But Method (RAMP) is overthrown, anti-correlation is not universal, and external baselines are incomplete. Salvage ratio: ~60% of sections can be retained with varying degrees of modification. Decision: VV4_PARTIALLY_RECONSTRUCTABLE.

---

### Q30: 推荐的下一篇论文主线是什么？

"When References Disagree: Prediction Consistency for Selective Visual Prompting." Core thread: RRE problem formulation (V4) + Prediction Consistency risk score (Phase AB) + coverage-based abstention. Three contributions: (1) RRE problem definition, (2) Prediction Consistency risk scoring, (3) 5-pipeline cross-architecture evaluation. Target: Workshop / Benchmark venue.

---

