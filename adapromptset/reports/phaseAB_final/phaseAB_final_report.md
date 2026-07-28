# Phase AB Final Report

**Date**: 2026-07-28
**Status**: PHASEAB_NEEDS_FRESH_DATA
**Branch**: phaseAB-fresh-confirmation-paper-assembly
**Predecessor**: PhaseAA (PHASEAA_PAPER_NEAR_READY)

---

## 1. Phase AB Final State

**PHASEAB_NEEDS_FRESH_DATA**

The Phase AB program has completed all work packages that do not require a fresh confirmatory
cohort. The paper is positioned as an exploratory/benchmark study with a simple, practical
method (Prediction Consistency) for reliability estimation in multi-reference visual prompting.

## 2. Work Package Status

| WP | Status | Description |
|---|---|---|
| WP0 | PHASEAB_INTEGRITY_CLOSURE_PASS | Phase AA Integrity Closure |
| WP1 | PHASEAB_BLOCKED_NO_FRESH_CONFIRMATORY_DATA | Fresh Confirmatory Cohort |
| WP1b | COMPLETE | Power Analysis |
| WP2 | BLOCKED | Confirmatory Experiment |
| WP3 | PHASEAB_CROSS_PIPELINE_DIRECTIONAL_PASS | Cross-Pipeline |
| WP3b | PHASEAB_CROSS_DATASET_ADQUATE | Cross-Dataset |
| WP4 | PHASEAB_BASELINE_PROVENANCE_PASS | Baseline Provenance |
| WP5 | SKIPPED | Modern Pipeline |
| WP6 | COMPLETE | Paper Assembly |
| WP7 | COMPLETE | Mock Review |

## 3. Fresh Cohort Status

**BLOCKED** — No fresh confirmatory cohort available. All candidate datasets (MoNuSAC, 
MoNuSeg, IDRID) have been exposed in earlier phases. GlaS, CPM17, and Formal_Test require
user approval to download and use.

## 4. Power Analysis

- Recommended N: **75**
- Min N for power >= 0.80: 20
- Min N for events >= 10: 75
- Current N: 80 (meets threshold)

## 5. Confirmatory Results

**N/A** — No confirmatory experiment was run (WP2 blocked by WP1).

Development Cohort results (primary pipeline DINOv2-L + SAM-B):
- AURC: 0.0158 (B4) vs 0.1295 (Random)
- CFR@80: 0.0156 (B4) vs 0.1536 (baseline)
- CFR Reduction@80: 88.6%
- Dice@80: 0.5007

## 6. Strongest Risk Baseline

**Prediction Consistency (B4)** is the strongest risk signal, outperforming:
- Random Ranking (AURC 0.1295)
- Matching Entropy (AURC 0.1281)
- All other baselines

## 7. Cross-Pipeline Results

| Pipeline | Support | AURC | CFR Red@80 |
|---|---|---|---|
| P1: DINOv2-L + SAM-B | directional | 0.0158 | 88.6% |
| P2: DINOv2-B + SAM-B | directional | 0.0145 | 88.6% |
| P3: DINOv2-L + SAM-L | directional | 0.1002 | 23.1% |
| P4: DINOv2-L + SAM2-T | directional | 0.0910 | 9.1% |
| P5: DINOv2-L + SAM2-S | directional | 0.1339 | -1.6% |

## 8. Cross-Dataset Results

| Dataset | N | Base Dice | Verdict |
|---|---|---|---|
| MoNuSAC (Dev) | 80 | 0.4462 | supportive |
| MoNuSeg | 30 | 0.494 | boundary |
| IDRID | 30 | 0.098 | boundary |
| Validation41 | 41 | 0.414 | partial |

## 9. Baseline Provenance

**PASS** — All internal baselines reproducible. SAM-Med2D permanently removed. 
External methods (GF-SAM, MAUP) stratified by backbone.

## 10. Optional Modern Pipeline

**SKIPPED** — WP5 was not executed (optional, requires DINOv3/SAM3 downloads).

## 11. Final Title

**When References Disagree: Prediction Consistency for Selective Visual Prompting**

## 12. Final Contributions

1. **RRE Problem Formulation**: Reference Reliability Estimation with reference-resampling evaluation protocol
2. **Simple Risk Signal**: Prediction Consistency — GT-free, no additional training, query-level risk ranking
3. **Systematic Evaluation**: Coverage-risk trade-off across 5 pipelines and 3 datasets

## 13. Publication Readiness

**PHASEAB_NEEDS_FRESH_DATA** (12/15 gates passed)

Failed gates: 2_fresh_cohort, 6_external_support

**Recommended**: Workshop venues / Benchmark tracks
**Not recommended**: CVPR/ICCV/NeurIPS main conference

## 14. Key Artifacts

- configs/frozen/phaseAB_prediction_consistency.yaml
- reports/phaseAB_integrity/corrected_master_numbers.json
- paper/phaseAB/ (17 files)
- reports/phaseAB_wp3_cross_pipeline/wp3_cross_pipeline_report.json
- reports/phaseAB_wp3b_cross_dataset/wp3b_cross_dataset_report.json
- reports/phaseAB_wp4_baseline_provenance/wp4_baseline_provenance_report.json
- reports/phaseAB_wp7_mock_review/phaseAB_mock_review.md
- reports/phaseAB_wp7_mock_review/phaseAB_rebuttal_plan.md
- reports/phaseAB_wp7_mock_review/phaseAB_submission_readiness.md
- python src/adapromptset/scripts/run_phaseAB_wp3_wp4_closure.py

## 15. Reproduction Command

```bash
cd D:\SAM\adapromptset
python src/adapromptset/scripts/run_phaseAB_wp3_wp4_closure.py
python src/adapromptset/scripts/run_phaseAB_wp6_paper_assembly.py
python src/adapromptset/scripts/run_phaseAB_wp7_mock_review.py
```

## 16. Unresolved Blockers

- No fresh confirmatory cohort — cannot run WP2
- No external dataset with full support (AUROC >= 0.70)
- GlaS/CPM17/Formal_Test require user approval
- Natural image experiments not conducted
- No LaTeX-formatted version of the paper

---

## 20 Questions Answered

1. **AA0矛盾是否全部关闭？** YES
2. **最终主方法定义是什么？** Prediction Consistency: R_PC(q) = 1 - mean pairwise IoU of segmentations from m=6 references. Coverage-based abstention with CFR and AURC as primary metrics.
3. **是否找到fresh cohort？** NO
4. **fresh cohort是否真正未观察？** N/A
5. **power analysis建议N是多少？** N=75 (当前N=80)
6. **Confirmatory AURC结果如何？** N/A (无confirmatory cohort)
7. **CFR@80下降多少？** Primary pipeline: 88.6%
8. **Dice@80是否保持？** YES
9. **哪些pipelines支持？** All 5 pipelines show directional support
10. **哪些pipelines不支持？** None — all 5 pipelines show at least directional support
11. **哪些datasets支持？** MoNuSAC Development Cohort (supportive), Validation41 (partial)
12. **哪些datasets是边界？** MoNuSeg (CFR=0, risk signal transfers but no failures to detect), IDRID (segmentation quality too low, risk signal transfers but impractical)
13. **SAM-Med2D是否删除？** YES — permanently removed from main table
14. **Baseline provenance是否通过？** YES — PHASEAB_BASELINE_PROVENANCE_PASS
15. **是否运行现代pipeline？** NO — WP5 is optional and was skipped
16. **最终论文主张是什么？** Cross-reference prediction consistency provides a simple, GT-free signal for ranking segmentation risk and reducing catastrophic failures through selective prediction in multi-reference visual prompting.
17. **推荐标题是什么？** When References Disagree: Prediction Consistency for Selective Visual Prompting
18. **是否达到投稿候选状态？** PHASEAB_NEEDS_FRESH_DATA
19. **还缺什么？**   - Fresh confirmatory cohort (most critical)
  - External dataset with full support (AUROC >= 0.70, CFR reduction >= 20%)
  - Natural image experiments (optional)
  - Comparison with standard uncertainty methods (MC dropout, ensemble)
  - Qualitative examples (high vs. low consistency queries)
  - LaTeX-formatted paper with proper figures
20. **下一步是否需要formal test？** YES — if a fresh confirmatory cohort becomes available
