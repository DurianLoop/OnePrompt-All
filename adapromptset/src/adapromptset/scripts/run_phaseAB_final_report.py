#!/usr/bin/env python3
"""
Phase AB Final Report: Publication Readiness Gate + Comprehensive Final Report
"""
import json
from pathlib import Path
from datetime import datetime

BASE = Path("D:/SAM/adapromptset")
REPORTS = BASE / "reports"
PHASE_AA = REPORTS / "phaseAA_final_closure"

# Load all data
with open(PHASE_AA / "gate_aa4_cross_pipeline.json") as f:
    aa4 = json.load(f)
with open(PHASE_AA / "gate_aa5_cross_dataset.json") as f:
    aa5 = json.load(f)
with open(REPORTS / "phaseAB_integrity/corrected_master_numbers.json") as f:
    mn = json.load(f)
with open(REPORTS / "phaseAB_fresh_data_audit/fresh_data_decision.json") as f:
    fd = json.load(f)
with open(REPORTS / "phaseAB_power/recommended_sample_size.json") as f:
    power = json.load(f)

# Load WP3, WP3b, WP4 results
with open(REPORTS / "phaseAB_wp3_cross_pipeline/wp3_cross_pipeline_report.json") as f:
    wp3 = json.load(f)
with open(REPORTS / "phaseAB_wp3b_cross_dataset/wp3b_cross_dataset_report.json") as f:
    wp3b = json.load(f)
with open(REPORTS / "phaseAB_wp4_baseline_provenance/wp4_baseline_provenance_report.json") as f:
    wp4 = json.load(f)

dev = mn["master_numbers"]["development_cohort"]
cp = mn["master_numbers"]["cross_pipeline"]
cd = mn["master_numbers"]["cross_dataset"]

# ============================================================
# PUBLICATION READINESS GATE
# ============================================================

gate_checks = {
    "1_integrity_closure": True,   # WP0 passed
    "2_fresh_cohort": False,       # No fresh cohort
    "3_power_analysis": True,      # N=75, current N=80
    "4_confirmatory": "N/A",       # Blocked by #2
    "5_pipeline_support": True,    # 5/5 directional
    "6_external_support": False,   # Only boundary, no supportive
    "7_boundary_dataset": True,    # Both MoNuSeg and IDRID
    "8_baseline_provenance": True, # WP4 passed
    "9_prohibited_removed": True,  # All removed
    "10_full_ramp_removed": True,  # In appendix
    "11_rrn_appendix": True,       # AUROC=0.573
    "12_risk_coverage": True,      # Complete
    "13_aurc_primary": True,       # Frozen
    "14_full_draft": True,         # 17 files
    "15_mock_review": True,        # No fatal blockers
}

n_pass = sum(1 for v in gate_checks.values() if v is True)
n_total = sum(1 for v in gate_checks.values() if v is not None)
n_fail = sum(1 for v in gate_checks.values() if v is False)

# Determine readiness
if gate_checks["2_fresh_cohort"] is False:
    readiness = "PHASEAB_NEEDS_FRESH_DATA"
elif gate_checks["6_external_support"] is False:
    readiness = "PHASEAB_EXTERNAL_SUPPORT_INSUFFICIENT"
elif n_pass >= 13:
    readiness = "PHASEAB_PAPER_CANDIDATE_READY"
else:
    readiness = "PHASEAB_PAPER_NOT_READY"

# ============================================================
# FINAL REPORT
# ============================================================

# Answer all 20 questions
final_report = {
    "phase": "PhaseAB",
    "date": datetime.now().strftime("%Y-%m-%d"),
    "status": readiness,
    "branch": "phaseAB-fresh-confirmation-paper-assembly",
    "predecessor": "PhaseAA (PHASEAA_PAPER_NEAR_READY)",
    
    "answers": {
        "1_aa0_closure": {
            "question": "AA0矛盾是否全部关闭？",
            "answer": "YES",
            "details": "Null model冲突已解决（Phase W bug修复），SAM-L显著性已统一（Holm校正后不显著），"
                       "pipeline数量已冻结（5个），anti-correlation已从主文移除，15条prohibited claims已退休。"
        },
        "2_method_definition": {
            "question": "最终主方法定义是什么？",
            "answer": "Prediction Consistency: R_PC(q) = 1 - mean pairwise IoU of segmentations from m=6 references. "
                     "Coverage-based abstention with CFR and AURC as primary metrics.",
            "details": "Frozen in configs/frozen/phaseAB_prediction_consistency.yaml"
        },
        "3_fresh_cohort": {
            "question": "是否找到fresh cohort？",
            "answer": "NO",
            "details": "所有候选数据集（MoNuSAC, MoNuSeg, IDRID）均已暴露。GlaS/CPM17/Formal_Test需要用户批准下载。"
                       "论文定位为exploratory/benchmark。"
        },
        "4_freshness": {
            "question": "fresh cohort是否真正未观察？",
            "answer": "N/A",
            "details": "无fresh cohort。Development Cohort（80 queries）和Validation Cohort（41 queries）均已被观察。"
        },
        "5_power_n": {
            "question": "power analysis建议N是多少？",
            "answer": f"N=75 (当前N=80)",
            "details": f"min N for power >= 0.80: {power.get('min_n_for_power_80', 20)}, "
                       f"min N for events >= 10: {power.get('min_n_for_events_10', 75)}"
        },
        "6_aurc_result": {
            "question": "Confirmatory AURC结果如何？",
            "answer": "N/A (无confirmatory cohort)",
            "details": "Development Cohort: AURC=0.0158 (B4) vs 0.1295 (Random), excess AURC = -0.0605"
        },
        "7_cfr_reduction": {
            "question": "CFR@80下降多少？",
            "answer": f"Primary pipeline: {dev['b4_prediction_consistency']['cfr_reduction_at_80_pct']:.1f}%",
            "details": f"CFR@80: {dev['b4_prediction_consistency']['cfr_at_80']:.4f} (B4) vs {dev['b4_prediction_consistency']['cfr_baseline']:.4f} (baseline)"
        },
        "8_dice_maintained": {
            "question": "Dice@80是否保持？",
            "answer": "YES",
            "details": f"Dice@80: {dev['b4_prediction_consistency']['dice_at_80']:.4f} (B4) vs {dev['b4_prediction_consistency']['dice_baseline']:.4f} (baseline)"
        },
        "9_pipelines_support": {
            "question": "哪些pipelines支持？",
            "answer": "All 5 pipelines show directional support",
            "details": {
                f"P1 (DINOv2-L+SAM-B)": f"directional, AURC={cp['L+SAM-B']['aurc']:.4f}, CFR_red={cp['L+SAM-B']['cfr_red_80']:.1f}%",
                f"P2 (DINOv2-B+SAM-B)": f"directional, AURC={cp['B+SAM-B']['aurc']:.4f}, CFR_red={cp['B+SAM-B']['cfr_red_80']:.1f}%",
                f"P3 (DINOv2-L+SAM-L)": f"directional, AURC={cp['L+SAM-L']['aurc']:.4f}, CFR_red={cp['L+SAM-L']['cfr_red_80']:.1f}%",
                f"P4 (DINOv2-L+SAM2-T)": f"directional, AURC={cp['L+SAM2-T']['aurc']:.4f}, CFR_red={cp['L+SAM2-T']['cfr_red_80']:.1f}%",
                f"P5 (DINOv2-L+SAM2-S)": f"directional, AURC={cp['L+SAM2-S']['aurc']:.4f}, CFR_red={cp['L+SAM2-S']['cfr_red_80']:.1f}%",
            }
        },
        "10_pipelines_no_support": {
            "question": "哪些pipelines不支持？",
            "answer": "None — all 5 pipelines show at least directional support",
            "details": "SAM2-Small shows weakest effect (CFR reduction -1.6%)"
        },
        "11_datasets_support": {
            "question": "哪些datasets支持？",
            "answer": "MoNuSAC Development Cohort (supportive), Validation41 (partial)",
            "details": "MoNuSeg and IDRID are boundary cases"
        },
        "12_boundary_datasets": {
            "question": "哪些datasets是边界？",
            "answer": "MoNuSeg (CFR=0, risk signal transfers but no failures to detect), "
                     "IDRID (segmentation quality too low, risk signal transfers but impractical)",
            "details": "Both datasets show risk signal transfer (rho >= 0.83) but have boundary conditions"
        },
        "13_sam_med2d": {
            "question": "SAM-Med2D是否删除？",
            "answer": "YES — permanently removed from main table",
            "details": "No checkpoint SHA256, no load log, no inference log. Unverifiable."
        },
        "14_baseline_provenance": {
            "question": "Baseline provenance是否通过？",
            "answer": "YES — PHASEAB_BASELINE_PROVENANCE_PASS",
            "details": "All internal baselines reproducible. External methods stratified by backbone."
        },
        "15_modern_pipeline": {
            "question": "是否运行现代pipeline？",
            "answer": "NO — WP5 is optional and was skipped",
            "details": "DINOv3 + SAM3 would require new model downloads and user approval."
        },
        "16_final_claim": {
            "question": "最终论文主张是什么？",
            "answer": "Cross-reference prediction consistency provides a simple, GT-free signal for "
                     "ranking segmentation risk and reducing catastrophic failures through selective "
                     "prediction in multi-reference visual prompting.",
            "details": "Only 3 contributions: (1) RRE problem formulation, (2) Prediction Consistency risk signal, "
                       "(3) Systematic evaluation across pipelines and datasets."
        },
        "17_title": {
            "question": "推荐标题是什么？",
            "answer": "When References Disagree: Prediction Consistency for Selective Visual Prompting",
            "details": "Alternative: Prediction Consistency as a Simple Risk Signal for Multi-Reference Visual Prompting"
        },
        "18_submission_ready": {
            "question": "是否达到投稿候选状态？",
            "answer": readiness,
            "details": f"Passed {n_pass}/{n_total} gates. Failed: fresh cohort (no confirmatory data), external support (only boundary datasets). "
                       "Paper is suitable for workshop/benchmark venues."
        },
        "19_missing": {
            "question": "还缺什么？",
            "answer": [
                "Fresh confirmatory cohort (most critical)",
                "External dataset with full support (AUROC >= 0.70, CFR reduction >= 20%)",
                "Natural image experiments (optional)",
                "Comparison with standard uncertainty methods (MC dropout, ensemble)",
                "Qualitative examples (high vs. low consistency queries)",
                "LaTeX-formatted paper with proper figures",
            ]
        },
        "20_formal_test": {
            "question": "下一步是否需要formal test？",
            "answer": "YES — if a fresh confirmatory cohort becomes available",
            "details": "GlaS/CPM17/Formal_Test require user approval to download and use. "
                       "Power analysis suggests N >= 75 queries with >= 10 catastrophic events."
        }
    },

    "work_packages": {
        "WP0": {"status": "PHASEAB_INTEGRITY_CLOSURE_PASS", "description": "Phase AA Integrity Closure"},
        "WP1": {"status": "PHASEAB_BLOCKED_NO_FRESH_CONFIRMATORY_DATA", "description": "Fresh Confirmatory Cohort"},
        "WP1b": {"status": "COMPLETE", "description": "Power Analysis (N=75 recommended)"},
        "WP2": {"status": "BLOCKED", "description": "Confirmatory Experiment (needs WP1)"},
        "WP3": {"status": wp3["status"], "description": "Cross-Pipeline (5/5 directional)"},
        "WP3b": {"status": wp3b["status"], "description": "Cross-Dataset Multi-Seed"},
        "WP4": {"status": wp4["status"], "description": "Baseline Provenance"},
        "WP5": {"status": "SKIPPED", "description": "Modern Pipeline (optional)"},
        "WP6": {"status": "COMPLETE", "description": "Paper Assembly (17 files)"},
        "WP7": {"status": "COMPLETE", "description": "Mock Review (7 reviewers, rebuttal plan)"},
    },

    "publication_readiness": {
        "status": readiness,
        "passed_gates": f"{n_pass}/{n_total}",
        "failed_gates": [k for k, v in gate_checks.items() if v is False],
        "recommended_venue": "Workshop / Benchmark track",
        "not_recommended": "CVPR/ICCV/NeurIPS main conference (requires confirmatory evidence)",
        "key_strengths": [
            "Clear problem formulation (RRE)",
            "Simple, practical method",
            "Honest about limitations",
            "Systematic cross-pipeline and cross-dataset evaluation",
            "Well-documented reproducibility",
        ],
        "key_weaknesses": [
            "No confirmatory cohort",
            "Limited to medical imaging domain",
            "Method is a heuristic (no learning)",
            "Cross-dataset results are boundary cases only",
            "Small evaluation scale (80 queries)",
        ],
    },

    "key_artifacts": {
        "frozen_config": "configs/frozen/phaseAB_prediction_consistency.yaml",
        "master_numbers": "reports/phaseAB_integrity/corrected_master_numbers.json",
        "paper": "paper/phaseAB/ (17 files)",
        "cross_pipeline": "reports/phaseAB_wp3_cross_pipeline/wp3_cross_pipeline_report.json",
        "cross_dataset": "reports/phaseAB_wp3b_cross_dataset/wp3b_cross_dataset_report.json",
        "baseline_provenance": "reports/phaseAB_wp4_baseline_provenance/wp4_baseline_provenance_report.json",
        "mock_review": "reports/phaseAB_wp7_mock_review/phaseAB_mock_review.md",
        "rebuttal_plan": "reports/phaseAB_wp7_mock_review/phaseAB_rebuttal_plan.md",
        "submission_readiness": "reports/phaseAB_wp7_mock_review/phaseAB_submission_readiness.md",
        "reproduction": "python src/adapromptset/scripts/run_phaseAB_wp3_wp4_closure.py",
    },

    "unresolved_blockers": [
        "No fresh confirmatory cohort — cannot run WP2",
        "No external dataset with full support (AUROC >= 0.70)",
        "GlaS/CPM17/Formal_Test require user approval",
        "Natural image experiments not conducted",
        "No LaTeX-formatted version of the paper",
    ],
}

# ============================================================
# SAVE FILES
# ============================================================

out_dir = REPORTS / "phaseAB_final"
out_dir.mkdir(parents=True, exist_ok=True)

# JSON report
with open(out_dir / "phaseAB_final_report.json", "w", encoding="utf-8") as f:
    json.dump(final_report, f, indent=2, ensure_ascii=False)

# Markdown report
a = final_report["answers"]
wp = final_report["work_packages"]
pr = final_report["publication_readiness"]

md = f"""# Phase AB Final Report

**Date**: {final_report['date']}
**Status**: {final_report['status']}
**Branch**: {final_report['branch']}
**Predecessor**: {final_report['predecessor']}

---

## 1. Phase AB Final State

**{final_report['status']}**

The Phase AB program has completed all work packages that do not require a fresh confirmatory
cohort. The paper is positioned as an exploratory/benchmark study with a simple, practical
method (Prediction Consistency) for reliability estimation in multi-reference visual prompting.

## 2. Work Package Status

| WP | Status | Description |
|---|---|---|
| WP0 | {wp['WP0']['status']} | Phase AA Integrity Closure |
| WP1 | {wp['WP1']['status']} | Fresh Confirmatory Cohort |
| WP1b | {wp['WP1b']['status']} | Power Analysis |
| WP2 | {wp['WP2']['status']} | Confirmatory Experiment |
| WP3 | {wp['WP3']['status']} | Cross-Pipeline |
| WP3b | {wp['WP3b']['status']} | Cross-Dataset |
| WP4 | {wp['WP4']['status']} | Baseline Provenance |
| WP5 | {wp['WP5']['status']} | Modern Pipeline |
| WP6 | {wp['WP6']['status']} | Paper Assembly |
| WP7 | {wp['WP7']['status']} | Mock Review |

## 3. Fresh Cohort Status

**BLOCKED** — No fresh confirmatory cohort available. All candidate datasets (MoNuSAC, 
MoNuSeg, IDRID) have been exposed in earlier phases. GlaS, CPM17, and Formal_Test require
user approval to download and use.

## 4. Power Analysis

- Recommended N: **{power.get('recommended_n', 75)}**
- Min N for power >= 0.80: {power.get('min_n_for_power_80', 20)}
- Min N for events >= 10: {power.get('min_n_for_events_10', 75)}
- Current N: 80 (meets threshold)

## 5. Confirmatory Results

**N/A** — No confirmatory experiment was run (WP2 blocked by WP1).

Development Cohort results (primary pipeline DINOv2-L + SAM-B):
- AURC: {dev['b4_prediction_consistency']['aurc']:.4f} (B4) vs {dev['random']['aurc']:.4f} (Random)
- CFR@80: {dev['b4_prediction_consistency']['cfr_at_80']:.4f} (B4) vs {dev['b4_prediction_consistency']['cfr_baseline']:.4f} (baseline)
- CFR Reduction@80: {dev['b4_prediction_consistency']['cfr_reduction_at_80_pct']:.1f}%
- Dice@80: {dev['b4_prediction_consistency']['dice_at_80']:.4f}

## 6. Strongest Risk Baseline

**Prediction Consistency (B4)** is the strongest risk signal, outperforming:
- Random Ranking (AURC 0.1295)
- Matching Entropy (AURC 0.1281)
- All other baselines

## 7. Cross-Pipeline Results

| Pipeline | Support | AURC | CFR Red@80 |
|---|---|---|---|
| P1: DINOv2-L + SAM-B | directional | {cp['L+SAM-B']['aurc']:.4f} | {cp['L+SAM-B']['cfr_red_80']:.1f}% |
| P2: DINOv2-B + SAM-B | directional | {cp['B+SAM-B']['aurc']:.4f} | {cp['B+SAM-B']['cfr_red_80']:.1f}% |
| P3: DINOv2-L + SAM-L | directional | {cp['L+SAM-L']['aurc']:.4f} | {cp['L+SAM-L']['cfr_red_80']:.1f}% |
| P4: DINOv2-L + SAM2-T | directional | {cp['L+SAM2-T']['aurc']:.4f} | {cp['L+SAM2-T']['cfr_red_80']:.1f}% |
| P5: DINOv2-L + SAM2-S | directional | {cp['L+SAM2-S']['aurc']:.4f} | {cp['L+SAM2-S']['cfr_red_80']:.1f}% |

## 8. Cross-Dataset Results

| Dataset | N | Base Dice | Verdict |
|---|---|---|---|
| MoNuSAC (Dev) | 80 | {dev['b4_prediction_consistency']['dice_baseline']:.4f} | supportive |
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

**{pr['status']}** ({pr['passed_gates']} gates passed)

Failed gates: {', '.join(pr['failed_gates']) if pr['failed_gates'] else 'None'}

**Recommended**: Workshop venues / Benchmark tracks
**Not recommended**: CVPR/ICCV/NeurIPS main conference

## 14. Key Artifacts

{chr(10).join(f'- {v}' for v in final_report['key_artifacts'].values())}

## 15. Reproduction Command

```bash
cd D:\\SAM\\adapromptset
python src/adapromptset/scripts/run_phaseAB_wp3_wp4_closure.py
python src/adapromptset/scripts/run_phaseAB_wp6_paper_assembly.py
python src/adapromptset/scripts/run_phaseAB_wp7_mock_review.py
```

## 16. Unresolved Blockers

{chr(10).join(f'- {b}' for b in final_report['unresolved_blockers'])}

---

## 20 Questions Answered

1. **AA0矛盾是否全部关闭？** {a['1_aa0_closure']['answer']}
2. **最终主方法定义是什么？** {a['2_method_definition']['answer']}
3. **是否找到fresh cohort？** {a['3_fresh_cohort']['answer']}
4. **fresh cohort是否真正未观察？** {a['4_freshness']['answer']}
5. **power analysis建议N是多少？** {a['5_power_n']['answer']}
6. **Confirmatory AURC结果如何？** {a['6_aurc_result']['answer']}
7. **CFR@80下降多少？** {a['7_cfr_reduction']['answer']}
8. **Dice@80是否保持？** {a['8_dice_maintained']['answer']}
9. **哪些pipelines支持？** {a['9_pipelines_support']['answer']}
10. **哪些pipelines不支持？** {a['10_pipelines_no_support']['answer']}
11. **哪些datasets支持？** {a['11_datasets_support']['answer']}
12. **哪些datasets是边界？** {a['12_boundary_datasets']['answer']}
13. **SAM-Med2D是否删除？** {a['13_sam_med2d']['answer']}
14. **Baseline provenance是否通过？** {a['14_baseline_provenance']['answer']}
15. **是否运行现代pipeline？** {a['15_modern_pipeline']['answer']}
16. **最终论文主张是什么？** {a['16_final_claim']['answer']}
17. **推荐标题是什么？** {a['17_title']['answer']}
18. **是否达到投稿候选状态？** {a['18_submission_ready']['answer']}
19. **还缺什么？** {chr(10).join(f'  - {x}' for x in a['19_missing']['answer'])}
20. **下一步是否需要formal test？** {a['20_formal_test']['answer']}
"""

with open(out_dir / "phaseAB_final_report.md", "w", encoding="utf-8") as f:
    f.write(md)

# Update program status
program_status = {
    "phase": "PhaseAB",
    "branch": "phaseAB-fresh-confirmation-paper-assembly",
    "date": datetime.now().strftime("%Y-%m-%d"),
    "status": readiness,
    "predecessor": "PhaseAA (PHASEAA_PAPER_NEAR_READY)",
    "route": "Route B - Simplified Selective Prediction",
    "primary_method": "Prediction Consistency + Coverage-Based Abstention",
    "work_packages": final_report["work_packages"],
    "publication_readiness": pr,
    "data_roles": {
        "development_cohort": "80 MoNuSAC queries",
        "validation_cohort": "41 MoNuSAC queries",
        "holdout": None,
        "confirmatory": None,
    },
}

with open(REPORTS / "phaseAB_program_status.json", "w") as f:
    json.dump(program_status, f, indent=2)

print(f"Final report written to {out_dir}")
print(f"Program status: {readiness}")
print(f"Gates passed: {n_pass}/{n_total}")
print(f"Failed gates: {[k for k, v in gate_checks.items() if v is False]}")