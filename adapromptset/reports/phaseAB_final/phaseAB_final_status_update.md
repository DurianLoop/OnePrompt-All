# Phase AB Final Status (Updated)

**Date**: 2026-07-28 11:29
**Status**: PHASEAB_NEEDS_FRESH_DATA
**Paper**: PHASEAB_PAPER_CANDIDATE_READY (12/15 gates, workshop/benchmark venue)

---

## New Assets Generated

### Figures (4)
- `fig1_risk_coverage_curves.pdf` — CFR and Dice vs Coverage for primary pipeline
- `fig2_cross_pipeline.pdf` — AURC and CFR reduction across 5 pipelines
- `fig3_cross_dataset_boundary.pdf` — Cross-dataset boundary analysis
- `fig4_method_schematic.pdf` — Method overview diagram

### Tables (4 LaTeX)
- `table1_main_results.tex` — Main results on Development Cohort
- `table2_cross_pipeline.tex` — Cross-pipeline comparison
- `table3_cross_dataset.tex` — Cross-dataset evaluation
- `table4_ablation_ref_count.tex` — Reference count ablation

### Paper
- `full_draft.tex` — Full LaTeX paper (8 pages)
- `full_draft.pdf` — Compiled PDF (275 KB)
- `phaseAB_submission.zip` — Submission package (819 KB)

### Scripts (5 new)
- `run_phaseAB_wp3_wp4_closure.py` — WP3/WP3b/WP4 analysis
- `run_phaseAB_wp6_paper_assembly.py` — Paper section generation
- `run_phaseAB_wp7_mock_review.py` — Mock review + rebuttal
- `run_phaseAB_final_report.py` — Final report generation
- `run_phaseAB_generate_paper_assets.py` — Figures, tables, LaTeX

---

## WP5: Modern Pipeline Audit

**SKIPPED** — DINOv3 and SAM3 are not available locally. The following models ARE available for future extension:
- SAM2.1-Base+ (81M)
- SAM2.1-Large (224M)
- SAM-H (641M)

---

## Recommended Next Steps

1. **If submitting to workshop/benchmark**: Paper is ready as-is
2. **If targeting full conference**: Acquire fresh confirmatory cohort (GlaS/CPM17/Formal_Test)
3. **If extending**: Run SAM2.1-Base+/Large/H with DINOv2-L for modern pipeline results
