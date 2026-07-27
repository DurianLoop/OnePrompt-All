# Phase W — Experiment Registry

**Date**: 2026-07-27
**Branch**: phaseW-critical-experiment-closure

## Registry

| WP | Experiment | Status | Input Data | Output | Notes |
|----|-----------|--------|------------|--------|-------|
| WP0 | Split provenance audit | pending | phaseU_wp1/*/phase2_*.csv | reports/phaseW_split_audit/ | CPU only |
| WP1 | Component ablation re-run | pending | phaseU_wp1/sam_b/phase2_raw_per_call.csv | reports/phaseW_component_ablation/ | Needs GPU |
| WP2 | Simple risk baseline comparison | pending | phaseU_wp1/*/phase2_episode_instability.csv | reports/phaseW_simple_baselines/ | CPU mostly |
| WP3 | Risk-coverage curves | pending | phaseU_wp1/*/phase2_episode_instability.csv | reports/phaseW_risk_coverage/ | CPU |
| WP4 | RRN grouped nested CV | done | reports/phaseV_rrn_grouped_cv/ | reports/phaseV_rrn_grouped_cv/ | Already completed |
| WP5 | SAM-Med2D audit | pending | TBD | reports/phaseW_sammed2d_audit/ | Investigation |
| WP6 | External baseline restructuring | pending | phaseU_external_*/ | reports/phaseW_external_baselines/ | CPU |
| WP7 | Anti-correlation residual effect | pending | reports/phaseV_query_level_stats/ | reports/phaseW_anticorrelation/ | CPU |
| WP8 | Cross-domain calibration | pending | phaseT_wp2_*/ | reports/phaseW_calibration/ | CPU |
| WP9 | Pascal/COCO 4-fold | pending | TBD | reports/phaseW_pascal_coco/ | Needs GPU |
| WP10 | Claim registry rebuild | pending | full_draft_v5.tex | reports/phaseW_claim_audit/ | CPU |