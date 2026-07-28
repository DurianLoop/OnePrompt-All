# Phase W — Data Role Registry

**Date**: 2026-07-27

## Current Data Sources

### MoNuSAC (80 queries)
- **Source**: phaseU_wp1/sam_b/phase2_episode_instability.csv
- **Current roles claimed**: holdout, development, ablation, CV
- **Issue**: Same 80 queries used for all purposes
- **Resolution**: To be determined in WP0

### MoNuSeg (30 queries)
- **Source**: phaseT_wp2_monuseg/phase2_episode_instability.csv
- **Current role**: cross-dataset validation

### IDRID (30 queries)
- **Source**: phaseT_wp2_idrid/phase2_episode_instability.csv
- **Current role**: cross-dataset validation

### Pascal-5i (fold-0 only)
- **Source**: TBD
- **Current role**: preliminary cross-domain evidence

### COCO-20i (fold-0 only)
- **Source**: TBD
- **Current role**: preliminary cross-domain evidence

## Legal Role Definitions

| Role | Training | Feature Design | Threshold Selection | Model Selection | Ablation | Paper Decision |
|------|----------|----------------|---------------------|-----------------|----------|----------------|
| Development | Yes | Yes | Yes | Yes | Yes | Yes |
| Validation | No | No | Yes | Yes | No | Limited |
| Internal Holdout | No | No | No | No | No | No |
| Formal Test | No | No | No | No | No | No (locked) |