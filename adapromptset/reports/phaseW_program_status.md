# Phase W — Program Status

**Date**: 2026-07-27
**Branch**: phaseW-critical-experiment-closure
**Phase V Baseline**: PHASEV_EMPIRICAL_SELECTIVE_PREDICTION_READY

## WP Status

| WP | Status | Gate | Notes |
|----|--------|------|-------|
| WP0 | PENDING | — | Split provenance audit |
| WP1 | PENDING | — | Component ablation re-run |
| WP2 | PENDING | — | Simple baseline comparison |
| WP3 | PENDING | — | Risk-coverage curves |
| WP4 | DONE | APPENDIX_ONLY | RRN grouped CV completed |
| WP5 | PENDING | — | SAM-Med2D audit |
| WP6 | PENDING | — | External baseline restructuring |
| WP7 | PENDING | — | Anti-correlation residual effect |
| WP8 | PENDING | — | Cross-domain calibration |
| WP9 | PENDING | — | Pascal/COCO 4-fold |
| WP10 | PENDING | — | Claim registry rebuild |

## Key Known Issues

1. Component ablation: RAMP-full = RAMP-no-weights (weighting has zero effect)
2. RRN: AUROC=0.573, below 0.65 main-text threshold
3. SAM-Med2D: Checkpoint cannot be loaded, but results exist in paper
4. MoNuSAC 80 queries: Used as development + holdout simultaneously
5. Pascal/COCO: Only fold-0 available

## Current Budget

| Item | GPU-hours |
|------|-----------|
| WP1 Component ablation | 30 (if re-run) |
| WP2 Simple baselines | 20 |
| WP3 Risk-coverage | CPU |
| WP4 RRN | 0 (done) |
| WP5 SAM-Med2D | 15 (if reproducible) |
| WP6 External baselines | 35 |
| WP7 Anti-correlation | CPU |
| WP8 Calibration | 10 |
| WP9 Pascal/COCO | 45 |
| Total (max) | 155 |