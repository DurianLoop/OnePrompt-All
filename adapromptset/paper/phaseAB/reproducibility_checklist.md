# Reproducibility Checklist

## Models and Data
- [x] DINOv2-L: facebook/dinov2-large (public)
- [x] DINOv2-B: facebook/dinov2-base (public)
- [x] SAM-B: facebook/sam-vit-base (public)
- [x] SAM-L: facebook/sam-vit-large (public)
- [x] SAM2-Tiny: facebook/sam2 (public)
- [x] SAM2-Small: facebook/sam2 (public)
- [x] MoNuSAC: Public dataset
- [x] MoNuSeg: Public dataset
- [x] IDRID: Public dataset

## Configuration
- [x] Frozen config: configs/frozen/phaseAB_prediction_consistency.yaml
- [x] All hyperparameters documented
- [x] Reference sampling seed: 42

## Code
- [x] Core scripts: src/adapromptset/scripts/
- [x] No data leakage between development and external datasets
- [x] Query-level statistics with bootstrap CIs

## Results
- [x] All numbers traceable to source artifacts
- [x] No query GT used in risk construction
- [x] Baseline methods clearly documented
- [ ] Fresh confirmatory cohort (not available—see Limitations)

## Prohibited Claims
- [x] No "training-free" claims
- [x] No "holdout-confirmed" claims
- [x] No "universal" claims
- [x] Full RAMP removed from main text
- [x] RRN in Appendix only
- [x] SAM-Med2D removed
