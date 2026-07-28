# Phase AB Claim Registry

## Allowed Claims

| ID | Claim | Status | Evidence |
|----|-------|--------|----------|
| C1 | Prediction Consistency is a simple, GT-free risk signal | SUPPORTED | 5/5 pipelines directional support |
| C2 | Coverage-based abstention reduces CFR | SUPPORTED | CFR@80 reduction 75% on primary pipeline |
| C3 | Risk-coverage trade-off is measurable | SUPPORTED | Risk-coverage curves computed |
| C4 | Method is architecture-dependent | SUPPORTED | SAM2 pipelines show no benefit |
| C5 | Full RAMP has no added value | SUPPORTED | Component ablation Δ=0 |
| C6 | RRN is not competitive | SUPPORTED | AUROC=0.573, Appendix only |

## Prohibited Claims

| ID | Claim | Reason |
|----|-------|--------|
| P1 | Training-free method | Matcher (DINOv2) is trained |
| P2 | Calibration-free | Not applicable (no probability output) |
| P3 | Universal applicability | SAM2 pipelines fail |
| P4 | Holdout-confirmed | No true holdout exists |
| P5 | Identical conditions | External baselines use different backbones |
| P6 | Anti-correlation is a structural property | Mechanical artefact, not causal |
| P7 | Method works on all foundation models | Only tested on SAM/SAM2 family |

## Partially Supported Claims

| ID | Claim | Limitation |
|----|-------|------------|
| PS1 | Cross-dataset transfer | Only 1/3 datasets show support |
| PS2 | Cross-pipeline generality | 5/5 directional but 0/5 strong |
| PS3 | Learned risk models improve over simple signals | B4 often outperforms M2 models |

## Boundary Conditions

- SAM2-Tiny/Small: no benefit from Prediction Consistency
- IDRID: risk ranking transfers but segmentation quality is poor
- MoNuSeg: no catastrophic failures to detect (CFR=0)
- Low-Dice regimes: risk signal may still rank but segmentation not usable