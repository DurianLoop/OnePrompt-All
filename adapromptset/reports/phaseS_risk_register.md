# Phase S Risk Register

Date: 2026-07-22
Branch: `phaseS-multitrack-reference-robust-research`

## Identified Risks

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Phase R variance metric was computed over wrong axis (cross-query vs within-query) | Medium | High | Track A §6.3 code trace + corrected definitions |
| R2 | Reference pool insufficient for k=6 with 10 seeds | Low | High | Phase R confirmed 70-80 available; Phase S reuses |
| R3 | Reliability model overfits train, no validation signal | Medium | Medium | Model ladder M0→M5 with inner CV; stop at first non-improving |
| R4 | Weight collapse in softmax weighting | Medium | Medium | Weight floor ε≥0.05; monitor max weight share |
| R5 | Holdout accidentally accessed before freeze | Low | Critical | Lock notice + path checks in scripts |
| R6 | Cross-dataset failure (GlaS/DDR) despite MoNuSAC pass | Medium | Medium | Distinguish zero-shot vs retraining; report honestly |
| R7 | Compute budget exceeded before any Gate passes | Low | High | Checkpoints at 50/100/160/210; reuse Phase R caches |
| R8 | Prompt interface is the bottleneck but Track D also fails | Medium | High | Negative-result paper with mechanistic explanation |
| R9 | Patient leakage across splits | Low | Critical | Reuse Phase Y/R manifests; no new sampling |
| R10 | Test set accidentally accessed | Low | Critical | Test lock notice; path validation in all scripts |
| R11 | Post-hoc threshold tuning after Gate | Medium | High | Frozen configs; claim-boundary audit |
| R12 | Search Oracle mistakenly treated as deployable | Low | High | Prohibited in charter; audit scripts |
| R13 | Mean Dice reported without variance / tail risk | Medium | Medium | Metric definitions require full reporting suite |
| R14 | Set model (Track C) overfits train via high capacity | Medium | Medium | <5M param cap; leave-one-ref-out stability check |
| R15 | Direct method (Track D) needs labels not in protocol | Low | Medium | Use only reference-conditioned losses; no query GT at inference |

## Risk Monitoring

- After each Gate, update this register with realized risks.
- Any risk that materializes triggers a decision-tree re-evaluation.
- Critical risks (R5, R9, R10) trigger immediate program pause if realized.
