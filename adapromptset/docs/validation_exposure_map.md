# Validation Exposure Map

Date: 2026-07-19

## Known Observed Validation Evidence

| Dataset/protocol | Observed artifacts | Exposure consequence |
| --- | --- | --- |
| MoNuSAC MVP action/oracle pilots | Gate F, fixed-oracle, representation, bootstrap, final project reports | Cannot train MoNuSAC policy or redesign MoNuSAC action bank from these observations. |
| GlaS Gate C redesign | Phase 2 GlaS candidate artifacts | Candidate generator is frozen for that route; no further redesign on same evidence. |
| GlaS Phase 2D Gate D confirmation | `reports/phase2_glas/gate_d_confirmation/` | Grouped sparse point route closed. |
| GlaS Phase 3 R1 | `reports/phase3_glas/gate_r1_region_proposal/` | Automatic local-peaks region proposal route closed on same strict validation. |
| Phase 3 paper/report | `paper_phase3/`, `reports/phase3_final_report.*` | Negative/gated artifact only; not a method-selection source for test. |
| DDR Route B manifest gate | `manifests/phase4plus/ddr_route_b_train_*`; `manifests/phase4plus/ddr_route_b_valid_*`; `reports/route_b_ddr_manifest/` | DDR train/valid are now observed for file hashes, mask dimensions, and positive-query prevalence; DDR test remains unobserved for Phase 4+. |
| Route B metric/control freeze | `configs/experiments/route_b_promptability_benchmark_v1.json`; `reports/route_b_benchmark_freeze/` | Freezes development metrics and controls; not an empirical result and not a formal-test access event. |
| DDR Route B episode manifests | `manifests/phase4plus/ddr_route_b_train_reference_query_episodes.csv`; `manifests/phase4plus/ddr_route_b_valid_reference_query_episodes.csv` | Train/valid reference/query pairings are observed; valid references are train-only; DDR test remains unobserved. |
| DDR Route B matcher smoke | `reports/route_b_ddr_development_smoke/` | One valid MA query and one train MA reference were used for DINOv2-L development smoke; query GT was used only for evaluation metrics, not generation; DDR test remains unobserved. |

## Locked Formal Tests

- GlaS testA: locked and unused.
- GlaS testB: locked and unused.
- DDR test: candidate fresh formal domain; not yet touched by Phase 4+.

## Phase 4 Read-Only Rule

Phase 4 may inspect existing artifacts to diagnose failure and choose a research track. It may not create new method outputs, run new SAM/DINOv2 inference, or use formal test GT.
