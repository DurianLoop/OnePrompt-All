# AdaPromptSet Phase 2 Gate Status

Date: 2026-07-19
Protocol version: `adapromptset.phase2.glas_strict.v2`
Branch: `phase2-glas-strict`

| Gate | Status | Notes |
| --- | --- | --- |
| Phase 2 read receipt | `PASS_READ` | Required docs and reports read and hashed in `docs/phase2_goal_read_receipt.md`. |
| MoNuSAC freeze | `PASS_FROZEN` | Frozen at `PIVOT_ORACLE_ANALYSIS`; no MoNuSAC policy training. |
| GlaS exposure | `TEST_SPLITS_LOCKED_NOT_FULLY_BLIND` | Prior manifest/profile exposure exists; no audited method-selection use found. |
| GlaS protocol v2 | `PRE_EXPERIMENT_PROTOCOL` | 55/30 official-train diagnostic split specified; formal test remains locked. |
| GlaS 55/30 manifest | `PASS_MANIFEST_BUILT` | `manifests/phase2/glas/*` created with 55 utility-train queries, 30 strict-val queries, and locked testA/testB cases. |
| Gate C candidate quality | `PASS_AFTER_PREDECLARED_REDESIGN` | Frozen generator failed component coverage (`0.418820 < 0.600000`). The single predeclared `local_peaks` redesign passed on 30 strict-val queries / 90 reference episodes with component coverage `0.630221`. |
| Gate D segmenter smoke | `FAIL_SEGMENTER_CONTROLS_CLOSE` | SAM-B and the single predeclared SAM-L alternative produced target-related privileged masks, but privileged prompts did not clearly exceed random/mismatched controls on every selected case. |
| Gate E action bank | `BLOCKED_BY_GATE_D` | New GlaS action bank not generated because Gate D failed. |
| Gate F GlaS oracle gap | `BLOCKED_BY_GATE_D` | Oracle-gap analysis prohibited because Gate D failed. |
| Gate G utility reliability | `BLOCKED_BY_GATE_D` | Requires Gate F continuation state, which cannot be reached. |
| Gate H policy learning | `PROHIBITED_BY_GATE_D` | Policy training prohibited because Gate D failed. |
| Gate I formal test | `PROHIBITED_BY_GATE_D` | Formal testA/testB evaluation prohibited because Gate D failed. |

Current overall status: `STOP_GLAS_SEGMENTER_FAILURE`

## Gate C Evidence

| Metric | Value |
| --- | --- |
| Frozen candidate summary | `reports/phase2_glas/candidates/gate_c_summary.json` |
| Redesign candidate summary | `reports/phase2_glas/candidates_redesign_v1/gate_c_summary.json` |
| Validation query count | `30` |
| Reference episode count | `90` |
| Max positive candidate pool | `20` |
| Mean point precision | `0.971667` |
| Mean random precision | `0.537778` |
| Precision delta | `0.433889` |
| Precision ratio | `1.806818` |
| Mean component coverage | `0.630221` |
| Required component coverage | `0.600000` |
| Mean negative blank fraction | `0.135556` |
| Coordinate validation | `PASS` |
| Gate C decision | `CONTINUE_TO_GATE_D` |

## Gate D Evidence

| Metric | SAM-B | SAM-L |
| --- | --- | --- |
| Summary | `reports/phase2_glas/gate_d_segmenter_smoke/gate_d_summary.json` | `reports/phase2_glas/gate_d_segmenter_smoke_saml/gate_d_summary.json` |
| Cases | `6` | `6` |
| Mean privileged best Dice | `0.733908` | `0.762905` |
| Mean control best Dice | `0.638274` | `0.412671` |
| Mean privileged margin | `0.095634` | `0.350234` |
| Status | `FAIL` | `FAIL` |
| Issue | privileged prompts did not clearly exceed controls on at least one case | privileged prompts did not clearly exceed controls on at least one case |

The SAM-L run was the only allowed segmenter alternative and was pre-registered in `docs/glas_gate_d_segmenter_alternative_v1.md`. No SAM2 matrix was run.

All later gates are stopped/prohibited. GlaS testA/testB remain locked and were not used.
