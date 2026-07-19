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
| Phase 2D Gate D confirmation | `STOP_GLAS_SEGMENTER_FAILURE_CONFIRMED` | The preregistered 30-query strict-validation confirmation ran after commit `f5c2de3`. SAM-B and SAM-L both failed the P3 privileged-vs-matched-control viability criterion. |

Previous Phase 2 status: `STOP_GLAS_SEGMENTER_FAILURE`

Current overall status: `STOP_GLAS_SEGMENTER_FAILURE_CONFIRMED`

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

All later Phase 2 gates remain stopped/prohibited unless the preregistered Phase 2D confirmation reaches a pass state. GlaS testA/testB remain locked and were not used.

## Phase 2D Confirmation Preregistration

The original 6-case Gate D reports remain preserved as:

`EXPLORATORY_GATE_D_FAIL_UNDER_ALL_CASE_MARGIN_RULE`

Phase 2D confirmation artifacts created before new SAM inference:

| Artifact | Status |
| --- | --- |
| `docs/glas_gate_d_confirmation_protocol_v1.md` | frozen protocol |
| `docs/glas_gate_d_amendment_rationale.md` | amendment rationale |
| `configs/segmenter/glas_gate_d_confirmation_v1.yaml` | frozen config |
| `reports/phase2_glas/gate_d_confirmation/protocol_snapshot.json` | protocol snapshot |
| `reports/phase2_glas/gate_d_confirmation/input_hashes.json` | input hashes |
| `reports/phase2_glas/gate_d_confirmation/compute_plan.json` | compute budget |
| `src/adapromptset/scripts/run_glas_gate_d_confirmation.py` | confirmation runner |
| `scripts/run_glas_gate_d_confirmation.py` | wrapper |
| `tests/test_run_glas_gate_d_confirmation.py` | focused tests |

Primary confirmation rule:

- use all 30 strict-validation query images;
- use `P3_component_grouped_centroids` as primary;
- compare against `mean(C1_random_coordinate, C2_mismatched_reference)`;
- evaluate SAM-B and SAM-L independently;
- require P3 mean paired Dice margin `>= 0.05`, query win rate `>= 0.70`, catastrophic failure rate `<= 0.10`, positive bootstrap point estimate, coordinate validation, cost matching, and no hidden oracle.

SAM confirmation has now run. Gate E/F/G/H/I remain closed because the final Phase 2D state is `STOP_GLAS_SEGMENTER_FAILURE_CONFIRMED`.

## Phase 2D Confirmation Result

| Metric | SAM-B | SAM-L |
| --- | ---: | ---: |
| Query count | `30` | `30` |
| P3 mean privileged Dice | `0.509561` | `0.470705` |
| P3 matched control Dice | `0.667498` | `0.567614` |
| P3 paired margin | `-0.157937` | `-0.096909` |
| P3 win rate | `0.433333` | `0.466667` |
| P3 bootstrap CI | `[-0.250074, -0.076146]` | `[-0.196157, -0.001773]` |
| P3 catastrophic failure rate | `0.666667` | `0.400000` |
| Coordinate audit | `PASS` | `PASS` |
| Cost audit | `PASS` | `PASS` |
| Status | `FAIL` | `FAIL` |

Final consequence:

- Gate E action-bank validation: `CLOSED_BY_CONFIRMED_GATE_D_FAILURE`
- Gate F oracle-gap analysis: `CLOSED_BY_CONFIRMED_GATE_D_FAILURE`
- Gate H policy learning: `PROHIBITED_BY_CONFIRMED_GATE_D_FAILURE`
- Gate I formal test: `PROHIBITED_BY_CONFIRMED_GATE_D_FAILURE`

GlaS testA/testB remain locked and unused.
