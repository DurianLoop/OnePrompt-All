# GlaS Region Prompt Protocol v1

Date: 2026-07-19
Protocol: `adapromptset.phase3.glas_region_prompt.v1`

## Scope

This protocol studies automatic region prompt feasibility on GlaS using frozen reference-conditioned matching evidence.

It does not train adaptive policy, modify P3, alter Gate D thresholds, change the matcher, change the Gate C candidate generator, or access GlaS testA/testB.

## Data Roles

- 55 official-train utility images: Phase 3 internal region pool.
- 40 of 55: region proposal calibration.
- 15 of 55: internal region method selection.
- 30 official-train strict-validation images: Gate R1/R2/R3 validation only.
- testA/testB: locked formal tests, not accessible unless a final method is frozen by the state machine.

Artifacts:

- `manifests/phase3/glas_region_calibration_40.csv`
- `manifests/phase3/glas_region_selection_15.csv`
- `manifests/phase3/glas_region_references.csv`

## Evidence Aggregation

Two aggregation modes are preregistered:

- `E1_mean`: mean of normalized per-reference evidence maps.
- `E2_max`: max of normalized per-reference evidence maps.

If complete dense maps are unavailable, Phase 3 may use the frozen local-peaks candidate evidence as a derived sparse matching-evidence map and must record this limitation. Regeneration of dense maps is allowed only with the frozen DINOv2-L matcher and Gate C configuration/hash.

No per-query aggregation routing is allowed on strict validation.

## Region Bank

The frozen bank is `configs/actions/glas_region_bank_v1.{yaml,json}`.

Automatic actions:

- R00 direct matching mask
- R01 full-image box
- R02 size-matched random box
- R03 automatic union box
- R04 automatic largest-component box
- R05 automatic top-3 component boxes
- R06 automatic hard mask prompt
- R07 automatic soft-logit mask prompt
- R08 automatic largest-box + soft mask
- R09 frozen matching top-point baseline

Privileged upper bounds:

- U01 GT union box
- U02 GT component boxes
- U03 degraded GT coarse mask

Upper bounds are not automatic methods.

## Parameter Grid

Only the following grid is allowed:

- contrast quantile: `{0.70, 0.80, 0.90}`
- minimum component area ratio: `{0.001, 0.005}`
- closing radius: `{0, 3, 5}`
- soft-logit temperature: `{5, 10}`
- top component count: `{1, 3}`

Selection uses the 40 calibration and 15 internal selection pool only. The 30 strict-validation queries cannot select thresholds or actions.

## Gate R1

Gate R1 evaluates region proposal quality before SAM inference.

Pass if at least one automatic box route or mask route meets the preregistered thresholds:

- box route: box-IoU improvement over size-matched random box `>= 0.10`, win rate `>= 0.70`, foreground recall `>= 0.70`, empty proposal rate `<= 0.10`;
- mask route: raw-mask Dice `>= 0.30`, control improvement `>= 0.15`, foreground recall `>= 0.60`, empty proposal rate `<= 0.10`.

Fail state:

`STOP_GLAS_REGION_PROPOSAL_FAILURE`

Pass state:

`PHASE3_GATE_R1_PASS`

## Gate R2

Gate R2 runs SAM-B only after R1 passes.

Automatic region methods:

- R03, R04, R05, R06, R07, R08

Controls:

- R01, R02, R09, R00

Upper bounds:

- U01, U02, U03

Pass if at least one automatic region method meets:

- mean Dice margin over matched control `>= 0.05`;
- query win rate `>= 0.70`;
- paired bootstrap point estimate `> 0`;
- catastrophic failure rate `<= 0.20`;
- cost matching passes;
- region recovery `>= 0.30`;
- no query GT used for automatic evidence.

Possible states:

- `PHASE3_GATE_R2_PASS`
- `PIVOT_DIRECT_MATCHING_MASK`
- `STOP_AUTOMATIC_REGION_GAP_TOO_LARGE`
- `STOP_REGION_PROMPT_INTERFACE_FAILURE`

## Gate R3

Gate R3 runs only if R2 passes. It estimates oracle action heterogeneity among the frozen automatic region actions.

Continue to router only if oracle gap `>= 0.010`, at least 25% of queries prefer a non-global-fixed action, gain is not explained only by calls, and at least two region representations win stably.

Possible states:

- `CONTINUE_PROMPT_TYPE_ROUTER`
- `PIVOT_DETERMINISTIC_REGION_PROMPT`
- `PIVOT_REGION_ORACLE_ANALYSIS`

## Test Boundary

testA/testB remain inaccessible until a final deterministic, direct-mask, or router method is frozen with `docs/phase3_test_freeze.md`.
