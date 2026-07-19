# GlaS Gate D Segmenter Alternative v1 Pre-registration

Date: 2026-07-19
Parent protocol: `adapromptset.phase2.glas_strict.v2`
Parent Gate D status: `FAIL`

## Failure Being Addressed

SAM-B Gate D smoke produced target-related privileged masks but failed the control-margin check on at least one selected GlaS strict-validation case:

- cases: `6`;
- mean privileged best Dice: `0.733908`;
- mean control best Dice: `0.638274`;
- mean privileged margin: `0.095634`;
- issue: privileged prompts did not clearly exceed random/mismatched controls on every selected case.

## Single Allowed Alternative

The only segmenter alternative allowed in this branch is:

`SAM-L`

Checkpoint:

`D:\SAM\assets\checkpoints\sam_vit_l_0b3195.pth`

Rationale:

- SAM-L keeps the same SAM v1 prompt API as SAM-B;
- it changes less of the protocol than switching to SAM2;
- no matcher, candidate generator, action bank, Gate threshold, or test split is changed.

## Prohibited Changes

- Do not test SAM2 unless this SAM-L branch is explicitly abandoned and user approval is requested.
- Do not compare many segmenters and report the best.
- Do not use GlaS testA/testB.
- Do not proceed to Gate E/F unless this single alternative passes Gate D.
- Do not change the selected Gate D cases after seeing SAM-L results.

## Evaluation

Run the same Gate D script and same selected-case policy on the fixed GlaS strict-validation manifest. Outputs must be isolated under:

`reports/phase2_glas/gate_d_segmenter_smoke_saml/`

Pass/fail criteria are unchanged:

- privileged prompts must produce target-related masks;
- privileged best Dice must be at least `0.20` per selected case;
- privileged best Dice must exceed random/mismatched controls by at least `0.05` per selected case;
- SAM call count must match prompt group count;
- failed calls must be fully counted.

If SAM-L fails, the Phase 2 GlaS branch must stop at:

`STOP_GLAS_SEGMENTER_FAILURE`
