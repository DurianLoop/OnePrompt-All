# GlaS Gate D Confirmation Protocol v1

Date: 2026-07-19
Protocol version: `adapromptset.phase2d.glas_gate_d_confirmation.v1`
Config: `configs/segmenter/glas_gate_d_confirmation_v1.yaml`

## Scope

This protocol confirms whether GlaS segmenter execution is viable on the full 30-query strict-validation pool. It does not test AdaPromptSet policy learning, action-bank utility, oracle gap, transfer, or formal test generalization.

## Inputs

Allowed inputs:

- `manifests/phase2/glas/episodes_val.csv`;
- GlaS 30 strict-validation query images from official train;
- frozen 55-image utility/reference pool through the existing episode manifest;
- frozen redesigned `local_peaks` candidate output for diagnostic C3 anchors;
- SAM-B checkpoint `D:\SAM\assets\checkpoints\sam_vit_b.pth`;
- SAM-L checkpoint `D:\SAM\assets\checkpoints\sam_vit_l_0b3195.pth`.

Forbidden inputs:

- GlaS testA;
- GlaS testB;
- MoNuSAC reruns;
- IDRID/DDR/CPM-17;
- SAM2, SAM-H, MedSAM, or any third segmenter;
- new candidate redesign;
- new action bank;
- policy training.

## Statistical Unit

The statistical unit is the query image. All 30 strict-validation query images must be included. Prompt families, components, references, and SAM calls are repeated measurements inside a query and must not be counted as independent samples.

## Prompt Families

Primary:

- `P3_component_grouped_centroids`: take up to three largest foreground connected components, create one positive centroid prompt group per component, run one SAM call per group, and fuse group masks by binary union.

Secondary:

- `P1_largest_component_centroid`: one positive point at the largest-component centroid.
- `P2_fixed_multi_centroid_one_group`: up to three largest component centroids in one group.
- `P4_union_mask_bounding_box`: tight union foreground bounding box.
- `P5_positive_plus_controlled_negative`: largest-component centroid plus one deterministic near-boundary negative point.

## Matched Controls

Primary matched control score:

`mean(C1_random_coordinate, C2_mismatched_reference)`

Controls:

- `C1_random_coordinate`: fixed-seed random coordinates with the same point labels, group count, and SAM call count as the privileged family.
- `C2_mismatched_reference`: coordinates or box from the first frozen utility/reference support mask for the query episode, with matching point labels, group count, and SAM call count. This uses the existing support-reference manifest and never uses the query GT mask.
- `C3_candidate_anchor`: frozen redesigned candidate safe-anchor diagnostic, not part of the primary Gate D margin.

## Shared Inference Rules

- multimask selection: highest SAM score;
- mask threshold: SAM boolean output;
- image resize: `518x518` RGB bicubic;
- mask resize: `518x518` nearest;
- group fusion: binary union;
- empty or failed call: counted as failed and catastrophic;
- no privileged-only postprocess;
- no per-query best family selection;
- no per-query best SAM mask oracle selection.

## Gate D Pass Criteria

For SAM-B and SAM-L independently:

1. Coordinate validation, point label audit, and cost matching pass.
2. P3 mean paired Dice margin over matched control is at least `0.05`.
3. P3 win rate is at least `0.70`.
4. P3 catastrophic failure rate is at most `0.10`.
5. Query-level paired bootstrap point estimate is positive.
6. No hidden oracle or result-conditioned selection is used.

Catastrophic failure means any of:

- completely empty output;
- nearly image-wide unrelated mask;
- valid foreground prediction area below the minimum threshold;
- Dice below `0.20`;
- invalid coordinate or output;
- failed SAM call.

## Segmenter Selection

- SAM-B pass and SAM-L fail: `PHASE2_GATE_D_PASS_SAMB`.
- SAM-B fail and SAM-L pass: `PHASE2_GATE_D_PASS_SAML`.
- both pass: default `PHASE2_GATE_D_PASS_BOTH_FREEZE_SAMB`.
- both pass and SAM-L has at least `0.03` higher P3 mean Dice and at least `0.05` lower catastrophic failure rate: `PHASE2_GATE_D_PASS_BOTH_FREEZE_SAML`.
- both fail: `STOP_GLAS_SEGMENTER_FAILURE_CONFIRMED`.
- overall fail but a predeclared morphology subgroup shows stable strong response: `PIVOT_GLAS_MORPHOLOGY_CONDITIONAL`.

No second Gate D criterion revision is allowed after this confirmation.
