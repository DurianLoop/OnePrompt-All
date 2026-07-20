# Fixed-Rule Manuscript Evidence Manifest

This manifest is the numerical and claim authority for
`paper/spatial_layout_matching_prompts_draft.md`.  It limits the manuscript to
the completed fixed-layout evidence and identifies results that must be
reported only as limitations or negative findings.

## Fixed-layout replication authority

All rows compare `spread5_n3 - top5_n3` on the same 303 reference-query
episodes.  Means, paired 95% bootstrap intervals (10,000 resamples, seed
20260719), and win/tie/loss counts come from
`artifacts/fixed_layout_effects.json`.

| Frozen matcher + segmenter / reference seed | Top-5 Dice | Spread Dice | Mean gain | 95% paired bootstrap CI | Win / tie / loss | Permitted interpretation |
|---|---:|---:|---:|---:|---:|---|
| DINOv2-L + SAM-B / 20260716 | 0.4196 | 0.4526 | +0.0330 | [+0.0194, +0.0467] | 125 / 110 / 68 | Primary matched-budget MoNuSAC result. |
| DINOv2-L + SAM-B / 20260717 | 0.4008 | 0.4362 | +0.0354 | [+0.0203, +0.0514] | 115 / 117 / 71 | Independent fixed-reference-seed replication. |
| DINOv2-B + SAM-B / 20260716 | 0.3835 | 0.4344 | +0.0508 | [+0.0353, +0.0674] | 131 / 101 / 71 | Matcher-size compatibility replication. |
| DINOv2-L + SAM-L / 20260716 | 0.4507 | 0.4659 | +0.0152 | [+0.0064, +0.0250] | 108 / 111 / 84 | Segmenter compatibility replication. |
| DINOv2-L + SAM-H / 20260716 | 0.4289 | 0.4439 | +0.0149 | [+0.0036, +0.0265] | 110 / 111 / 82 | Segmenter compatibility replication after balanced-row validation. |

The permissible high-level claim is: at a matched budget of five positive and
three negative prompts, the deterministic spatially diversified action has a
positive paired effect over Top-5 across the five completed MoNuSAC
replications.  This is not a cross-domain, clinical, state-of-the-art, or
trained-medical-segmenter claim.

## Prompt-geometry authority

`docs/2026-07-16-prompt-geometry-analysis.md` provides the mechanism check.
For DINOv2-L + SAM-B / 20260716, mean positive nearest-neighbour distance is
7.460 patches for Top-5 and 9.673 for spread, a paired change of +2.213
patches with CI [+1.861, +2.579].  The three additional completed geometry
checks have positive paired changes: +2.133, +2.443, and +2.213 patches.  The
manuscript may use these results only to show that the rule changes point
layout at equal prompt budget; they do not select a radius or demonstrate a
universal optimum.

## Clean adaptive-policy authority: external negative

`artifacts/stage17_clean_adaptive_layout/summary.json` records a
clean-development adaptive-layout policy whose external IDRiD action set was
locked before the once-read test.  On the 81 IDRiD reference-query episodes,
the fixed rules score 0.0958 (`top5_n3`), 0.0917 (`spread5_n3_d2`), 0.0533
(`spread5_n3_d3`), and 0.0425 (`spread5_n3_d4`); the frozen selector scores
0.0725.  It is an external negative result.  The manuscript must not retune
the selector against IDRiD or describe it as transferable.

## Qualitative authority

`artifacts/stage17_clean_adaptive_layout/qualitative_manifest.json` locked
queries `IDRiD_68`, `IDRiD_75`, and `IDRiD_81` before evaluation.
`artifacts/stage17_clean_adaptive_layout/qualitative_rendered/render_summary.json`
lists their rendered PNGs.  Each panel shows all three fixed IDRiD references
(`IDRiD_20`, `IDRiD_47`, `IDRiD_50`), query ground truth, matched Top-5, and
the frozen policy action.  The renderer uses no Dice or utility for selection;
the qualitative figure cannot be used as an outcome-selected superiority
claim.

## Exclusions

- Do not include the stale MoNuSeg cross-dataset table as a main result.
- Do not include the old policy-oracle number, old policy means, or a claim
  that adaptive layout selection is the method contribution.
- Do not claim state-of-the-art performance, clinical utility, cross-domain
  transfer, or superiority to trained medical segmenters.
