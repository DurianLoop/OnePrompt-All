# Prompt Geometry Analysis

## Measure

For every five-positive-prompt episode, compute the mean nearest-neighbour
distance among selected positive points on the frozen 37 x 37 DINO patch grid.
This is action-derived and needs no ground-truth label at inference.

## Existing held-out evidence

| Matcher / segmenter | Reference seed | Rule | Episodes | Mean positive nearest-neighbour distance (patches) | Mean Dice |
|---|---:|---|---:|---:|---:|
| DINOv2-L / SAM-B | 20260716 | Top-5 + 3 negative | 303 | 7.460 | 0.4196 |
| DINOv2-L / SAM-B | 20260716 | Spread radius 3 + 3 negative | 303 | 9.673 | 0.4526 |
| DINOv2-L / SAM-B | 20260717 | Top-5 + 3 negative | 303 | 7.865 | 0.4008 |
| DINOv2-L / SAM-B | 20260717 | Spread radius 3 + 3 negative | 303 | 9.998 | 0.4362 |
| DINOv2-B / SAM-B | 20260716 | Top-5 + 3 negative | 303 | 6.757 | 0.3835 |
| DINOv2-B / SAM-B | 20260716 | Spread radius 3 + 3 negative | 303 | 9.200 | 0.4344 |
| DINOv2-L / SAM-L | 20260716 | Top-5 + 3 negative | 303 | 7.460 | 0.4507 |
| DINOv2-L / SAM-L | 20260716 | Spread radius 3 + 3 negative | 303 | 9.673 | 0.4659 |

## Paired geometry effect

Treating each episode's mean nearest-neighbour distance as the paired outcome
and applying the same fixed 10,000-resample bootstrap (seed 20260719) gives:

| Matcher / segmenter / reference seed | Spread minus Top-5 (patches) | 95% paired bootstrap interval | Win / tie / loss |
|---|---:|---:|---:|
| DINOv2-L / SAM-B / 20260716 | +2.213 | [+1.861, +2.579] | 175 / 110 / 18 |
| DINOv2-L / SAM-B / 20260717 | +2.133 | [+1.814, +2.461] | 173 / 116 / 14 |
| DINOv2-B / SAM-B / 20260716 | +2.443 | [+2.113, +2.778] | 192 / 100 / 11 |
| DINOv2-L / SAM-L / 20260716 | +2.213 | [+1.861, +2.579] | 175 / 110 / 18 |

These are geometry checks only: they confirm that the fixed rule actually
changes the prescribed mechanism at equal prompt budget.  They neither choose
a radius nor add a new test-time policy.

## Interpretation

The spatial rule changes prompt geometry substantially while retaining the
same number of positive and negative prompts.  Its positive Dice gains
therefore cannot be attributed to a larger prompt budget.  This is a
mechanistic observation, not proof that larger separation is always better:
the radius sweep was exploratory, and MoNuSeg did not show a meaningful gain
with the fixed radius-3 setting.

The DINOv2-L/SAM-L geometry is exactly the DINOv2-L/SAM-B geometry for the
same reference seed because matching, candidate ranking, and point selection
are frozen before the SAM segmenter is invoked.  The distinct Dice values in
that pair therefore test segmenter compatibility without changing prompt
geometry.  The pending SAM-H run follows the same fixed separation rule and
will be added only after its 303-episode rows pass the balanced-row check.

## Planned qualitative figure

To avoid outcome-selected examples, the final qualitative figure will use
three query IDs fixed by the clean protocol seed.  For each, it will show the
reference mask, matching contrast map, Top-K prompts, spread prompts, their
SAM masks, and the query ground truth.  Prompt coordinates and masks will be
saved with the figure-generation manifest so panels can be regenerated.
