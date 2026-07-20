# Spatial Layout Actions for the Prompt Policy

## Objective

Extend the current count-only prompt action space with a deterministic spatial
layout action.  The extension tests the design-document claim that prompt
layout, rather than only the number of points, should be selected adaptively.

## Action set

All actions use the same frozen DINO contrast map and SAM predictor:

| Action | Positive layout | Positive / negative points |
|---|---|---:|
| `top5_n3` | Highest contrast patches | 5 / 3 |
| `spread5_n3` | High contrast patches, spatially separated | 5 / 3 |

`spread5_n3` uses deterministic non-maximum suppression in patch-grid space:
rank patches by descending contrast, accept the first candidate, and accept
each later candidate only if its Euclidean grid distance from every accepted
positive point is at least three patches.  If insufficient points survive,
fill remaining slots from the ranked list.  Negative points retain the
existing lowest-contrast rule and never duplicate a positive coordinate.

This is a reproducible layout operation, not stochastic KMeans/DBSCAN; it
does not add a dependency or a test-time learned clustering parameter.

## Policy features

The utility policy receives two new action-conditioned features:

- mean nearest-neighbour distance among selected positive patches;
- positive layout mode encoded as `0` for top-K and `1` for spread.

It still observes no ground-truth mask at inference.  Offline Dice remains
the train/validation-only utility target.

## Experimental protocol

Use the already audited MoNuSAC multi-reference partition.  First run a
bounded feasibility pass with three train references, 12 train queries and
six validation queries.  It evaluates both layout actions for every episode,
fits only on train episode/action rows, and selects policy complexity only on
validation episodes.  The 101 test cases remain unread.

Advance to a larger run only if the layout action wins a nontrivial number of
validation episodes and the policy improves validation Dice over fixed
`top5_n3` by more than the paired standard error.
