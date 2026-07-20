# Stage 2 Plan: Fixed Prompt Baselines under a Leakage-Safe Protocol

## Goal

Establish reproducible fixed-rule baselines for the proposed adaptive prompt
policy.  These measurements are pilot evidence for the method development
cycle; they do not constitute a cross-dataset paper result.

## Data protocol

Use the official MoNuSeg split discovered in Stage 1:

- reference, validation and configuration selection use only the 37 official
  training cases;
- one reference case and six validation cases are selected deterministically
  from a seeded permutation of the training cases;
- all 14 official test cases remain untouched until reporting;
- no test mask, score, or qualitative result is used to choose a prompt rule.

The exact case IDs, seed, candidate rules and aggregate scores will be written
alongside the results.

## Prespecified fixed rules

| Name | Positive points | Negative points | Background weight |
|---|---:|---:|---:|
| `top1_n0` | 1 | 0 | 1.0 |
| `top3_n0` | 3 | 0 | 1.0 |
| `top5_n0` | 5 | 0 | 1.0 |
| `top5_n3` | 5 | 3 | 1.0 |

All prompt coordinates are ranked from the same contrast-map candidate pools.
The primary fixed baseline is selected by mean validation Dice; ties favour a
lower total prompt count and then lexical rule name.  Every prespecified rule
is also reported on the official test split, clearly labelled as a fixed-rule
baseline rather than a tuned result.

## Implementation tasks

1. Add a deterministic split/selection protocol with unit tests.
2. Add a model-cached evaluator that computes candidate pools and SAM masks
   for all fixed rules, writing case-level CSV and aggregate JSON artifacts.
3. Add a one-command runner with complete provenance: checkpoints, device,
   protocol seed, case IDs, rules, metrics, and elapsed time.
4. Run unit tests, then execute the official MoNuSeg pilot on the locally
   available frozen DINOv2-L/14 and SAM ViT-B checkpoints.

## Advancement criterion

Proceed to adaptive-policy labels once the runner completes and its artifacts
show: disjoint train-only reference/validation partitions, all 14 test cases,
and finite metrics for every prespecified rule.  A weak score is still useful
evidence; it triggers diagnostic work rather than test-set retuning.
