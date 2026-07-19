# Research Program Status

Date: 2026-07-19

Branch: `phase4plus-submission-program`

Phase 4+ preregistration commit: `b4ee8e2`

Top-level state: `PRIMARY_TRACK_B_SELECTED`

## Frozen Prior States

| Route | State |
| --- | --- |
| MoNuSAC | `PIVOT_ORACLE_ANALYSIS` |
| GlaS grouped sparse point | `STOP_GLAS_GROUPED_POINT_PROMPT_PATH_CONFIRMED` |
| GlaS automatic region | `STOP_GLAS_REGION_LOCALIZATION_ADVANTAGE_NOT_ESTABLISHED` |
| GlaS operational Phase 3 | `STOP_GLAS_REGION_PROPOSAL_FAILURE` |
| GlaS testA/testB | `LOCKED_UNUSED` |

## Current Phase

Phase 4 read-only audit is complete and supports the benchmark/diagnostic track.

Selected primary track:

`Route B -- Reference Promptability Benchmark`

Selected fallback track:

`Route A -- Promptability-Aware Routing`

Deferred track:

`Route C -- Direct Reference Segmentation`

Next legal state:

`TRACK_B_BENCHMARK_READY` or `TRACK_B_INSUFFICIENT_BREADTH`, after fresh-data benchmark protocol and legal development evidence.

## Current Prohibitions

- No new model inference before preregistration commit.
- No formal test access.
- No same-validation action/grid expansion for GlaS R1.
- No MoNuSAC policy training.
- No submission-ready claim.
