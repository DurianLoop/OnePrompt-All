# Research Program Status

Date: 2026-07-19

Branch: `phase4plus-submission-program`

Phase 4+ preregistration commit: `b4ee8e2`

Top-level state: `PRIMARY_TRACK_B_SELECTED`

Current development gate: `PASS_DDR_ROUTE_B_DEV_MANIFEST`

Current prereq freeze: `PASS_ROUTE_B_B1_PREREQ_FREEZE`

Current episode gate: `PASS_DDR_ROUTE_B_EPISODES`

Current development smoke: `PASS_ROUTE_B_DDR_MATCHER_SMOKE`

Current stratified development evidence: `PASS_ROUTE_B_DDR_STRATIFIED_MATCHER_SMOKE`

Current failure taxonomy: `PASS_ROUTE_B_DDR_FAILURE_TAXONOMY`

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

`TRACK_B_BENCHMARK_READY` or `TRACK_B_INSUFFICIENT_BREADTH`, after multi-dataset benchmark protocol, legal development evidence, and required breadth checks.

Latest legal progress:

- DDR train/valid manifest gate passed under Route B.
- DDR train: 383 images, 964 positive image-intent queries.
- DDR valid: 149 images, 401 positive image-intent queries.
- DDR test remains locked; no test files were read.
- Route B metric/control config is frozen for development-only evidence generation.
- DDR Route B episodes: train 2892 reference rows for 964 positive queries; valid 1203 reference rows for 401 positive queries.
- One DDR valid MA matcher smoke ran with train support under DINOv2-L only; point precision and component coverage were both 0.0, so this is a pipeline/failure-taxonomy signal rather than positive benchmark evidence.
- Stratified DDR DINOv2-L matcher evidence covered 16 train/valid query-intent units, two per split/intent. Mean point precision was 0.0125 and zero precision fraction was 0.9375.
- Control-aware DDR failure taxonomy records `small_sparse_target=16/16`, `diffuse_matching_map=16/16`, `no_candidate_hit=15/16`, `no_component_covered=15/16`, and `random_not_beaten=15/16`.

## Current Prohibitions

- No new model inference before preregistration commit.
- No formal test access.
- No same-validation action/grid expansion for GlaS R1.
- No MoNuSAC policy training.
- No submission-ready claim.
- No DDR formal claim or DDR test tuning.
- No Route B method-positive claim from the B1 prereq freeze alone.
- No independent-sample claim from reference rows or image-intent rows.
- No method-positive claim from the one-episode DDR matcher smoke.
- No direct expansion of the same candidate-only DINOv2 matcher smoke as if it were a promising method route; the next step must be failure-taxonomy/control-aware benchmark analysis.
- No benchmark-ready claim until prompt-interface controls, broader datasets, model/representation combinations, and statistics exist.
