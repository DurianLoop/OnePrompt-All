# Route B DDR Failure Taxonomy

Status: `PASS_ROUTE_B_DDR_FAILURE_TAXONOMY`

Rows: `16`
Formal test access: `false`
DDR test files read: `false`

## Aggregate

- Mean DINOv2 point precision: `0.0125`
- Mean random point precision: `0.0`
- Mean DINOv2 component coverage: `0.0025`
- Mean random component coverage: `0.0`

## Failure Flags

- `diffuse_matching_map`: `16`
- `low_score_margin`: `10`
- `no_candidate_hit`: `15`
- `no_component_covered`: `15`
- `random_not_beaten`: `15`
- `small_sparse_target`: `16`

## Boundary

- This is a control-aware development taxonomy, not a formal benchmark result.
- Query GT is used only for evaluation and failure taxonomy labels.
- DDR test remains locked.
