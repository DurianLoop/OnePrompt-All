# Route B DDR Episode Report

Status: `PASS_DDR_ROUTE_B_EPISODES`

References per query: `3`
Formal test access: `false`
DDR test files read: `false`

## Split Summary

| Split | Positive queries | Covered queries | Episode rows | References/query |
| --- | ---: | ---: | ---: | ---: |
| train | 964 | 964 | 2892 | 3 |
| valid | 401 | 401 | 1203 | 3 |

## Boundary

- Valid references come only from DDR train positive-query rows.
- Train references exclude the query image.
- DDR test remains locked and unused.
- This report creates development input manifests only.

## Outputs

- `train_episodes`: `D:\SAM\adapromptset\manifests\phase4plus\ddr_route_b_train_reference_query_episodes.csv`
- `valid_episodes`: `D:\SAM\adapromptset\manifests\phase4plus\ddr_route_b_valid_reference_query_episodes.csv`
