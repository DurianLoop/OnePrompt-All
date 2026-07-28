# Route B DDR Development Manifest Report

Status: `PASS_DDR_ROUTE_B_DEV_MANIFEST`

Route: `B_reference_promptability_benchmark`

Formal test access: `false`
DDR test files read: `false`

## Split Summary

| Split | Images | All intent rows | Positive queries | EX+ | HE+ | MA+ | SE+ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 383 | 1532 | 964 | 245 | 294 | 314 | 111 |
| valid | 149 | 596 | 401 | 70 | 113 | 132 | 86 |

## Boundaries

- Positive-query manifests include only non-empty intent masks.
- All-intent manifests retain empty masks for prevalence accounting.
- DDR test remains locked and unused.
- Image-intent rows are repeated task views of the same image and are not independent patient units.

## Outputs

- `train_all_intents`: `D:\SAM\adapromptset\manifests\phase4plus\ddr_route_b_train_all_intents.csv`
- `train_positive_queries`: `D:\SAM\adapromptset\manifests\phase4plus\ddr_route_b_train_positive_queries.csv`
- `valid_all_intents`: `D:\SAM\adapromptset\manifests\phase4plus\ddr_route_b_valid_all_intents.csv`
- `valid_positive_queries`: `D:\SAM\adapromptset\manifests\phase4plus\ddr_route_b_valid_positive_queries.csv`
