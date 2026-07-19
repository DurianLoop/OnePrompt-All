# GlaS Phase 3 Gate R1 Region Proposal Summary

Status: `STOP_GLAS_REGION_PROPOSAL_FAILURE`

| Family | Route | Metric | Improvement | Win rate | Recall | Empty | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| R00_direct_matching_mask | mask | `0.178754` | `-0.416321` | `0.066667` | `0.099628` | `0.000000` | `FAIL` |
| R03_automatic_union_box | box | `0.744785` | `0.025965` | `0.200000` | `0.790664` | `0.000000` | `FAIL` |
| R04_automatic_largest_component_box | box | `0.019410` | `0.002199` | `0.100000` | `0.029807` | `0.000000` | `FAIL` |
| R05_automatic_top3_component_boxes | box | `0.271747` | `0.016568` | `0.166667` | `0.063224` | `0.000000` | `FAIL` |

No SAM inference was run for Gate R1. GlaS testA/testB were not accessed.
