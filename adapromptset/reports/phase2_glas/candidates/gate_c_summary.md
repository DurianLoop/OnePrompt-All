# GlaS Phase 2 Gate C Candidate Diagnostics

Status: `FAIL`
Decision: `REDESIGN_GLAS_CANDIDATES_ONCE_ALLOWED`

## Core Metrics

- Validation queries: `30`
- Reference episodes: `90`
- Max positive limit: `5`
- Mean point precision: `0.971111`
- Mean random precision: `0.548889`
- Precision delta: `0.422222`
- Precision ratio: `1.769231`
- Mean component coverage: `0.418820`
- Mean negative blank fraction: `0.135556`

## Checks

- `expected_query_count`: `PASS`
- `expected_reference_episode_count`: `PASS`
- `precision_enrichment`: `PASS`
- `component_coverage`: `FAIL`
- `coverage_curve_nondecreasing`: `PASS`
- `negative_pool_not_easy_blank`: `PASS`
- `coordinate_validation`: `PASS`

## Issues

- component coverage 0.4188 < required 0.6000

## Figures

- `paper_phase2/figures/candidate_diagnostics/precision_vs_pool.png` SHA256 `4072FAFE6BC07DE02A734FF37F3A91CC79C47FCA846B4ED5F9B2E8F16584A831`
- `paper_phase2/figures/candidate_diagnostics/coverage_vs_pool.png` SHA256 `8895F76801D10384A954D764C6466A42EDC0E4444BC425DEE036B9A2522C0C42`
- `paper_phase2/figures/candidate_diagnostics/negative_blank_vs_pool.png` SHA256 `7AF0B94D124F6A1750A3FA4F0C3E0C0EEC1051C7A8C50266EED571F72DA99A24`
