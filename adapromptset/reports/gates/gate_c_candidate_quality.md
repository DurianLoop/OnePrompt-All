# Gate C: Candidate Quality

Status: `PASS`
Pass/fail: `PASS`

## Rule Applied

Positive candidates must beat a support-matched random baseline and candidate validation must report coordinate/blank-background issues.

## Diagnostics

| Field | Value |
|---|---|
| mean_precision | `None` |
| mean_random_precision | `0.16` |
| mean_component_coverage | `0.052004904966278354` |
| glas_status | `PASS` |

## Commands

- `scripts/run_candidate_pilot.py`

## Input Hashes

- `reports/candidate_pilot/candidate_pilot_summary.json`: `3F9214C2AECC9DC9D45FD1BFABA475A8B72840E0D6A6CDE0090FF1901F29CF58`
- `reports/candidate_pilot_glas/candidate_pilot_summary.json`: `B965D979182B23D0185CC2F5BB509227B58A02CDBB0FCEE39C16781392156562`
- `reports/gate_c_status.md`: `FE73188BD2AE40C89B318EEE5C2371EF4FAA6C038AC22671B63A29EC649943E2`

## Allowed Next Actions

- `segmenter_execution_gate`

## Prohibited Next Actions


## Deviations

- GlaS evidence is a strict one-episode extension smoke, not a formal GlaS Gate F claim.
