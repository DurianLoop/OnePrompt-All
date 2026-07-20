# Gate B: Protocol and Exposure Integrity

Status: `PASS`
Pass/fail: `PASS`

## Rule Applied

Reference-query episodes must be deterministic, versioned, and free of test-support leakage.

## Diagnostics

| Field | Value |
|---|---|
| status_table_present | `True` |
| contains_fail | `False` |

## Commands

- `scripts/build_episodes.py`

## Input Hashes

- `reports/gate_b_status.md`: `0DAF8E02633DDDDDF8E2F366B32BE8AF3B3606507D9E893499BCE2065554B043`

## Allowed Next Actions

- `candidate_quality_gate`

## Prohibited Next Actions


## Deviations

- none
