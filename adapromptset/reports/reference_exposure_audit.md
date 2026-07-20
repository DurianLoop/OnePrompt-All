# Reference Exposure Audit

Status: `PASS`

This audit verifies that the reference pool is train-only and that reference-test leakage is zero.

| Dataset | Episodes | Support cases | Test cases | Test patients | Status |
|---|---:|---:|---:|---:|---|
| GlaS | 88 | 7 | 80 | 12 | PASS |
| MoNuSAC | 1505 | 162 | 101 | 25 | PASS |
| MoNuSeg | 51 | 23 | 14 | 14 | PASS |

## Checks

- reference pool train only: `PASS`
- reference test leakage zero: `PASS`

## Issues

- none
