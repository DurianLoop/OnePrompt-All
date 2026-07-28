| Stage | SAM calls | GPU-hours | Runtime (s) | Device | Description |
|---|---|---|---|---|---|
| R0 audit | 0 | 0.00 | — | cpu | Read-only historical audit (8 files) |
| Gate R1 | 7200 | 0.13935 | 501.7 | cuda | 80 queries x 5 seeds x 6 refs x 3 configs |
| Gate R2 | 3138 | 0.07546 | 271.7 | cuda | 80 train + 41 val consensus evaluation |
| TOTAL | 10338 | 0.2153 | — | — | Of 85 GPU-hour cap; remaining 84.7847 |

Source: phaseR_compute_ledger.csv, run_metadata.json. Total SAM calls = 7200 + 3138 = 10338.
