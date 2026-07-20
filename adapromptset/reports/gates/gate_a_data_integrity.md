# Gate A: Data Integrity

Status: `PASS`
Pass/fail: `PASS`

## Rule Applied

All formal cases require verified image-mask pairs, hashes, and zero strict train/test identity overlap.

## Diagnostics

| Field | Value |
|---|---|
| dataset_count | `3` |
| datasets | `["GlaS", "MoNuSAC", "MoNuSeg"]` |
| issues | `{"GlaS": [], "MoNuSAC": [], "MoNuSeg": []}` |

## Commands

- `scripts/audit_data.py --config configs/experiments/e0_data_audit.yaml`

## Input Hashes

- `reports/dataset_audit/glas.json`: `46335899AD5F3417C7B61AC43D3A525D68F3391C2BFF2C1E912C2FDA480887E4`
- `reports/dataset_audit/monusac.json`: `6AD35B0607EB0042AC61495B84FC64BC8975E964F822F288309B4BB1C3C60248`
- `reports/dataset_audit/monuseg.json`: `54B442884BE5A7F4D24ED997EF0B7BC5370E74F8822BD61A8B244766040DB41C`
- `reports/gate_status.md`: `A977087F33671D2DDE26131F4A86DB882879B41532103A888604F7CCE62106F0`

## Allowed Next Actions

- `gate_b_protocol_exposure_audit`

## Prohibited Next Actions


## Deviations

- none
