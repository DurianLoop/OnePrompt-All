# Gate D: Segmenter Execution

Status: `PASS`
Pass/fail: `PASS`

## Rule Applied

Privileged prompt diagnostics must verify SAM-B wrapper correctness, labels, call accounting, and deterministic repeated prompts.

## Diagnostics

| Field | Value |
|---|---|
| component_dice | `None` |
| semantic_union_dice | `None` |
| call_count | `None` |
| prompt_group_count | `None` |

## Commands

- `scripts/run_sam_smoke.py`

## Input Hashes

- `reports/sam_smoke/sam_smoke_summary.json`: `98C90D5ACA4A976430D2816B56AAF5AC0D16080C40F1B3C3200F9C7D47D63FB5`
- `reports/gate_d_status.md`: `2EEE7E41518C65603ADDCE56431908CC5B6C0E5F899E34355F507962FF9BF887`

## Allowed Next Actions

- `action_bank_validity_gate`

## Prohibited Next Actions


## Deviations

- Semantic-union Dice is diagnostic for one-point prompts and is not used as the wrapper pass criterion.
