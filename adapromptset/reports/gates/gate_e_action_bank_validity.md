# Gate E: Action-Bank Validity

Status: `PASS_RESCUE_SUBSET`
Pass/fail: `PASS`

## Rule Applied

Actions must be predeclared, versioned, hashable, cost-accounted, and built from the shared candidate pool.

## Diagnostics

| Field | Value |
|---|---|
| bank_name | `prompt_set_bank_v2` |
| bank_hash | `35E44B3B937A9D6AA9E969CDAFC9AC6F45A77E5BA1EF72BC6770E4B5F16D75B8` |
| action_count | `13` |
| valid_action_count | `12` |

## Commands

- `scripts/build_action_bank.py --bank-version v2`

## Input Hashes

- `reports/action_bank/action_bank_summary.json`: `9A04120FBD18A77DA617CD61E14FC7EFC7BCDA6009AC8BF156FF10178BBA5ADE`
- `reports/gate_e_status.md`: `CF089B0E266BAC67E6A65E33B56E6088B404A9F671D32E23081E08643A28B46F`

## Allowed Next Actions

- `oracle_gap_gate`

## Prohibited Next Actions

- `post_hoc_action_bank_redesign_after_gate_f`

## Deviations

- Current bank is a rescue subset; full KMeans/SynPo/GF-SAM/safe-anchor matrix remains incomplete.
