# Phase X Formal Test Lock Notice

Date: 2026-07-20

## Inheritance

Phase X inherits the formal-test locks recorded in
`docs/formal_test_lock_registry.md`:

| Formal test | Phase X status | Required before access |
|---|---|---|
| GlaS testA | `LOCKED_UNUSED` | `docs/test_freeze_glas.md`, frozen final config, hashes, code commit, metrics, qualitative sampling, cost/statistics plan |
| GlaS testB | `LOCKED_UNUSED` | testA report complete with no method changes; same frozen method |
| DDR test | `LOCKED_UNUSED_CANDIDATE_FRESH_DOMAIN` | DDR four-intent protocol freeze, train/valid-only selection, frozen final method, hashes |
| MoNuSeg formal use | `NOT_FORMAL_BY_DEFAULT` | re-audit official split/patient identity and define role |
| IDRID | `READ_ONLY_NON_TUNABLE` | proof of unobserved subset if any formal claim is desired |

## Phase X prohibitions

Per section 6 of the Phase X directive, the following are prohibited this round:

- Using formal test to select K, cluster count, or budget.
- Using GlaS testA/testB.
- Using IDRID for tuning.
- Repackaging observed test as validation.
- Adding Top-K candidates based on formal-test results.
- Changing cluster count or budget based on formal-test results.

## Scope

Phase X is a **development-stage action-space audit** on MoNuSAC
train/validation only. It is not a formal test paper result. All queries,
references, and splits must use the frozen MoNuSAC manifests.

## Block condition

If the historical manifest is incomplete or unreproducible:

```text
PHASEX_BLOCKED_MANIFEST_INCONSISTENCY
```

and the round stops.

## Current manifest anchor

- MoNuSAC validation episodes: `D:\SAM\adapromptset\manifests\episodes\monusac\episodes_val.csv`
- Historical v2 oracle pilot episodes: 10
- Reference exposure audit historical status: `PASS` (per
  `reports/reference_exposure_audit.json`).
