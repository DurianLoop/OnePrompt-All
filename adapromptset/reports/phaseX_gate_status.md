# Phase X Gate Status

Date: 2026-07-20

Phase: preregistration (no new SAM inference yet).

## Gate X1 — Historical Baseline Audit

Status: `PENDING_PREREGISTRATION_COMMIT`

Through conditions:

- Old 12-action exact definitions recoverable. -> DONE (see
  `reports/phaseX_historical_bank_audit/action_bank_exact_definitions.csv`).
- Strongest fixed exact config recoverable. -> DONE
  (`B_top5_pos3neg`).
- All absolute Dice recoverable. -> DONE.
- Cost metadata recoverable. -> DONE.

Failure code if any artifact is missing: `BLOCKED_HISTORICAL_ARTIFACT_INCOMPLETE`.

## Gate X2 — Candidate Scaling

Status: `NOT_RUN_YET`.

Will report one of:

- `CANDIDATE_SCALING_SUPPORTS_LARGE_K`
- `CANDIDATE_SCALING_SATURATES_SMALL_K`
- `CANDIDATE_UNIVERSE_INADEQUATE`

## Gate X3 — Dense Prompt Viability

Status: `NOT_RUN_YET`.

Will report one of:

- `DENSE_PROMPT_VIABILITY_ESTABLISHED`
- `DENSE_PROMPT_NO_GAIN`
- `DENSE_PROMPT_COST_ONLY_GAIN`

## Gate X4 — Action-Space Saturation

Status: `NOT_RUN_YET`.

Will report one of:

- `ACTION_SPACE_NOT_SATURATED`
- `ACTION_SPACE_SATURATED`
- `OLD_ACTION_BANK_INADEQUATE`

## Gate X5 — Search-Oracle Adequacy

Status: `NOT_RUN_YET`.

Will report one of:

- `SEARCH_ORACLE_EXCEEDS_BANK_ORACLE`
- `SEARCH_ORACLE_CLOSE_TO_BANK_ORACLE`
- `SEARCH_ORACLE_ABSOLUTE_PERFORMANCE_LOW`

## Gate X6 — Adaptive Value

Status: `NOT_RUN_YET`.

Reopening policy requires *all* of:

1. `D_search_oracle - D_fixed >= 0.01` Dice
2. At least 25% of queries have a
   best structure different from the strongest fixed.
3. The gain is not solely from more calls.
4. Absolute performance reaches a usable level.
5. Action-space saturation is essentially established.
6. Query-specific best config has a predictable relation to morphology /
   candidate statistics.

Pass: `REOPEN_POLICY_LEARNING_WITH_EXPANDED_ACTION_SPACE`.

Fail (one of): `PIVOT_STRONG_FIXED_DENSE_PROMPT`,
`PIVOT_PROMPT_INTERFACE_LIMITATION`, `PIVOT_CANDIDATE_LIMITATION`,
`PIVOT_ORACLE_ANALYSIS_EXPANDED_SPACE`.
