# Phase X Historical Claim Correction

Date: 2026-07-20

## Old wording (prohibited from now on)

> Oracle gap = 0.005 proves adaptive prompting is ineffective on MoNuSAC.

## New wording (required)

> Within the historical 12-action bank on MoNuSAC validation, the per-query
> best action beats the strongest fixed action by 0.005011 mean Dice. Both
> configurations have low absolute performance (mean Dice ~0.3712). This
> `FINITE_BANK_ORACLE_GAP_SMALL` result does not establish action-space
> saturation: the bank never tested K+ >= 10, K- >= 10, dense kmeans/grid/
> graph/component groupings, or call budgets above 5.

## Anchor numbers

- mean best fixed Dice: 0.371238
- mean finite-bank oracle Dice: 0.376250
- gap: 0.005011
- AJI gap: 0.003820
- best fixed action: `B_top5_pos3neg`
- v2 bank hash: `35E44B3B937A9D6AA9E969CDAFC9AC6F45A77E5BA1EF72BC6770E4B5F16D75B8`
- historical Phase 5 status: `PIVOT_ORACLE_ANALYSIS`

## What is *not* allowed

- Abbreviating the gap as "Oracle gap" without the `FINITE_BANK` qualifier.
- Using the small gap to claim adaptive prompting has no value.
- Using the small gap to justify skipping the action-space saturation audit.
- Lowering the 0.01 adaptive-value threshold after seeing results.

## Cross-references

- `docs/historical_oracle_correction.md`
- `docs/action_space_problem_statement.md`
- `reports/phaseX_historical_bank_audit/old_bank_oracle_summary.md`
