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

## Phase X final verdict (2026-07-20)

The `FINITE_BANK_ORACLE_GAP_SMALL` claim has been **refuted** by Phase X.
The historical 0.005 gap was an artifact of an inadequate action space
(max K+=5, max K-=3, max calls=5), not evidence of action-space
saturation.

Phase X findings (all on the same 10 MoNuSAC validation episodes):

| Stage | Finding | Mean Dice |
|---|---|---|
| Historical best fixed (B_top5_pos3neg) | FINITE_BANK_ORACLE_GAP_SMALL | 0.3712 |
| Historical finite-bank oracle | (old 12-action bank) | 0.3762 |
| Stage S2 K=20 (dense prompt) | DENSE_PROMPT_VIABILITY_ESTABLISHED | 0.5237 |
| Stage S4 K-=1 N_BOTTOMK | NEGATIVE_BEST_BEATS_NONE | 0.5411 |
| A_large best fixed (Gate X4) | OLD_ACTION_BANK_INADEQUATE | 0.5411 |
| A_large finite-bank oracle (Gate X4) | 91-action bank | 0.6119 |
| **Search oracle (Gate X5)** | SEARCH_ORACLE_EXCEEDS_BANK_ORACLE | **0.7256** |

The search oracle (0.7256) exceeds the historical finite-bank oracle
(0.3762) by **0.3494** — the historical "oracle" was capturing only 52%
of the achievable performance. The 0.005 gap was measuring saturation
within an inadequate action space, not true action-space saturation.

**Final Gate X6 verdict**: `REOPEN_POLICY_LEARNING_WITH_EXPANDED_ACTION_SPACE`.
All six conditions met. Policy learning may resume with the expanded
action space (A_large, 91 actions), using hierarchical prediction per
section 18 of the directive. PPO/DQN remain prohibited.

See `reports/phaseX_final_report.md` for the complete audit.
