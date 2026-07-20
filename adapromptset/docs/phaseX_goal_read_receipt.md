# Phase X Goal Read Receipt

Date: 2026-07-20

## Source directive

`D:\SAM\adapromptset\AdaPromptSet_ActionSpace_Oracle_Retry.md`

## Goal

Phase X — Action-Space Adequacy, Dense Prompt Scaling, and Search-Oracle Audit.

## Core scientific question (section 3)

Given the frozen matcher, candidate generator, segmenter, and reference-query
protocol, is the historical 12-action bank sufficient to approximate the prompt
search space? Concretely: for dense, spatially-dispersed segmentation queries,
does optimal prompt performance keep rising as positive count, negative count,
spatial grouping, and SAM-call budget scale up? And in an already-saturated
action space, does query-specific Oracle still exceed the strongest fixed action
by enough to justify policy learning?

## Mandatory corrections acknowledged (section 0)

The historical "Oracle gap = 0.005" claim is renamed to
`FINITE_BANK_ORACLE_GAP_SMALL`. It is *not* a Global Prompt Oracle and *not* a
saturation proof. It only states that within the historical 12-action bank, the
per-query best action beats the strongest fixed action by a small margin.

## Strict ordering acknowledged (section 25)

1. Read historical action-bank/oracle artifacts.
2. Create historical Oracle correction doc.
3. Export old 12-action exact definitions.
4. Export per-action absolute Dice.
5. Pinpoint old strongest fixed exact config.
6. Freeze candidate scaling protocol.
7. Freeze dense prompt scaling protocol.
8. Freeze search oracle protocol.
9. Freeze K/negative/grouping/call grids.
10. Generate config/input hashes.
11. Run focused tests.
12. Preregistration commit.

No new SAM inference runs before the preregistration commit. No formal test
access. No test-driven Top-K, cluster count, or budget changes. MoNuSAC
development/validation scope only.

## Frozen historical anchors

- Historical v2 bank: `configs/actions/prompt_set_bank_v2.json`
  - bank_hash: `35E44B3B937A9D6AA9E969CDAFC9AC6F45A77E5BA1EF72BC6770E4B5F16D75B8`
  - action_count: 13
  - valid_action_count: 12
- Historical best fixed action: `B_top5_pos3neg`
- Historical finite-bank oracle mean Dice: 0.376250
- Historical strongest fixed mean Dice: 0.371238
- Historical finite-bank oracle gap: 0.005011
- Historical Phase 5 status: `PIVOT_ORACLE_ANALYSIS`
- MoNuSAC validation episodes used historically: 10
