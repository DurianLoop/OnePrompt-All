# GT-Guided Search Oracle Protocol (Phase X, Section 13)

Date: 2026-07-20

## Purpose

The historical finite-bank oracle cannot establish action-space adequacy. A
GT-guided search oracle is therefore built on the legal train/validation audit
subset. It is **diagnostic only**: it must not be deployed, must not become an
automatic action, and must not influence the formal test.

## Search variants

### 13.1 Greedy Add

1. Start from a safe anchor (e.g. the strongest fixed historical action).
2. For each remaining positive candidate, evaluate the Dice of adding it.
3. Add the candidate with the largest Dice gain.
4. Repeat until no improvement or the point/call budget is exhausted.
5. Then search negatives in the same greedy manner.
6. Then search grouping changes (merge/split/reassign).

### 13.2 Greedy Add/Remove/Swap

Operations allowed on the current state:
- add positive, remove positive, swap positive
- add negative, remove negative, swap negative
- merge groups, split groups, change group assignment

### 13.3 Beam Search

- beam_width in {4, 8}
- max depth in {10, 20, 50}
- retain multiple high-Dice candidate states within the budget.

## Search budgets (section 13.4)

- B_call in {1, 3, 5, 10, 20}
- B_point in {10, 20, 50, 100}

## Comparisons (section 13.5)

Report:

- `D_search_oracle(B)` - mean Dice of the search oracle at budget B.
- `D_search_oracle(B) - D_bank_oracle(B)` - hidden capacity beyond finite bank.
- `D_search_oracle(B) - D_fixed(B)` - adaptive value over strongest fixed.

## Interpretation rules (section 13)

- If `D_search_oracle - D_bank_oracle >= 0.01`:
  `OLD_ACTION_BANK_INADEQUATE`.
- If `D_search_oracle - D_fixed >= 0.01` and best configs differ across
  queries: `POLICY_SPACE_POTENTIALLY_MEANINGFUL`.
- If `D_search_oracle` itself is still low in absolute terms:
  `PROMPT_INTERFACE_OR_CANDIDATE_LIMITATION`.

## Three Oracles (section 14)

Reports must distinguish:

1. **Finite-Bank Oracle** - per-query best within the frozen action bank.
2. **Search Oracle** - GT-guided greedy/beam search over the frozen candidate
   pool and budget.
3. **Privileged Coordinate Oracle** - upper bound using GT
   component/centroid/box prompts.

The unqualified word "Oracle" is prohibited.
