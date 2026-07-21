# Phase Y — Point-Subset Policy Protocol (Gate Y4)

Date: 2026-07-21

Directive section: 11, 12.

## 1. Track

Track B: Point-Subset Policy learns to select a specific subset of candidate
points (not just K+/K-/selector). Supervision comes from the GT-guided
Search Oracle (Gate Y2 stable labels).

## 2. Scope

- Train on Policy Train Queries (80) with search-oracle labels.
- Validate on Policy Validation Queries (41).
- Holdout: single frozen run (40).
- Search oracle runs on train + validation ONLY (section 9.1). Holdout and
  test are FORBIDDEN for search.

## 3. Inputs

### 3.1 Candidate pool

Frozen 100-positive / 100-negative candidate pool per query (from DINOv2-L
matcher, Phase X candidate generator). Same pool as Phase X.

### 3.2 Features

All test-time-safe features from `phaseY_bank_policy_protocol.md` section 3,
PLUS:

- per-candidate score
- per-candidate coordinate (relative)
- per-candidate local density
- per-candidate reference agreement

### 3.3 Labels

- Search Oracle's selected positive subset (set of candidate indices)
- Search Oracle's selected negative subset
- Search Oracle's final K+, K-

## 4. Model Ladder P0-P5

| Model | Description |
|---|---|
| P0 | Top-K contrast — pick top-K positives + hardest negative, K from bank policy |
| P1 | Pointwise regression — score each candidate, pick top-K |
| P2 | Pairwise ranking — learn relative order of candidates |
| P3 | Diversity reranker — top-K then re-rank by diversity |
| P4 | Set encoder — encode full candidate set, output subset |
| P5 | Learned greedy selector — mimic search oracle's add/remove/swap |

## 5. Forbidden

- PPO, DQN, policy gradient
- Direct coordinate regression (must pick from candidate pool)
- GT at inference
- Search trajectory as deployable action (section 10)

## 6. Training

- 3 seeds per learned model.
- Query-level split.
- Early stopping on validation.
- Calibration on validation.
- Holdout does NOT participate in model selection.

## 7. Gate Y4 — Pass Conditions

On Validation Queries, the point-subset policy must satisfy EITHER:

1. `Delta Dice >= 0.015` over strongest fixed, OR
2. `Delta Dice >= 0.0075` over bank policy (Gate Y3 winner)

PLUS: paired bootstrap 95% CI lower bound > 0.

## 8. Outcomes

```text
PHASEY_GATE_Y4_POINT_SUBSET_LEARNABLE       (proceed to holdout)
PHASEY_GATE_Y4_POINT_SUBSET_NOT_LEARNABLE   (point-subset track stops)
```

## 9. If Both Y3 and Y4 Fail

If the bank-policy track fails Y3 AND the point-subset track fails Y4 (or Y2
blocked Y4), the correct conclusion is:

```text
ORACLE_EXISTS_BUT_IS_NOT_PREDICTABLE
```

The paper then reports an honest negative result: the oracle gap is real but
not exploitable by a deployable policy using test-time-safe features. This is
a valid submission outcome per the research charter.

## 10. If Y3 or Y4 Passes

The single best frozen model (from Y3 or Y4, decided on validation) runs
ONCE on the Confirmatory Holdout. Holdout result is the formal claim.
