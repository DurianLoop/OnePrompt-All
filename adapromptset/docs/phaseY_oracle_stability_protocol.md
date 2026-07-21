# Phase Y — Oracle Stability Protocol (Gates Y1, Y2)

Date: 2026-07-21

Directive section: 8, 9.

## 1. Purpose

Verify that the Phase X oracle gaps (bank oracle 0.0708, search oracle 0.1845
over strongest fixed) are NOT artifacts of:

- Winner's Curse (maximum selection bias over 91+ actions),
- reference sampling noise,
- 10-episode insufficiency,
- search-oracle overfitting to a specific reference.

## 2. Gate Y1 — Fresh-Query Action Heterogeneity

### 2.1 Scope

- Queries: Policy Validation (41 cases)
- Bank: full audit bank (210 actions)
- References: Set A (3) and Set B (3) per query
- SAM calls: 1 per action (frozen)

### 2.2 Four Upper Bounds Computed

| Bound | Formula |
|---|---|
| Fixed-A | `argmax_a mean_q U_A(q,a)` evaluated on A |
| Oracle-A | `mean_q max_a U_A(q,a)` on A |
| Cross-Ref Oracle A->B | choose `a*_q,A = argmax_a U_A(q,a)`, evaluate on B |
| Reverse Cross-Ref B->A | symmetric |

**Stable Oracle** = `(U_cross_A->B + U_cross_B->A) / 2`

### 2.3 Oracle Optimism

```text
Optimism_A = U_oracle_A - U_cross_A->B
Optimism_B = U_oracle_B - U_cross_B->A
```

### 2.4 Action Stability Metrics

- exact action agreement (A vs B per query)
- K+ bin agreement
- K- agreement
- selector agreement
- action-rank Spearman correlation (per query, median over queries)
- top-3 action overlap
- utility correlation
- reference-set win consistency

### 2.5 Pass Conditions (ALL must hold)

1. `stable_oracle_gap >= 0.015 Dice` (over strongest fixed)
2. paired bootstrap 95% CI lower bound > 0
3. `>= 30% validation queries` stable best-action family != strongest fixed
4. cross-reference action-rank `median Spearman rho > 0.30`
5. `oracle optimism <= 60% of raw oracle gap`
6. all actions fixed at 1 SAM call (no call-count confound)

### 2.6 Outcomes

```text
PHASEY_GATE_Y1_STABLE_HETEROGENEITY       (all 6 conditions met)
PHASEY_GATE_Y1_REFERENCE_UNSTABLE         (cross-ref gap collapses)
PHASEY_GATE_Y1_SELECTION_BIAS_DOMINATES   (optimism > 60%)
PHASEY_GATE_Y1_ORACLE_GAP_SMALL           (gap < 0.015)
```

On Y1 fail: bank-action policy training is BLOCKED. Pivot to point-subset
stability audit only. Holdout / test access FORBIDDEN.

## 3. Gate Y2 — Search Oracle Stability

### 3.1 Scope

- Queries: Policy Train (80) + Policy Validation (41) = 121
- Search budget: B_point=100, B_call=1
- References: Set A and Set B per query (dual search)
- **FORBIDDEN**: Confirmatory Holdout, formal test

### 3.2 Dual-Reference Search

For each query:

1. Run GT-guided search with Reference Set A -> subset `S_A`
2. Run GT-guided search with Reference Set B -> subset `S_B`
3. Cross-evaluate: `S_A` on B, `S_B` on A

### 3.3 Reported Metrics

- raw search Dice (A on A, B on B)
- cross-reference search Dice (A on B, B on A)
- K+ stability (|K_A - K_B|)
- K- stability
- point-coordinate overlap
- spatial coverage overlap
- candidate-rank overlap
- search optimism = raw - cross
- final prompt area bias

### 3.4 Interpretation Rules

| Situation | Code |
|---|---|
| raw high, cross high | `SEARCH_ORACLE_REFERENCE_STABLE` |
| raw high, cross drops | `SEARCH_ORACLE_REFERENCE_OVERFIT` |
| K unstable, cross Dice stable | `MULTIPLE_EQUIVALENT_PROMPTS` |
| both unstable | `SEARCH_LABELS_NOT_SUITABLE_FOR_SUPERVISION` |

### 3.5 Pass Conditions (ALL must hold)

1. `cross-reference search >= 0.03 Dice` over strongest fixed
2. paired bootstrap CI lower bound > 0
3. `search optimism <= 50% of raw search gap`
4. `median |K_A - K_B| <= 8` OR cross Dice diff <= 0.01
5. `>= 70% queries`: A-search prompt beats strongest fixed on B

### 3.6 Outcomes

```text
PHASEY_GATE_Y2_SEARCH_LABELS_STABLE      (may proceed to point-subset policy)
PHASEY_GATE_Y2_SEARCH_ORACLE_OVERFIT     (point-subset policy blocked)
```

## 4. Reporting Requirements

Gate Y1 and Y2 reports MUST include:

- raw bank oracle
- cross-reference oracle
- stable oracle
- oracle optimism
- strongest fixed
- stable oracle gap
- per-query action agreement
- Spearman rank correlation distribution
- bootstrap CI

Raw oracle alone is INSUFFICIENT.
