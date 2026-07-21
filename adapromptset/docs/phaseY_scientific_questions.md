# Phase Y — Scientific Questions

Date: 2026-07-21

Directive section: 1 (five serious doubts) and 2 (main question).

## Five Doubts To Verify

### D1. Winner's Curse in the 91-action bank oracle

Phase X selected the per-query maximum Dice over many actions. More actions
inflate the maximum via reference sampling noise, query-specific fluctuation,
mask-threshold sensitivity, and action-selection noise.

**Verify**: does the A_large oracle gap (0.0708) survive on new queries with
independent reference sets and cross-reference evaluation?

### D2. 10-episode insufficiency

Phase X drew all core conclusions from 10 validation episodes. 10 units can
reveal an old-bank flaw but cannot establish: stable K+ distribution, stable
K- distribution, stable grouping preference, reliable oracle-gap CI,
generalizable action differences, or trainability.

**Verify**: expand to >= 80 train / >= 40 validation / >= 40 holdout
independent query units, with query/case as the statistical unit.

### D3. Search Oracle overfitting

Search Oracle uses GT and ~558 SAM search calls per query to select a specific
point subset. It does not just choose (K+, K-, grouping, calls); it also
chooses which specific positive/negative candidates and which add/remove/swap
moves. Therefore 0.7256 is NOT a direct upper bound for a 91-action router.

**Verify**: separate Bank-Action Policy (finite action config only) from
Point-Subset Policy (learn the actual point subset).

### D4. Matched-budget fairness

Search Oracle's final prompt needs 1 SAM call, but finding it requires GT and
hundreds of search calls. It only proves the prompt *interface* has high
potential; it is not a deployable method comparable to strongest fixed.

**Verify**: all paper tables strictly distinguish `automatic deployable
method`, `finite-bank oracle`, `cross-reference oracle`, `GT-guided search
oracle`, and `privileged upper bound`.

### D5. Action heterogeneity predictability

Even if the oracle gap is real, it may not be predictable from
inference-time-visible information.

**Verify**: using only matching map, candidate statistics, reference
consistency, and query image features (NO query GT), can a policy select an
action that beats strongest fixed on new queries?

If not, the correct conclusion is `ORACLE_EXISTS_BUT_IS_NOT_PREDICTABLE`,
not "increase model complexity".

## Main Question (Narrowed)

> Can a query-conditioned, single-call prompt-configuration policy outperform
> the strongest fixed dense prompt under a strictly matched deployment budget?

English:

> Can a query-conditioned, single-call prompt-configuration policy outperform
> the strongest fixed dense prompt under a strictly matched deployment budget?

## Out of Scope (This Round)

- Multi-call SAM
- Multi-mask union
- Complex graph fusion
- SAM-L / SAM2 / MedSAM
- GlaS / DDR
- Formal test
- PPO / DQN
- Continuous coordinate generation

## Falsifiable Predictions

| ID | Prediction | Falsifier |
|---|---|---|
| P1 | Stable oracle gap >= 0.015 on new validation queries | gap < 0.015 |
| P2 | Cross-reference oracle gap survives (optimism <= 60%) | optimism > 60% |
| P3 | Bank policy beats strongest fixed by >= 0.010 on validation | Delta < 0.010 |
| P4 | Point-subset policy beats strongest fixed by >= 0.015 OR bank policy by >= 0.0075 | neither |
| P5 | Holdout confirms validation result (within bootstrap CI) | holdout CI excludes validation gain |

Any falsifier triggers the corresponding fail-state in the gate registry.
