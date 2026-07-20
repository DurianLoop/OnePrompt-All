# Phase X Action-Space Problem Statement

## Problem

The historical MoNuSAC experiment concluded `PIVOT_ORACLE_ANALYSIS` based on a
finite-bank oracle gap of 0.005011 Dice.
This conclusion is only valid if the 12-action bank spans the actionable prompt
space. The bank's coverage is:

- max positive K = 5 (Top-10 was *invalid*: required 10 candidates but only 5
  were available from the 5-positive candidate pilot).
- max negative K = 3.
- max SAM calls = 5 (only `C_top5_independent` and `R_top5_independent_localneg`).
- groupings tested: one-group, one-point-per-group, and a single 3+2 split.
- cluster-based groupings (kmeans / spatial grid / graph / candidate-component):
  *never tested*.

The absolute performance of every historical action is low:

| Action | Mean Dice | Notes |
|---|---|---|
| B_top5_pos3neg (strongest fixed) | 0.3712 | 5 positives + 3 negatives, 1 call |
| Finite-bank oracle | 0.3762 | per-query best of 12 |
| A_top1_one_group | ~0.03 | single positive |
| A_top3_one_group | ~0.26 | three positives |
| A_top5_one_group | ~0.33 | five positives |

Two low-performance configurations differing by 0.005 Dice cannot establish
"Oracle is near-optimal". They can only establish "no historical action is much
better than another historical action".

## Hypotheses to test

H1 (Action-space inadequacy). Top-K scaling to 20/50/75/100 positives
significantly improves absolute Dice and component recall.

H2 (Grouping matters at high K). Dense positives benefit from spatial clustering
(kmeans / grid / graph / candidate-component) over a single group or
one-per-group.

H3 (Negative scaling matters at high K). Hard-negative counts of 10/20/30/50
further improve Dice at large positive K, beyond what 0/1/3 negatives achieve.

H4 (Action-space saturation). Beyond some K/calls budget, fixed Dice saturates
and bank-oracle gap saturates; only in this regime can a small gap rule out
adaptive value.

H5 (Search oracle reveals hidden capacity). GT-guided greedy/beam search over
the legal candidate pool and budget finds prompts whose mean Dice exceeds the
finite-bank oracle by >= 0.01.

H6 (Adaptive value). After saturation is established, the matched-budget search
oracle beats the strongest fixed action by >= 0.01 Dice in >= 25% of queries
with morphology-predictable structure.

## Out-of-scope

- Formal test (MoNuSAC test, GlaS testA/testB, IDRID test, DDR test).
- Tuning K, cluster count, or budget on test.
- Promoting GT-guided search outputs into automatic actions without a separate
  train-only abstraction step.
- PPO/DQN policy training (Section 18 explicitly forbids these even when policy
  reopens; only hierarchical classifiers / regressors / routers are allowed).
