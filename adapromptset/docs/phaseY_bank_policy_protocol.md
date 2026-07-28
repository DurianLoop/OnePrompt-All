# Phase Y — Bank-Action Policy Protocol (Gate Y3)

Date: 2026-07-21

Directive section: 10.

## 1. Track

Track A: Bank-Action Policy is the main route. The policy selects an action
from the compact policy bank; it does NOT generate coordinates.

## 2. Policy Output

```text
positive_selector
K+
negative_selector
K-
```

Fixed: `G_ONE_GROUP`, `1 SAM call`.

## 3. Test-Time-Safe Features (ONLY allowed features)

### 3.1 Query global features

- image embedding mean / std
- patch embedding entropy
- stain / intensity statistics
- image size / aspect ratio

### 3.2 Matching-map features

- foreground score mean / std / max
- background score mean / std / max
- contrast mean / std / max
- contrast entropy
- positive mass ratio
- high-score area ratio
- low-score area ratio
- peak count
- peak density
- estimated fragmentation
- estimated foreground area
- score-map spatial dispersion

### 3.3 Candidate features

- candidate count
- top-score decay
- top-5 / top-20 margin
- candidate coordinate dispersion
- estimated component count
- local peak count
- redundancy
- positive-negative separation

### 3.4 Reference consistency

- A-internal 3-reference map disagreement
- reference-query embedding gap
- per-reference candidate overlap
- score variance across references

### 3.5 Action descriptors

- K+, K-, selector token, point budget, expected runtime

## 4. FORBIDDEN Features

- GT target count
- GT component count
- GT area
- GT boundary
- oracle action
- oracle Dice
- validation rank
- search trajectory info not available at inference

## 5. Model Ladder (train in order, no skipping)

| Model | Description |
|---|---|
| M0 | Global prior — always pick train strongest fixed |
| M1 | Morphology heuristic — manual buckets on estimated target count + map dispersion |
| M2 | Multiclass action classifier — predict compact action ID |
| M3 | Per-action utility regression — predict `U_hat(q,a)`, pick `argmax_a` |
| M4 | Pairwise ranking — predict `P(U(q,a_i) > U(q,a_j))` |
| M5 | Hierarchical router — K+ bin -> K- -> pos selector -> neg selector |

## 6. Forbidden Models

- PPO
- DQN
- policy gradient
- continuous point coordinate generation

## 7. Training Protocol

- Train on Policy Train Queries (80) only.
- Validation Queries (41) select model and capacity.
- Confirmatory Holdout (40) runs the final frozen model ONCE.
- Each learned model: 3 seeds.
- Query-level split; reference episodes are NOT independent samples.
- Early stopping on validation only.
- Calibration on validation only.
- Holdout does NOT participate in any model selection.

## 8. Baselines (all must be reported)

1. Historical `B_top5_pos3neg`
2. Fixed Top20 positive
3. Strongest fixed dense: `Top20 + 1 N_BOTTOMK negative + G_ONE_GROUP + 1 call`
4. Validation-selected strongest fixed compact action
5. Morphology heuristic (M1)
6. Random action router
7. Frequency-prior router
8. Bank oracle (upper bound, diagnostic)
9. Stable cross-reference oracle (upper bound, diagnostic)

## 9. Gate Y3 — Pass Conditions

On Validation Queries, the learned policy must simultaneously satisfy:

1. `Delta Dice >= 0.010` over strongest fixed
2. paired bootstrap 95% CI lower bound > 0
3. `oracle recovery >= 30%`
   `recovery = (policy - strongest_fixed) / (oracle - strongest_fixed)`

## 10. Outcomes

```text
PHASEY_GATE_Y3_BANK_POLICY_VALIDATED       (proceed to holdout)
PHASEY_GATE_Y3_BANK_POLICY_INSUFFICIENT    (bank-policy track stops)
```

If Y3 fails, the bank-policy track stops. Point-subset track (Y4) may still
proceed if Y2 passed.
