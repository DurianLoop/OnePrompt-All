# Phase R Reference Reliability Protocol

Date: 2026-07-22

Per directive section 8. Only test-time-safe features are allowed.

## 8.1 Reference-only features

- reference foreground area ratio
- foreground component count
- mask compactness
- mask fragmentation
- reference embedding mean/std
- foreground/background prototype separation
- reference stain/intensity statistics

## 8.2 Reference-query pair features

- global embedding cosine
- foreground prototype max/mean score
- background prototype max/mean score
- contrast mean/std/entropy
- top-score decay
- peak count
- candidate dispersion
- estimated foreground ratio
- map smoothness
- high-score connected components
- cross-reference disagreement
- candidate overlap with other references

## 8.3 FORBIDDEN features

- query GT
- query component count GT
- query Dice
- oracle reference ID
- best-reference label from validation
- formal-test information
- search trajectory

## Training labels

y_r = Dice(q, r) — computed ONLY on train queries.
Validation does NOT generate training labels.

## Model ladder (Gate R3)

- M0: Uniform weights (baseline)
- M1: Handcrafted reliability score (e.g., prototype separation - lambda * map entropy)
- M2: Linear regression
- M3: Gradient boosting (lightweight)
- M4: Small MLP (only if M2/M3 insufficient)

## Forbidden models

- Large transformers
- RL reference selector
- SAMIX-style sequential selection
- Validation GT supervision

## Training objective

L = L_utility + lambda * L_rank
L_utility = (y_hat_r - y_r)^2
L_rank = max(0, 1 - (y_hat_i - y_hat_j) * sign(y_i - y_j))
