# Table 10 — Dynamic reference budget (M1)

M1 dynamic reference budget policy comparison on the WP3 validation split (615 episodes, 41 queries). Policy selects k ∈ {1, 3, 6} per query using test-time-safe disagreement features.

| Model | Type | λ_c | λ_r | mean Dice | avg k | CVaR_0.1 | CFR | cost (k/6) | mean utility U |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M1-0_always_k1 | baseline | 0.0000 | 0.0000 | 0.4141 | 1.00 | 0.0598 | 0.1171 | 0.1667 | 0.3012 |
| M1-1_always_k3 | baseline | 0.0000 | 0.0000 | 0.4432 | 3.00 | 0.0544 | 0.1154 | 0.5000 | 0.3220 |
| M1-2_always_k6 | baseline | 0.0000 | 0.0000 | 0.4586 | 6.00 | 0.0490 | 0.1252 | 1.0000 | 0.3248 |
| M1-3_handcrafted_threshold | threshold | 0.0000 | 0.0000 | 0.4571 | 5.63 | 0.0537 | 0.1203 | 0.9390 | 0.3248 |
| M1-4_logistic_regression | logistic | 0.0000 | 0.0000 | 0.4552 | 5.88 | 0.0490 | 0.1252 | 0.9797 | 0.3220 |
| M1-5_gradient_boosting | gradient_boosting | 0.0000 | 0.0000 | 0.4516 | 4.98 | 0.0528 | 0.1171 | 0.8301 | 0.3221 |

**Notes:**
- Best model on validation: M1-3_handcrafted_threshold (selected λ_c = 0.0, λ_r = 0.0).
- Baseline k=6: mean Dice = 0.4586, CFR = 0.1252, avg_k = 6.0.
- Best policy: mean Dice = 0.4571, avg_k = 5.63 (only 6.1% reduction from k=6).
- Cost-saving mode cs4 FAIL: avg_k = 5.63 ≥ 4.5 threshold (needs ≥ 25% reference reduction).
- Quality-cost mode FAIL: qc1 requires avg_k in [2.5, 3.5] but policy avg_k = 5.63; qc2 and qc3 also fail.
- Decision: **PHASET_M1_DYNAMIC_BUDGET_NO_GAIN** — the learned/threshold policy cannot achieve a meaningful cost saving without hurting Dice, and cannot beat k=3 at k=3-cost. M1 is reported as a negative result.
