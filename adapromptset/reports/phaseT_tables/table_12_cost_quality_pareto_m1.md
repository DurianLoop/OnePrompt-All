# Table 12 — Cost-quality Pareto (M1)

Cost-quality Pareto comparison of M1 baselines and learned policies on the WP3 validation split. Cost = k/6 (normalized reference count).

| Model | cost (k/6) | avg k | mean Dice | CVaR_0.1 | CFR | Pareto-optimal? |
| --- | --- | --- | --- | --- | --- | --- |
| M1-0_always_k1 | 0.1667 | 1.00 | 0.4141 | 0.0598 | 0.1171 | YES |
| M1-1_always_k3 | 0.5000 | 3.00 | 0.4432 | 0.0544 | 0.1154 | YES |
| M1-5_gradient_boosting | 0.8301 | 4.98 | 0.4516 | 0.0528 | 0.1171 | YES |
| M1-3_handcrafted_threshold | 0.9390 | 5.63 | 0.4571 | 0.0537 | 0.1203 | YES |
| M1-4_logistic_regression | 0.9797 | 5.88 | 0.4552 | 0.0490 | 0.1252 | no |
| M1-2_always_k6 | 1.0000 | 6.00 | 0.4586 | 0.0490 | 0.1252 | YES |

**Notes:**
- Pareto frontier is computed on cost (k/6) vs mean Dice, with CVaR (≥) and CFR (≤) as secondary criteria.
- M1-2 always-k=6 is Pareto-optimal: it has the highest Dice and CVaR but also the highest cost.
- M1-0 always-k=1 is Pareto-optimal: it has the lowest cost but the lowest Dice.
- The learned/threshold policies (M1-3, M1-4, M1-5) lie between the two extremes but are dominated — they do not reach a strictly better cost-quality trade-off than the always-k baselines.
- This is the quantitative basis for the M1 NO_GAIN decision: no policy achieves the §9.6 cost-saving (cs4) or quality-cost (qc1-qc3) gates.
