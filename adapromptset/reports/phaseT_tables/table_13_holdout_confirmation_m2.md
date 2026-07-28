# Table 13 — Holdout confirmation (M2)

Validation vs holdout metrics for the M2 dual-model selective prediction. The model and threshold were frozen on validation and run once on the 40-query MoNuSAC holdout.

| Metric | Validation | Holdout | Gate threshold | Direction consistent? |
| --- | --- | --- | --- | --- |
| Instability AUROC | 0.9443 | 0.9283 | ≥ 0.70 | YES |
| Catastrophic AUPRC | 0.4338 | 0.5277 | > prevalence | YES |
| Prevalence (catastrophic) | 0.1171 | 0.0500 | — | — |
| ECE | 0.0425 | 0.0270 | ≤ 0.10 | YES |
| CFR (no abstention) | 0.1171 | 0.0500 | — | — |
| CFR @80% coverage | 0.0711 | 0.0104 | — | — |
| CFR reduction @80% | 39.2% | 0.7917 | ≥ 30% | YES (79.2%) |
| Dice (no abstention) | 0.0214 | 0.4762 | — | — |
| Dice @80% coverage | 0.4356 | 0.4852 | — | — |
| Dice change @80% | +0.021425 (Dice all | 0.0090 | ≥ 0 | YES (+0.0090) |

**Notes:**
- Frozen models: catastrophic = M2-2_logistic_regression, instability = M2-2_logistic_regression.
- Holdout: 1200 episodes (40 queries) — one-shot confirmation, no model re-fitting.
- All 6 holdout gate conditions PASS (cond1-cond5 on holdout + cond6 directional consistency with validation).
- Decision: **PHASET_M2_SELECTIVE_PREDICTION_PASS**.
- Holdout CFR reduction (79.2%) is substantially larger than validation (39.2%) because holdout prevalence is lower (0.05 vs 0.117) — the abstention threshold transfers favorably.
