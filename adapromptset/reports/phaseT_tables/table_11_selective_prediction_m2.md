# Table 11 — Selective prediction (M2, validation)

M2 dual-model selective prediction comparison on the WP3 validation split (615 episodes, 41 queries). The catastrophic-trained model drives CFR/Dice gates; the instability-trained model drives AUROC.

| Model | Type | Instability AUROC | Instability AUPRC | Catastrophic AUPRC | ECE | CFR (no abstention) | CFR @80% coverage | CFR reduction @80% | Dice (no abstention) | Dice @80% coverage | Dice change @80% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M2-0_entropy_threshold | threshold | 0.4977 | 0.8555 | 0.0831 | 0.0226 | 0.1171 | 0.0935 | 0.2014 | 0.4141 | 0.4301 | 0.0159 |
| M2-1_disagreement_threshold | threshold | 0.4977 | 0.8555 | 0.0831 | 0.0226 | 0.1171 | 0.0935 | 0.2014 | 0.4141 | 0.4301 | 0.0159 |
| M2-2_logistic_regression | sklearn | 0.9443 | 0.9894 | 0.4338 | 0.0425 | 0.1171 | 0.0711 | 0.3924 | 0.4141 | 0.4356 | 0.0214 |
| M2-3_gradient_boosting | sklearn | 0.9265 | 0.9830 | 0.3997 | 0.0658 | 0.1171 | 0.0650 | 0.4444 | 0.4141 | 0.4364 | 0.0223 |
| M2-4_small_mlp | sklearn | 0.9413 | 0.9890 | 0.4167 | 0.0583 | 0.1171 | 0.0671 | 0.4271 | 0.4141 | 0.4360 | 0.0218 |

**Notes:**
- Dual-model approach: catastrophic-trained model = M2-2_logistic_regression, instability-trained model = M2-2_logistic_regression.
- Validation prevalence: instability = 0.8472, catastrophic = 0.1171.
- cond1 uses the instability-trained model (AUROC ≥ 0.70); cond2-cond5 use the catastrophic-trained model.
- Best model M2-2 logistic regression: instability AUROC = 0.9443, catastrophic AUPRC = 0.4338 (> prevalence 0.1171), ECE = 0.0425, CFR reduction @80% = 39.2%, Dice change @80% = +0.0214.
- Threshold-based M2-0/M2-1 share the same metrics because their disagreement/entropy features collapse to the same ranking.
- All models report cfr_no_abstention = 0.1171 (= prevalence) because without abstention no episode is filtered.
