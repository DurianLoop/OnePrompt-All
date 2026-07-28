# Phase T WP3 M2 Cross-Dataset Transfer Evaluation

Date: 2026-07-23T19:34:43

## Frozen models (trained on MoNuSAC train, selected on MoNuSAC val)

- Catastrophic model: **M2-2_logistic_regression** (val AUPRC_cat = 0.4338)
- Instability model: **M2-2_logistic_regression** (val AUROC = 0.9443)
- MoNuSAC holdout: AUROC = 0.9283, AUPRC_cat = 0.5277, CFR reduction = 0.7917

## Transfer results

No re-tuning on transfer datasets. Frozen models applied as-is.

### MoNuSAC holdout (reference)

- Episodes: 1200 (40 queries)
- Prevalence: instability=0.8917, catastrophic=0.0500

| metric | value |
|--------|-------|
| instability AUROC | 0.9283 (query-level: 1.0000) |
| instability AUPRC | 0.9893 |
| catastrophic AUPRC | 0.5277 (prevalence=0.0500, AUPRC>prevalence: True) |
| ECE | 0.0270 |
| CFR (no abstention) | 0.0500 |
| CFR @80% coverage | 0.0104 |
| CFR reduction @80% | 0.7917 |
| Dice (no abstention) | 0.476220 |
| Dice @80% coverage | 0.485208 |
| Dice change @80% | +0.008987 |

- AUROC >= 0.70: PASS
- AUPRC > prevalence: PASS
- ECE <= 0.10: PASS
- CFR reduction >= 30%: PASS
- Dice not decrease @80%: PASS
- **Transfer 5/5 conditions pass**

### MoNuSeg (transfer)

- Episodes: 450 (30 queries)
- Prevalence: instability=0.6956, catastrophic=0.0000

| metric | value |
|--------|-------|
| instability AUROC | 0.8740 (query-level: 0.9375) |
| instability AUPRC | 0.9192 |
| catastrophic AUPRC | 0.0000 (prevalence=0.0000, AUPRC>prevalence: False) |
| ECE | 0.0601 |
| CFR (no abstention) | 0.0000 |
| CFR @80% coverage | 0.0000 |
| CFR reduction @80% | 0.0000 |
| Dice (no abstention) | 0.493751 |
| Dice @80% coverage | 0.506838 |
| Dice change @80% | +0.013087 |

- AUROC >= 0.70: PASS
- AUPRC > prevalence: FAIL
- ECE <= 0.10: PASS
- CFR reduction >= 30%: FAIL
- Dice not decrease @80%: PASS
- **Transfer 3/5 conditions pass**

### IDRID (transfer)

- Episodes: 450 (30 queries)
- Prevalence: instability=0.2644, catastrophic=0.8067

| metric | value |
|--------|-------|
| instability AUROC | 0.9938 (query-level: 0.9282) |
| instability AUPRC | 0.9827 |
| catastrophic AUPRC | 0.9973 (prevalence=0.8067, AUPRC>prevalence: True) |
| ECE | 0.4767 |
| CFR (no abstention) | 0.8067 |
| CFR @80% coverage | 0.7583 |
| CFR reduction @80% | 0.0599 |
| Dice (no abstention) | 0.098365 |
| Dice @80% coverage | 0.111204 |
| Dice change @80% | +0.012839 |

- AUROC >= 0.70: PASS
- AUPRC > prevalence: PASS
- ECE <= 0.10: FAIL
- CFR reduction >= 30%: FAIL
- Dice not decrease @80%: PASS
- **Transfer 3/5 conditions pass**

## Interpretation

This is NOT a gate condition. The §10.5 gate only requires MoNuSAC
holdout confirmation (cond6). Transfer results inform the paper's
discussion of M2 generalizability and are reported as additional
evidence, not as a pass/fail gate.

