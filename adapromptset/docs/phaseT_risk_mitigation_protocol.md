# Phase T Risk Mitigation Protocol

Date: 2026-07-23
Branch: `phaseT-publication-completion`
Directive: `D:\SAM\adapromptset\reports\AdaPromptSet_PhaseT_Publication_Completion.md`
Related: directive §8 (WP3), §9 (M1), §10 (M2), §11 (route selection)

## Purpose

Phase S established that reference instability is real and predictable but
cannot be mitigated through reference weighting (Track B `NO_GAIN`) or set
conditioning (Track C `NO_GAIN`). Phase T WP3 therefore registers two more
direct practical routes that do NOT rely on weighting as the sole mechanism:

- **M1 — Dynamic Reference Budget**: select how many references to use per
  query (k∈{1,3,6}) based on test-time-safe features.
- **M2 — Instability-Aware Selective Prediction**: classify each query as
  stable / unstable / catastrophic-risk and abstain or escalate accordingly.

## Route Selection Priority (directive §11)

1. M2 failure prediction (primary).
2. M1 dynamic budget (secondary).
3. Only if BOTH M1 and M2 fail, consider a new learned fusion route — but
   this is NOT the default, and Phase S Track B already showed learned
   weighting does not improve Dice.

Rationale: Phase S falsified simple weighting; instability is predictable;
selective prediction and dynamic budget fit the existing evidence better;
the method claims are easier to differentiate from SAMIX (reference
selection), ProSAM (prompt uncertainty), and MAUP (prompt quantity).

---

## M1 — Dynamic Reference Budget

### Motivation

Phase S showed:

- Instability can be predicted from test-time-safe features.
- k=6 mean consensus is more stable and more accurate than k=1.
- But k=6 requires more matching passes than k=1.

M1 asks: can we use more references only on high-risk queries and fewer
references on low-risk queries, achieving a better performance–cost
trade-off?

### Action Space

The M1 policy selects ONLY the reference subset size:

```
k ∈ {1, 3, 6}
```

Within each selected k, the aggregation is fixed to **mean consensus (B2)**.
The policy does NOT select:

- Specific reference IDs (the 6-reference pool is shared and the subset is
  drawn uniformly at random per the resampling seeds).
- Point subset.
- SAM call count.
- Grouping.
- K+ / K−.
- Segmentation model.

### Test-Time-Safe Features

M1 uses only test-time-safe features (no query ground truth at inference).
The frozen feature set (directive §9.3):

- Matching-map disagreement.
- Prototype separation.
- Score entropy.
- Candidate dispersion.
- Reference-set embedding spread.
- Leave-one-reference-out map deviation.
- Predicted foreground area variance.
- Prompt coordinate shift.
- Query / reference global similarity.

### Model Ladder

```
M1-0  Always k=1
M1-1  Always k=3
M1-2  Always k=6
M1-3  Handcrafted threshold
M1-4  Logistic / ordinal regression
M1-5  Gradient boosting
M1-6  Small MLP
```

Large set transformers are PROHIBITED unless a lightweight model (M1-3 to
M1-5) already demonstrates a stable positive signal.

### Training Objective

Risk–cost utility:

```
U(q, k) = Dice(q, k) − λ_c · C(k) − λ_r · Risk(q, k)
```

where:

- `C(k)` is the matching cost (proportional to k; SAM cost is fixed at 1
  call by construction of mean consensus).
- `Risk(q, k)` is a predicted instability, catastrophic probability, or
  CVaR surrogate.

Pre-registered hyperparameter grid (selected on train / inner CV only):

```
λ_c ∈ {0, 0.01, 0.025, 0.05}
λ_r ∈ {0, 0.25, 0.5, 1.0}
```

### M1 Gate (directive §9.6)

Relative to `Always k=6` (M1-2), M1 must satisfy ONE of two
publication-value modes:

#### Cost-Saving Mode

1. Mean Dice not below k=6 by more than 0.003.
2. CVaR not worsened by more than 0.005.
3. Catastrophic failure rate not increased.
4. Mean reference count reduced by at least 25%.

#### Quality-Cost Mode

1. At a mean reference count comparable to k=3: Δ Dice ≥ 0.0075.
2. CVaR improved.
3. Catastrophic failure rate not increased.

States:

```
PHASET_M1_DYNAMIC_BUDGET_PASS
PHASET_M1_DYNAMIC_BUDGET_NO_GAIN
```

---

## M2 — Instability-Aware Selective Prediction

### Motivation

Even if Dice cannot be improved on every query, it may be possible to
reliably identify high-risk cases. M2 predicts a three-class risk label:

```
stable
unstable
catastrophic-risk
```

and supports downstream actions: output normally, use more references, flag
low confidence, abstain, or request human review.

### Labels (train-only)

Labels are constructed ONLY on the train split (directive §10.2):

#### Instability label

```
best-worst gap >= 0.05   →  unstable
otherwise                →  stable
```

#### Catastrophic label

```
Dice < 0.1   →  catastrophic-risk
otherwise    →  not catastrophic
```

#### Tail-risk label

```
bottom 10% performance   →  tail-risk
otherwise                →  not tail-risk
```

### Model Ladder

```
M2-0  Entropy threshold
M2-1  Disagreement threshold
M2-2  Logistic regression
M2-3  Gradient boosting
M2-4  Small MLP
```

### Evaluation

M2 is evaluated on:

- **AUROC** — area under ROC for instability / catastrophic-risk
  classification.
- **AUPRC** — area under precision-recall (more informative than AUROC under
  class imbalance).
- **Calibration** — expected calibration error (ECE).
- **Risk-coverage curve** — selective Dice / selective CFR as a function of
  abstention rate.
- **Selective Dice** — mean Dice on the non-abstained subset.
- **Selective CFR** — CFR on the non-abstained subset.
- **Abstention rate** — fraction of queries abstained.
- **Cost** — additional feature-computation cost.

### M2 Gate (directive §10.5)

ALL of the following must be satisfied:

1. Instability AUROC ≥ 0.70.
2. Catastrophic-risk AUPRC clearly above prevalence.
3. ECE ≤ 0.10 (or ≤ 0.08 after validation calibration).
4. At 80% coverage: CFR relative reduction ≥ 30%.
5. At 80% coverage: mean Dice not decreased.
6. Holdout direction consistent with validation.

States:

```
PHASET_M2_SELECTIVE_PREDICTION_PASS
PHASET_M2_SELECTIVE_PREDICTION_NO_GAIN
```

---

## Final-Method Holdout (directive §12)

After M1 OR M2 passes on validation:

1. Freeze the model.
2. Freeze the threshold.
3. Freeze the calibration.
4. Freeze the cost definition.
5. Freeze the risk-coverage reporting rule.
6. Run on the MoNuSAC holdout (n=40) EXACTLY ONCE.

The holdout run must NOT be used to retune the model, threshold, or
calibration. See `docs/phaseT_holdout_protocol.md`.

## Prohibited Mitigation Operations

1. Using query ground truth at inference time.
2. Training on validation or holdout labels.
3. Retuning the model after holdout observation.
4. Claiming the instability is "solved" (prohibited claim #1).
5. Claiming the method works without references (prohibited claim #9).
6. Using a large set transformer before a lightweight model demonstrates
   stable positive signal.
7. Treating M1 and M2 as independent samples when they share the same
   query set (statistical unit = query).
