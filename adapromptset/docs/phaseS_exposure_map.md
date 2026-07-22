# Phase S Exposure Map

Date: 2026-07-22
Branch: `phaseS-multitrack-reference-robust-research`

## What Each Split Is Exposed To

| Split | Train labels | Hyperparameter tuning | Inner CV | Gate eval | Holdout | Cross-dataset | Formal test |
|---|---|---|---|---|---|---|---|
| MoNuSAC train (80) | YES | YES | YES | NO | NO | NO | NO |
| MoNuSAC validation (41) | NO | NO | NO | YES | NO | NO | NO |
| MoNuSAC holdout (40) | NO | NO | NO | NO | ONCE | NO | NO |
| MoNuSAC test (101) | NO | NO | NO | NO | NO | NO | NO |
| GlaS train (55) | NO | NO | NO | NO | NO | Reference pool only | NO |
| GlaS val (30) | NO | NO | NO | NO | NO | YES (after holdout) | NO |
| GlaS testA/B | NO | NO | NO | NO | NO | NO | NO |
| DDR train (383) | NO | NO | NO | NO | NO | Reference pool only | NO |
| DDR valid (149) | NO | NO | NO | NO | NO | YES (one intent, after holdout) | NO |
| DDR test (225) | NO | NO | NO | NO | NO | NO | NO |

## Train-Only Labels (Track B §7.2)

Generated ONLY on MoNuSAC train=80:
- Utility label `u(q, R_r) = Dice(M̂(q, R_r), M_q)`
- Pairwise label `y_ij = 1[u_i > u_j]`
- Harmfulness label `h_r = 1[u(q, R_r) < u_mean(q) - δ]`, δ ∈ {0.02, 0.05}

Validation, holdout, and test NEVER receive train-label generation.

## Test-Time-Safe Features (Track B §7.3)

Features used for reliability prediction MUST be computable without query GT:
- Reference-only features (foreground ratio, component count, compactness, etc.)
- Reference-query features (embedding similarity, matching statistics, etc.)
- Leave-one-reference-out features (consensus map change, candidate overlap
  change, prompt coordinate shift)

Any feature that requires query GT is prohibited at inference time.

## Holdout Exposure Rule

The holdout split is observed EXACTLY ONCE per frozen method. After
observation:
- No further tuning.
- No "let me try another method on holdout."
- If the holdout fails, the method is recorded as `PHASES_HOLDOUT_CONFIRMATION_FAIL`
  and may not be retried.

## Cross-Dataset Exposure Rule

Cross-dataset (GlaS / DDR) is allowed ONLY after MoNuSAC holdout passes.
Cross-dataset results may not feed back into MoNuSAC method tuning.

## Formal Test Lock

Formal test (MoNuSAC test=101, GlaS testA/testB, DDR test=225) is LOCKED for
the entire Phase S program. No exceptions.
