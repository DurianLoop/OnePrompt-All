# Phase R Consensus Methods (Frozen)

Date: 2026-07-22

Per directive section 7. All methods are frozen at preregistration.

## B0. Single Reference

Per-reference Dice. Report: mean, best, worst, std, catastrophic failure rate.
Best reference is privileged diagnostic only — NOT deployable.

## B1. Random Single Reference

Randomly select 1 of 6 references (fixed seed). Deployable high-variance baseline.

## B2. Mean Consensus

C_mean(x) = (1/m) * sum_r C_r(x)

## B3. Median Consensus

C_median(x) = median_r C_r(x)

## B4. Trimmed Mean

Remove max and min reference score per position, then mean. Requires >= 5 references.

## B5. Disagreement-Penalized Consensus

mu(x) = (1/m) * sum_r C_r(x)
sigma(x) = sqrt((1/m) * sum_r (C_r(x) - mu(x))^2)
C_stable(x) = mu(x) - beta * sigma(x)

Frozen beta grid: [0.25, 0.5, 1.0]
beta selected ONLY on train.

## B6. Quantile Consensus

C_q(x) = Q_tau({C_r(x)})

Frozen tau grid: [0.50, 0.67, 0.83]
tau selected ONLY on train.

## B7. Reliability-Weighted Mean

w_r = exp(g_r / T) / sum_j exp(g_j / T)
C_weighted(x) = sum_r w_r * C_r(x)

CONDITIONAL: Only trained after Gate R2 passes.

## Frozen Config

See `configs/frozen/phaseR_consensus_methods.yaml` for machine-readable frozen config.
