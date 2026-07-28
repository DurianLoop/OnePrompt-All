# Phase T Holdout Protocol

Date: 2026-07-23
Branch: `phaseT-publication-completion`
Directive: `D:\SAM\adapromptset\reports\AdaPromptSet_PhaseT_Publication_Completion.md`
Related: directive §5 (WP1), §5.2 (freeze list), §12 (final-method holdout)

## Holdout Definition

- **Dataset**: MoNuSAC internal holdout.
- **Size**: 40 queries (8 patients, patient-disjoint from train and
  validation).
- **Source**: `manifests/phaseY/confirmatory_holdout_queries.csv` (reused
  from Phase Y patient-level split, seed=20260721).
- **Formal test**: 101 cases — LOCKED, not part of the holdout.

## Run Policy

The MoNuSAC holdout may be observed EXACTLY ONCE after the protocol is
frozen. Two holdout observations are allowed in Phase T:

1. **WP1 holdout** — confirmatory benchmark (B1–B7, k∈{1,3,6}, 3 prompt
   configs) under the frozen protocol.
2. **WP3 final-method holdout** — a single frozen M1 OR M2 model run on the
   same 40 queries, after the validation Gate passes.

Each holdout run is independent and is preceded by its own hash verification.
No parameter may be retuned between the two runs based on holdout
observations.

## Freeze List

The following parameters MUST be frozen before the WP1 holdout hash is
verified. After freezing, NONE of these may be changed based on holdout
observations.

1. **Reference pool** — train-only, 80 cases / 18 patients, patient-disjoint
   from holdout queries.
2. **Reference subset sizes** — k ∈ {1, 3, 6}.
3. **Resampling seeds** — 10 seeds for train; fixed seeds for holdout
   (generated once, never regenerated).
4. **Prompt config** — the three frozen Phase R / Phase S configs:
   - `P20-N1` (K+=20, K-=1, N_BOTTOMK)
   - `P30-N0` (K+=30, K-=0, N_NONE)
   - `P50-N0` (K+=50, K-=0, N_NONE)
5. **Matching normalization** — z-score foreground / background normalization
   (Phase 0/1 frozen).
6. **Candidate selection** — P_TOPK positive selector, N_BOTTOMK negative
   selector (Phase R frozen).
7. **SAM config** — SAM-B, single call per consensus aggregation, multimask
   output rule frozen.
8. **Metric definitions** — all metrics defined in
   `docs/phaseT_metric_protocol.md`, including V_query, V_ref, V_ref_frac,
   MAD_ref, J_pred, CVaR_0.1, CVaR_0.2, CFR, worst-subset Dice, best-worst
   gap.
9. **Catastrophic threshold** — Dice < 0.1 (frozen; may NOT be retuned).
10. **Bootstrap procedure** — 10,000 iterations, patient-cluster resampling
    for MoNuSAC, 95% CI from 2.5th / 97.5th percentiles.
11. **Qualitative sampling rules** — queries selected for qualitative
    figures must be chosen by pre-registered rules (e.g., worst-5, best-5,
    median-5 by V_ref), NOT by hand-picking favorable cases.
12. **Cost accounting** — reference count, DINOv2 matching passes (cached +
    uncached), SAM calls, runtime seconds, GPU-hours.

## Hash Verification

Before the WP1 holdout run begins, the following files must be hashed and
recorded in `reports/phaseT_input_hashes.json`:

- `manifests/phaseY/confirmatory_holdout_queries.csv`
- `manifests/phaseY/reference_pool.csv`
- `configs/frozen/phaseR_prompt_configs.yaml`
- `configs/frozen/phaseR_consensus_methods.yaml`
- `configs/frozen/phaseR_statistics.yaml`
- `configs/frozen/phaseR_beta_tau_grid.yaml`
- The Phase T frozen benchmark config
  `configs/frozen/phaseT_monusac_benchmark.yaml` (to be created at WP1
  freeze time).

The hash verification step MUST be logged before any holdout inference. If
any hash differs from the preregistered value, the holdout run is BLOCKED
until the discrepancy is resolved and re-hashed.

## WP1 Gate (directive §5.6)

ALL 10 of the following conditions must be satisfied for WP1 to pass:

1. ≥ 50% holdout queries have best-worst gap ≥ 0.05.
2. Mean within-query reference std ≥ 0.02.
3. V_ref_frac ≥ 0.10.
4. Mean consensus (B2) vs random single (B1): Δ Dice ≥ 0.010.
5. Paired bootstrap 95% CI lower bound > 0 (B2 vs B1).
6. Catastrophic failure rate of B2 not increased vs B1.
7. CVaR_0.1 of B2 not decreased vs B1.
8. Prediction consistency (J_pred) of B2 improved vs B1.
9. Disagreement feature vs instability: Spearman |ρ| ≥ 0.25.
10. No data leakage and no code-axis confounding (audit PASS).

Pass state:

```
PHASET_WP1_MONUSAC_HOLDOUT_CONFIRMED
```

Fail state:

```
PHASET_WP1_MONUSAC_HOLDOUT_FAIL
```

If WP1 fails: stop publication advancement; do NOT access formal test.

## Prohibited Post-Holdout Operations

1. Retuning any parameter on the freeze list after the holdout run begins.
2. Re-running the holdout with a different seed, config, or method variant.
3. Selectively reporting holdout subsets.
4. Changing the catastrophic threshold based on observed holdout Dice.
5. Changing the bootstrap iteration count after observing the CI.
6. Promoting a method to "holdout-confirmed" without all 10 Gate conditions
   passing.
7. Using holdout observations to retrain M1 / M2 models.

## Audit

Every holdout run produces:

- `reports/phaseT_wp1_holdout_summary.{json,md}` — per-query metrics,
  Gate condition evaluation, hash record.
- `reports/phaseT_wp1_per_query.csv` — per-query Dice / V_ref / CVaR / CFR.
- `reports/phaseT_wp1_per_reference.csv` — per-(query, reference) raw
  metrics.
- `reports/phaseT_wp1_run_metadata.json` — environment, git, config,
  input hashes.

The claim-boundary audit scans all holdout reports for prohibited tokens
before the final state transition.
