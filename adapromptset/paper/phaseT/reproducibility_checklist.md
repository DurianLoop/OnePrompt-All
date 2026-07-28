# Reproducibility Checklist

This checklist documents the reproducibility posture of the Phase T
paper. Every item maps to a frozen artifact, manifest, or config that is
hash-verified before any holdout inference and recorded in the run
metadata. The formal test set is locked and unused for the entire Phase T
program.

## 1. Code

- **Source repository.** All scripts live under
  `adapromptset/scripts/` and `adapromptset/src/adapromptset/scripts/`.
  The Phase T entry points are:
  - `run_phaseT_wp1_holdout.py` (WP1 confirmatory benchmark).
  - `run_phaseT_wp3_m2_holdout.py` (WP3 M2 holdout confirmation).
  - `generate_phaseT_figures.py`, `generate_phaseT_tables.py` (paper
    assets).
- **Run metadata.** Every WP run produces a
  `reports/phaseT_*_run_metadata.json` recording the git commit,
  environment, frozen config, input hashes, runtime, SAM calls, and
  matching passes.
- **Metric implementations.** `V_ref`, `V_query`, `V_ref_frac`,
  `MAD_ref`, `J_pred`, `CVaR_0.1`, `CVaR_0.2`, CFR, worst-subset Dice,
  best-worst gap, paired bootstrap, and Holm correction are
  unit-tested against analytic references and audited in prior phases.
- **Claim-boundary audit.** A scan over all Phase T artifacts
  (`docs/phaseT_*.md`, `reports/phaseT_*.{md,json,csv}`,
  `paper/phaseT/*.md`) checks for prohibited tokens before any final
  state transition.

## 2. Data

- **Datasets.** MoNuSAC (primary), MoNuSeg (secondary), IDRiD
  (tertiary). All public. Dataset roles, splits, and exposure status are
  recorded in `docs/phaseT_dataset_registry.md`.
- **Reference pools.** Train-only, patient-disjoint from queries.
  - MoNuSAC: 80 cases / 18 patients (`manifests/phaseY/reference_pool.csv`).
  - MoNuSeg: 37 train cases.
  - IDRiD: 383 train cases (one frozen lesion intent).
- **Splits.**
  - MoNuSAC train 80 / validation 41 / holdout 40 / formal test 101
    (locked).
  - MoNuSeg: 37 reference / 30 strict-validation queries / 14 test
    (locked).
  - IDRiD: 383 reference / 30 validation queries / 225 test (locked).
- **Data-leakage audit.** `scripts/audit_reference_exposure.py` runs on
  every regenerated manifest; a failed audit blocks inference. WP1
  holdout: 0 violations across 2,400 reference assignments over 10 seeds.

## 3. Manifests

- `manifests/phaseY/confirmatory_holdout_queries.csv` (40 holdout
  queries).
- `manifests/phaseY/policy_train_queries.csv` (80 train queries).
- `manifests/phaseY/policy_validation_queries.csv` (41 validation
  queries).
- `manifests/phaseY/reference_pool.csv` (train-only reference pool).
- `manifests/phaseR/validation_AB_fixed.csv` (fixed A/B allocation).
- `manifests/phaseT/holdout_resampling_seed_{0..9}.csv` (frozen holdout
  reference allocation).
- `manifests/phaseT/wp2_monuseg_resampling_seed_{0..4}.csv`.
- `manifests/phaseT/wp2_idrid_resampling_seed_{0..4}.csv`.
- `reports/phaseT_input_hashes.json` (hash verification record).

## 4. Seeds

- **Resampling seeds.** MoNuSAC train: 10 seeds (reused from the prior
  10-seed allocation). MoNuSeg and IDRiD development: 5 seeds each.
  Holdout seeds generated once, frozen in manifests, never regenerated.
- **Model-fitting seeds.** M1 regressor and M2 logistic regression
  fitting seeds are recorded in the run metadata. M2 logistic regression
  L2 strength is selected by 5-fold cross-validation on train.
- **Bootstrap.** B=10,000 iterations, patient-cluster resampling for
  MoNuSAC, query-level for MoNuSeg / IDRiD. The bootstrap iteration
  count is frozen at 10,000 and may not be changed after gate
  evaluation.

## 5. Compute

- **Program cap.** 195 GPU-hours total. WP1 used 0 GPU-hours (CPU
  inference, 1,526.5 s wall-clock for 9,600 SAM calls). WP2 and WP3
  used a small CPU budget (model fitting is lightweight). The cap is
  not binding.
- **Checkpoints.** 40 / 90 / 140 / 195 GPU-hours.
- **Compute ledger.** `reports/phaseT_compute_ledger.csv` records SAM
  calls, matching passes (cached + uncached), runtime seconds, and
  GPU-hours per work package.
- **Caches.** DINOv2-L per-case feature cache, per-reference matching
  maps (`s_fg`, `s_bg`, contrast `C_r`), candidate pools, and SAM-B
  output cache are reused from prior phases.

## 6. Frozen configurations

- `configs/frozen/phaseR_prompt_configs.yaml` (P20-N1, P30-N0, P50-N0).
- `configs/frozen/phaseR_consensus_methods.yaml` (B1–B7 aggregators).
- `configs/frozen/phaseR_statistics.yaml` (bootstrap, Holm).
- `configs/frozen/phaseR_beta_tau_grid.yaml`.
- The Phase T frozen benchmark config
  `configs/frozen/phaseT_monusac_benchmark.yaml` (created at WP1 freeze
  time).
- Catastrophic threshold: Dice < 0.1 (frozen; not retunable after
  holdout).

## 7. Pre-registration and holdout policy

- The freeze list (reference pool, k∈{1,3,6}, seeds, prompt configs,
  matching normalization, candidate selection, SAM config, metric
  definitions, catastrophic threshold, bootstrap procedure,
  qualitative-sampling rules, cost accounting) is recorded in
  `docs/phaseT_holdout_protocol.md` and frozen before the WP1 holdout
  hash is verified.
- The MoNuSAC holdout is observed exactly twice: (1) WP1 confirmatory
  benchmark; (2) WP3 final-method one-shot confirmation. Each run is
  independent and preceded by its own hash verification.
- No parameter on the freeze list may be retuned between the two runs
  based on holdout observations.
- Qualitative figures are selected by pre-registered rules (worst-5,
  best-5, median-5 by `V_ref`); hand-picking favorable cases is
  prohibited.

## 8. External baselines

- Comparator tiers and reproduction status are recorded in
  `docs/phaseT_external_baseline_protocol.md`.
- GF-SAM and MAUP identified as Tier 2 reproducible; numerical
  reproduction pending user authorization for external code download.
- No Tier 3 numerical results fabricated; no external method claimed to
  be "fully outperformed".
- The internal B1–B7 families cover at least three distinct aggregation
  families, satisfying the strong-internal-baselines provision of the
  external-comparator minimum.

## 9. Reporting

- Every results table reports mean, paired difference, paired bootstrap
  95% CI, win/tie/loss counts, effect size (Cohen's d), within-query
  reference std, `V_ref`, `V_query`, `V_ref_frac`, `MAD_ref`, `J_pred`,
  `CVaR_0.1`, `CVaR_0.2`, CFR, and seed variability.
- Mean Dice is never reported without variance or tail-risk evidence
  (prohibited-claim #7).
- All gate outcomes (PASS / FAIL / NO_GAIN) are reported, including
  negatives (M1 NO_GAIN, IDRiD FAIL).

## 10. Prohibited outcomes

- No `SUBMISSION_READY` or `PUBLICATION_COMPLETED` state is written.
- No `SOTA` or "state-of-the-art" claim is made.
- No "instability solved" / "instability eliminated" claim is made.
- No "universal across datasets" claim is made (explicitly contradicted
  by IDRiD).
- No "mean Dice alone proves robustness" claim is made.
- No formal test access; the formal test set is locked and unused for
  the entire Phase T program.
