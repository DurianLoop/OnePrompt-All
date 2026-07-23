# Phase T Dataset Registry

Date: 2026-07-23
Branch: `phaseT-publication-completion`
Directive: `D:\SAM\adapromptset\reports\AdaPromptSet_PhaseT_Publication_Completion.md`

## Dataset Roles

- **Primary**: MoNuSAC — the dataset on which the core phenomenon is
  confirmed via the unobserved holdout.
- **Secondary**: GlaS, MoNuSeg — pathology-morphology replication datasets
  for cross-dataset directional / strong support.
- **Tertiary**: DDR — one frozen lesion intent for cross-domain
  (retinal) replication.

## MoNuSAC (Primary)

| Split | Cases | Role | Phase T Exposure Status |
|---|---:|---|---|
| Train | 80 | Model training, threshold selection, M1/M2 policy fitting, reference-budget utility labels | EXPOSED (training only) |
| Validation | 41 | Method selection, Gate evaluation, M1/M2 validation pass | EXPOSED (validation only) |
| Holdout | 40 | WP1 confirmatory benchmark; final-method one-shot confirmation | UNEXPOSED — hash-verified before WP1 run, observed exactly once |
| Formal test | 101 | LOCKED | LOCKED_UNUSED for entire Phase T |

Reference pool: 80 cases / 18 patients, train-only, patient-disjoint from
queries. Reused from Phase R / Phase Y manifests.

Historical Phase X 10 queries: EXCLUDED from Phase T primary statistics.
May appear only in historical-correction audit as referenced-but-not-recounted
artifact.

## GlaS (Secondary Pathology Morphology)

| Split | Cases | Role | Phase T Exposure Status |
|---|---:|---|---|
| Official train | 55 | Reference pool only (patient-disjoint from queries) | EXPOSED (reference pool) |
| Strict validation | 30 | WP2 cross-dataset replication (only after WP1 pass) | UNEXPOSED until WP2 run |
| testA | 60 | LOCKED | LOCKED_UNUSED |
| testB | 20 | LOCKED | LOCKED_UNUSED |

Phase 2D historical exposure: 30-query strict-validation Gate D confirmation
was run in Phase 2D and FAILED. Phase T WP2 treats GlaS strict validation as
a `development replication` role — the prior Gate D exposure is acknowledged
but the reference-resampling protocol is new, so the strict-validation
queries remain eligible for directional-support statistics under the unified
Phase T protocol.

## MoNuSeg (Secondary Pathology Morphology)

| Split | Cases | Role | Phase T Exposure Status |
|---|---:|---|---|
| Train | 37 | Reference pool only | EXPOSED (reference pool) |
| Test | 14 | LOCKED | LOCKED_UNUSED |

MoNuSeg patient IDs are unreliable; case_id is used as patient_id proxy.
Reference-query pair construction must still enforce `reference_case_id !=
query_case_id`. MoNuSeg serves as a `secondary diagnostic` role for
cross-dataset directional support.

## DDR (Tertiary Cross-Domain Retinal)

| Split | Cases | Role | Phase T Exposure Status |
|---|---:|---|---|
| Train | 383 | Reference pool for one frozen lesion intent | EXPOSED (reference pool) |
| Valid | 149 | One frozen lesion intent cross-domain feasibility (only after WP1 pass) | UNEXPOSED until WP2 run |
| Test | 225 | LOCKED | LOCKED_UNUSED |

Phase 4+ historical exposure: DDR train/valid were used for Route B matcher
smoke and failure-taxonomy analysis in Phase 4+. Phase T WP2 uses one
pre-frozen lesion intent (EX / HE / MA / SE) under the unified protocol; the
intent is selected on the development split and frozen before WP2 inference.
DDR serves as a `development replication` role; directional-support statistics
are reported per-intent.

## Exposure Status Legend

- `EXPOSED (training only)`: cases may be used for fitting, threshold
  selection, and inner CV.
- `EXPOSED (validation only)`: cases may be used for method selection and
  Gate evaluation.
- `EXPOSED (reference pool)`: cases serve as reference images only, never as
  queries.
- `UNEXPOSED until <WP> run`: cases are isolated until the named work package
  executes its frozen inference; hash verification is required before run.
- `LOCKED_UNUSED`: cases are not read, listed, or opened by any Phase T
  script.

## Frozen Boundaries

1. No patient/case leakage between train / validation / holdout / test within
   any dataset.
2. Reference pool is train-only and patient-disjoint from queries within each
   episode.
3. Validation reference allocation is fixed; no re-randomization after Gate
   evaluation.
4. Holdout reference allocation is frozen in
   `docs/phaseT_holdout_protocol.md` before any holdout inference.
5. Cross-dataset prompt config may be dataset-specific but must be frozen on
   the development split before WP2 inference.

## Data Reuse

Phase T reuses from Phase R / Phase Y / Phase S:

- `manifests/phaseY/confirmatory_holdout_queries.csv` (40 holdout queries).
- `manifests/phaseY/reference_pool.csv` (train-only reference pool).
- `manifests/phaseY/policy_train_queries.csv` (80 train queries).
- `manifests/phaseY/policy_validation_queries.csv` (41 validation queries).
- `manifests/phaseR/validation_AB_fixed.csv` (fixed A/B allocation).
- DINOv2-L feature cache (per-case, CPU-side).
- Per-reference matching maps `s_fg`, `s_bg`, contrast `C_r`.
- Candidate pools (P_TOPK / N_BOTTOMK / N_NONE).
- SAM-B predictor and checkpoint.
- Phase S Track A / B / C metric implementations.
