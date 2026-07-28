# Phase S Data Roles

Date: 2026-07-22
Branch: `phaseS-multitrack-reference-robust-research`

## MoNuSAC (Primary Dataset)

| Split | Cases | Role | Phase S Usage |
|---|---:|---|---|
| Train | 80 | Training, hyperparameter selection, inner CV | Train labels for utility/pairwise/harmfulness; model fitting; β/τ/T/λ/ε selection |
| Validation | 41 | Model selection and Gate evaluation | Track A/B/C/D Gate evaluation; held fixed across all tracks |
| Holdout | 40 | Final frozen-method confirmation | Observed exactly ONCE per frozen method; never used for tuning |
| Formal test | 101 | LOCKED | Not accessed in Phase S |
| Reference pool | 80 cases / 18 patients | 6 refs/query (A=3, B=3, patient-disjoint) | Reused from Phase R manifests |

### Resampling Protocol

- Train: 10 resampling seeds (Phase R used 5; Phase S extends to 10 for Track A benchmark).
- Validation: Fixed A/B allocation from `manifests/phaseR/validation_AB_fixed.csv`.
  Phase S adds 10 pre-registered subset seeds for Track A validation benchmark.
- Reference subset sizes: k ∈ {1, 3, 6} for Track A scaling analysis.

## GlaS (Secondary Pathology Morphology Diagnostic)

| Split | Cases | Role |
|---|---:|---|
| Official train | 55 | Reference pool only |
| Strict validation | 30 | Cross-dataset diagnostic (only after MoNuSAC holdout pass) |
| testA / testB | LOCKED | Not accessed in Phase S |

## DDR (Optional Tertiary Cross-Domain)

| Split | Cases | Role |
|---|---:|---|
| Train | 383 | Reference pool only |
| Valid | 149 | One lesion intent cross-domain feasibility (only after holdout pass) |
| Test | 225 | LOCKED |

## Historical Phase X 10 Queries

Excluded from Phase S primary statistics. May appear only in historical
correction audit (Track A §6.3) as a referenced-but-not-recounted artifact.

## Data Reuse

Phase S reuses from Phase R:
- `manifests/phaseR/train_resampling_seed_{0-4}.csv`
- `manifests/phaseR/validation_AB_fixed.csv`
- `manifests/phaseR/manifest_summary.json`
- DINOv2-L feature cache (CPU-side, per-case)
- Per-reference matching maps `s_fg`, `s_bg`, contrast `C_r`
- Candidate pools (P_TOPK / N_BOTTOMK / N_NONE)
- SAM-B predictor and checkpoint

## Frozen Boundaries

- No patient/case leakage between train / validation / holdout / test.
- Reference pool patients disjoint from query patients within each episode.
- Validation reference allocation fixed; no re-randomization after Gate evaluation.
- Holdout reference allocation to be frozen in `docs/phaseS_holdout_freeze.md`
  before any holdout inference.
