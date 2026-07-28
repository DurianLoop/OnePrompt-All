# Phase R Reference Resampling Protocol

Date: 2026-07-22

## Protocol

Per directive section 5:

### 5.1 References per query

Each MoNuSAC query uses 6 train-only references:
- Reference Set A = 3 references
- Reference Set B = 3 references
- A and B are completely non-overlapping
- References are patient-disjoint from query
- All methods share the same references
- Query is the statistical unit
- Reference episode is NOT an independent sample

### 5.2 Resampling

**Train (80 queries):** 5 different reference resampling seeds
- Seeds: [20260722, 20261722, 20262722, 20263722, 20264722]
- Each seed produces 6 references, split into A=3/B=3
- Used for: instability estimation, reliability training
- NOT used for validation model selection

**Validation (41 queries):** 1 fixed, preregistered A/B split
- Seed: 20260722
- Prevents multiple observations on validation

## Manifests

- `manifests/phaseR/train_resampling_seed_0.csv` through `seed_4.csv`
- `manifests/phaseR/train_resampling_all_seeds.csv` (combined)
- `manifests/phaseR/validation_AB_fixed.csv`

## Verification

- A/B non-overlap: VERIFIED for all query×seed pairs
- Patient-disjoint: VERIFIED for all rows
- Seed diversity: 80/80 queries have different reference sets across 5 seeds
