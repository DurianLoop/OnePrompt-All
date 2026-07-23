# Phase T Reference Construction Protocol

Date: 2026-07-23
Branch: `phaseT-publication-completion`
Directive: `D:\SAM\adapromptset\reports\AdaPromptSet_PhaseT_Publication_Completion.md`

## Purpose

This protocol defines how reference images are selected, pooled, and resampled
for every Phase T query. It applies identically to WP1 (MoNuSAC holdout),
WP2 (GlaS / MoNuSeg / DDR cross-dataset), and WP3 (M1 / M2 risk-aware
mitigation). It is frozen at preregistration; no reference-construction
parameter may be changed after the holdout hash is verified.

## Reference Pool

- The reference pool is **train-only**. No validation, holdout, or test case
  may serve as a reference.
- The reference pool is **patient-disjoint** from the query pool. Within any
  single episode, the reference patient must differ from the query patient.
- Pool composition per dataset is recorded in
  `docs/phaseT_dataset_registry.md` and reused from Phase R / Phase Y
  manifests:
  - MoNuSAC: 80 train cases / 18 patients (reused from
    `manifests/phaseY/reference_pool.csv`).
  - GlaS: 55 official-train cases.
  - MoNuSeg: 37 train cases.
  - DDR: 383 train cases (one frozen lesion intent).

## References per Query

- Each query receives **6 references** drawn from the train-only reference
  pool, patient-disjoint from the query.
- The 6 references are the upper bound on what M1 may select from; the actual
  subset size k is drawn from k∈{1,3,6} per the resampling protocol below.
- All methods (B1–B7 internal, external comparators, M1, M2) evaluated on the
  same query must share the **exact same 6-reference pool** for that query.
  Methods differ only in how they aggregate or select from the pool, not in
  the pool itself.

## Reference Subset Sizes

- k ∈ {1, 3, 6} — the subset sizes drawn from the 6-reference pool.
- k=1: single-reference baseline (B1 random single).
- k=3: intermediate consensus.
- k=6: full-pool consensus (B2 mean / B3 median / B4 trimmed mean / B5 / B6 /
  B7 learned weighting).
- The k=6 subset IS the full pool; no method may see more than 6 references.

## Resampling Seeds

- **Train**: 10 resampling seeds (Phase R used 5; Phase S extended to 10;
  Phase T reuses the Phase S 10-seed allocation). Each seed produces a
  different k-subset for each query, enabling within-query reference-variance
  estimation.
- **Validation**: fixed seeds. Validation reference allocation is fixed in
  `manifests/phaseR/validation_AB_fixed.csv` (A/B allocation) and is NOT
  re-randomized after Gate evaluation.
- **Holdout**: fixed seeds. Holdout reference allocation is frozen in
  `docs/phaseT_holdout_protocol.md` and hash-verified before WP1 inference.
  Holdout seeds are generated once and never regenerated.
- **Cross-dataset (GlaS / MoNuSeg / DDR)**: 10 resampling seeds on each
  dataset's development split, frozen before WP2 inference.

## Reference-Query Pair Constraint

For every (query, reference) pair in every episode:

1. `reference_case_id != query_case_id` — the reference and query must not be
   the same case.
2. `reference_patient_id != query_patient_id` — the reference and query must
   not share a patient.
3. `reference_role == "train"` — the reference must come from the train-only
   pool.
4. `query_role in {"train", "validation", "holdout"}` — the query may come
   from any non-test role.
5. `query_role != "test"` — test cases never serve as queries in Phase T.

This constraint is enforced by `scripts/audit_reference_exposure.py` (from
Phase 0/1) on every regenerated manifest. A failed audit blocks inference.

## Cost Accounting

Each reference induces:

- 1 DINOv2-L matching pass (cached per case_id; reused across seeds).
- 1 candidate-pool construction (cached per case_id).
- 1 SAM-B mask decode per consensus aggregation (NOT per reference; consensus
  is single-SAM-call by construction for B2 mean).

The cost ledger (`reports/phaseT_compute_ledger.csv`) records:

- `sam_calls`: number of SAM-B mask decodes.
- `matching_passes`: number of DINOv2-L forward passes (cached + uncached).
- `runtime_seconds`: wall-clock runtime.
- `gpu_hours`: GPU-hours consumed.

## Prohibited Reference Operations

1. No reference selection based on query ground truth at inference time.
   (B7 learned weighting uses train-only utility labels; the model is frozen
   before holdout.)
2. No reference re-ordering to favor a method.
3. No reference-pool expansion after the holdout hash is verified.
4. No cross-dataset reference sharing (a MoNuSAC query never receives a GlaS
   reference, and vice versa).
5. No test-set case may enter the reference pool under any circumstance.
