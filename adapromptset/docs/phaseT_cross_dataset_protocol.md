# Phase T Cross-Dataset Protocol

Date: 2026-07-23
Branch: `phaseT-publication-completion`
Directive: `D:\SAM\adapromptset\reports\AdaPromptSet_PhaseT_Publication_Completion.md`
Related: directive §6 (WP2), §6.3 (unified protocol)

## Purpose

WP2 must demonstrate that reference-induced instability is NOT a MoNuSAC
artifact. The phenomenon must replicate on at least two additional datasets
under a unified protocol, with dataset-specific prompt config allowed but
frozen on the development split.

## Datasets

### GlaS (Secondary Pathology Morphology)

- Official train: 55 cases (reference pool only).
- Strict validation: 30 cases (cross-dataset replication queries).
- testA / testB: LOCKED.

Role: `development replication`. The Phase 2D historical Gate D exposure is
acknowledged but the reference-resampling protocol is new, so the
strict-validation queries remain eligible for directional-support statistics
under the unified Phase T protocol.

### MoNuSeg (Secondary Pathology Morphology)

- Train: 37 cases (reference pool only).
- Test: 14 cases — LOCKED.

Role: `secondary diagnostic`. Patient IDs are unreliable; case_id is used as
patient_id proxy. Reference-query pair construction still enforces
`reference_case_id != query_case_id`.

### DDR (Tertiary Cross-Domain Retinal)

- Train: 383 cases (reference pool).
- Valid: 149 cases (cross-domain replication queries).
- Test: 225 cases — LOCKED.

Role: `development replication`. Phase 4+ historical Route B matcher smoke
exposure is acknowledged. Phase T WP2 uses ONE pre-frozen lesion intent
(EX / HE / MA / SE) under the unified protocol; the intent is selected on
the development split and frozen before WP2 inference.

## Unified Protocol (directive §6.3)

All datasets share the following:

| Component | Value |
|---|---|
| Matcher | DINOv2-L (frozen) |
| Segmenter | SAM-B (frozen) |
| SAM calls | 1 per consensus aggregation |
| Reference counts | k ∈ {1, 3, 6} |
| Aggregation | single / mean / median / best learned frozen |
| Metrics | V_ref / MAD / J_pred / CVaR / CFR / Dice |
| Statistical unit | query / case |
| Bootstrap | 10,000 iterations, 95% CI |

## Dataset-Specific Prompt Config

Dataset-specific prompt config is ALLOWED but must be:

1. Frozen on the dataset's development split BEFORE WP2 inference.
2. Recorded in `reports/phaseT_cross_dataset_audit/dataset_role_decision.md`.
3. Clearly distinguished in the report as a `dataset-specific setting`
   versus a `universal setting`.
4. NOT tuned on the formal test split (testA / testB / DDR test are locked).
5. NOT tuned on a different dataset's queries (no cross-dataset
   hyperparameter transfer).

## Cross-Dataset Audit (directive §6.2)

`reports/phaseT_cross_dataset_audit/` must contain:

```
glas_availability.md
monuseg_availability.md
ddr_availability.md
exposure_matrix.csv
dataset_role_decision.md
dataset_manifest_hashes.json
```

The exposure matrix records, for every (dataset, split, prior-phase) tuple,
whether the split was previously observed and in what role. This prevents
treating a historically observed validation split as a fresh holdout.

## Cross-Dataset Gate (directive §6.4)

Each additional dataset is judged SEPARATELY.

### Directional Support

ALL of the following:

1. ≥ 30% queries with best-worst gap ≥ 0.05.
2. Mean reference std ≥ 0.015.
3. Mean consensus Dice not below random single.
4. Catastrophic failure rate not increased.
5. Prediction consistency not decreased.

State:

```
PHASET_DATASET_DIRECTIONAL_SUPPORT
```

### Strong Support

ALL of the following (in addition to directional support):

1. Mean consensus Dice ≥ random single + 0.01.
2. Bootstrap point estimate positive.
3. CVaR improved.
4. Instability feature correlation ≥ 0.20.

State:

```
PHASET_DATASET_STRONG_SUPPORT
```

## Minimum Benchmark Submission Requirement

```
MoNuSAC holdout confirmed (WP1 pass)
+ at least 2 additional datasets directional support
```

This is the minimum for the Benchmark-only submission route (§13). The
Hybrid route additionally requires M1 OR M2 holdout pass.

## Prohibited Cross-Dataset Operations

1. Accessing testA / testB / MoNuSeg test / DDR test.
2. Tuning a dataset's prompt config on its own validation split and then
   reporting validation metrics as if frozen.
3. Treating a historically observed validation split as a fresh holdout
   without acknowledging prior exposure.
4. Sharing references across datasets (a MoNuSAC query never receives a GlaS
   reference).
5. Claiming "universal across datasets" (prohibited claim #2) — at most
   `directional support` or `strong support` may be claimed per dataset.
6. Pooling queries across datasets for an aggregate significance test
   (statistical unit = query within dataset; cross-dataset synthesis is
   qualitative unless a formal meta-analytic protocol is preregistered).
