# GlaS Exposure Audit for Phase 2

Date: 2026-07-19
Dataset root: `D:\SAM\external\research_data\extracted\GlaS\Warwick QU Dataset (Released 2016_07_08)`

## Local Data Inventory

| Source split | Image-mask pairs |
| --- | ---: |
| official train | 85 |
| official testA | 60 |
| official testB | 20 |

`Grade.csv` is present and records case name, patient ID, GlaS grade, and Sirinukunwattana et al. grade. Across all 165 rows, GlaS grade counts are `benign=74` and `malignant=91`.

## Patient-ID Boundary

Official train contains 15 patient IDs. Official testA/testB contain 12 patient IDs. The following patient IDs occur in both official train and official test: `1,2,3,4,5,7,9,10,11,13,15`.

Therefore Phase 2 can create an official-train-only 55/30 diagnostic split, but it cannot honestly claim patient-level isolation between official train and official testA/testB. The formal-test plan must record this as a limitation unless a different clean dataset role is approved.

## Prior Exposure Status

Current repository artifacts show that GlaS testA/testB have already appeared in dataset manifests and read-only audits:

- `manifests/glas_manifest.csv` includes official testA/testB rows.
- `manifests/episodes/glas/episodes_test.csv` exists from the previous one-episode smoke setup.
- `reports/data_profile_audit.json` profiled GlaS rows including image size and mask-area style summaries.
- `reports/reference_exposure_audit.json` checked reference leakage for existing GlaS episode manifests.

No current evidence found in the Phase 2 read set shows that GlaS testA/testB query-level segmentation outputs, action utilities, thresholds, policy features, or paper claim decisions were used for method selection. Still, because test image/mask metadata and profiles were read, Phase 2 must not describe testA/testB as fully blind. They are best treated as locked, no-retuning evaluation splits unless the user requires a stricter untouched formal-test standard.

## Phase 2 Exposure Rule

- Official train may be split into 55 policy/action-utility training images and 30 strict validation query images.
- Validation references must come only from the 55-image training pool.
- testA/testB must not be used for candidate design, action-bank redesign, threshold tuning, policy training, policy selection, title/claim selection, or qualitative case selection.
- If testA/testB are later used, the run must be preceded by `docs/glas_test_freeze.md`, `configs/frozen/glas_final_protocol.yaml`, and `reports/phase2_test_freeze_hashes.json`.

## Current Exposure Decision

Status: `TEST_SPLITS_LOCKED_NOT_FULLY_BLIND`

Reason: testA/testB have prior manifest/profile exposure but no observed method-selection use in the audited artifacts.

## Phase 2 Manifest Status

The new Phase 2 manifest is official-train-only for train/validation:

- `utility_train=55`
- `strict_val=30`
- `locked_testA=60`
- `locked_testB=20`
- `episodes_train=165`
- `episodes_val=90`

Independent leakage verification found no support outside the utility-train pool, no self-reference, and no locked test query in train/validation episodes.
