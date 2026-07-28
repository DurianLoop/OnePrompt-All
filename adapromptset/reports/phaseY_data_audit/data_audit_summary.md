# Phase Y — Data Audit Summary

Directive: `D:\SAM\adapromptset\AdaPromptSet_PhaseY_Policy_Feasibility.md` section 5.1

## 1. MoNuSAC Processed Manifest

- Source: `D:\SAM\code\datasets\MoNuSAC\processed\monusac_processed_manifest.csv`
- Total cases: 310
- Train cases: 209 (train_000 … train_208)
- Test cases: 101 (test_000 … test_100) — LOCKED
- Unique train patients: 46
- Unique test patients: 25

## 2. Phase X Prior Exposure

- Phase X query cases (10): train_111, train_190, train_132, train_204, train_200, train_116, train_123, train_207, train_113, train_017
- Phase X reference cases (2): train_004, train_005
- Phase X query patients (9): TCGA-86-7713, TCGA-BH-A18T, TCGA-EJ-5495, TCGA-G9-6342, TCGA-G9-6499, TCGA-L4-A4E5, TCGA-P4-AAVK, TCGA-UZ-A9PO, TCGA-YL-A9WY
- Phase X reference patients (1): TCGA-55-1594

**Critical finding**: Phase X reference cases `train_004` and `train_005` are
both from patient `TCGA-55-1594` — the historical reference pool used a single
patient for all 10 episodes. Phase Y must use diverse, patient-disjoint references.

**Patient-level leakage risk**: Phase X queries `train_111` and `train_113` are
both from patient `TCGA-UZ-A9PO`. Phase Y splits at patient level to prevent
any within-patient leakage between train / validation / holdout.

## 3. Eligible Fresh Query Pool

- Exposed patients (excluded): 10
- Fresh patients (eligible): 36
- Fresh cases (eligible): 161

## 4. Image / Mask Pairing

- Paired (image+mask exist): 310
- Missing image or mask: 0

## 5. Duplicate Hash Audit

- No duplicates found by 8KB head hash.

## 6. Patient-Level Split (target 80/40/40)

- train: 80 cases across 18 patients
- val: 41 cases across 9 patients
- holdout: 40 cases across 9 patients
- Total: 161 cases

Split strategy: sort patients by case count desc, shuffle ties with fixed seed
(20260721), greedily assign to the fold with the largest normalized
deficit. Patient-level disjoint: no patient appears in more than one fold.

## 7. Reference Sets A / B

- Reference pool: 80 train cases (all train-fold cases)
- Reference Set A rows: 483 (3 per query)
- Reference Set B rows: 483 (3 per query)
- Total queries with references: 161

Constraints enforced:
- Reference case comes from train fold only (no val/holdout as reference)
- Reference patient != query patient (patient-disjoint)
- Set A and Set B disjoint per query (6 distinct references)
- Fixed seed: 20260721
- No result-based reference selection

## 8. Output Files

### Audit (reports/phaseY_data_audit/)
- `image_mask_pairing.csv`
- `case_identity_map.csv`
- `prior_query_exposure.csv`
- `prior_reference_exposure.csv`
- `eligible_fresh_queries.csv`
- `duplicate_hash_audit.csv`
- `data_audit_summary.md`
- `split_summary.json`

### Manifests (manifests/phaseY/)
- `policy_train_queries.csv`
- `policy_validation_queries.csv`
- `confirmatory_holdout_queries.csv`
- `reference_pool.csv`
- `reference_set_A.csv`
- `reference_set_B.csv`

## 9. Status

```text
PHASEY_DATA_AUDIT_COMPLETE
```

No SAM inference was run. All outputs are manifest-level and read-only over
the MoNuSAC processed data tree.