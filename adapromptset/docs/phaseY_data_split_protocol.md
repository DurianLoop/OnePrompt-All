# Phase Y — Data Split Protocol

Date: 2026-07-21

Directive section: 5.

## 1. Source

```text
D:\SAM\code\datasets\MoNuSAC\processed\monusac_processed_manifest.csv
```

310 cases total: 209 train (train_000 … train_208), 101 test (test_000 …
test_100). Test is LOCKED.

Source manifest SHA-256 is recorded in
`reports/phaseY_data_audit/split_summary.json`.

## 2. Split Unit

The split unit is the **patient** (highest recoverable identity tier),
following `patient > slide > case > image`. No patient appears in more than
one fold. This prevents within-patient leakage.

## 3. Phase X Exclusion

Phase X exposed 12 cases across 10 patients. These patients are excluded
from the fresh pool:

| Patient | Exposed cases | Exposure |
|---|---|---|
| TCGA-55-1594 | train_004, train_005 | phasex_reference |
| TCGA-YL-A9WY | train_017 | phasex_query |
| TCGA-UZ-A9PO | train_111, train_113 | phasex_query (2 from 1 patient) |
| TCGA-G9-6499 | train_116 | phasex_query |
| TCGA-P4-AAVK | train_123 | phasex_query |
| TCGA-BH-A18T | train_132 | phasex_query |
| TCGA-G9-6342 | train_190 | phasex_query |
| TCGA-EJ-5495 | train_200 | phasex_query |
| TCGA-L4-A4E5 | train_204 | phasex_query |
| TCGA-86-7713 | train_207 | phasex_query |

**Critical leakage risk identified**: Phase X queries `train_111` and
`train_113` are both from patient `TCGA-UZ-A9PO`. Phase X treated them as
independent. Phase Y splits at patient level to eliminate this risk.

**Critical reference limitation identified**: Phase X references `train_004`
and `train_005` are both from patient `TCGA-55-1594`. All 10 Phase X episodes
used a single patient as reference. Phase Y uses diverse patient-disjoint
references (3+3 per query).

## 4. Fresh Pool

- Excluded patients: 10
- Eligible fresh patients: 36
- Eligible fresh cases: 161

## 5. Split Targets and Actuals

| Fold | Target | Actual | Patients |
|---|---:|---:|---:|
| Policy Train | >= 60 (pref 80) | 80 | 18 |
| Policy Validation | >= 30 (pref 40) | 41 | 10 |
| Confirmatory Holdout | >= 30 (pref 40) | 40 | 8 |
| **Total** | | **161** | **36** |

Split strategy: sort patients by case count desc, shuffle ties with fixed
seed `20260721`, greedily assign to the fold with the largest normalized
deficit. Patient-level disjointness verified.

## 6. Allowed Uses Per Fold

| Fold | Train models | Select models | Eval frozen | Confirmatory stats |
|---|---|---|---|---|
| Policy Train | yes | no | no | no |
| Policy Validation | no | yes | yes | no |
| Confirmatory Holdout | no | no | yes (once) | yes |

## 7. Historical 10 Episodes

The Phase X 10 episodes are labeled `EXPLORATORY_PHASEX_EVIDENCE`. They may
be used only for:

- method discovery
- debug
- action-grid design explanation
- historical appendix
- sanity check

They MUST NOT be used for:

- policy validation primary metric
- confirmatory bootstrap
- formal claim
- model selection
- final significance test

## 8. Test Lock

MoNuSAC formal test (101 cases) remains `LOCKED_UNUSED`. No test access for
K, action bank, router, features, model capacity, or seed selection.

## 9. Output Manifests

```text
manifests/phaseY/policy_train_queries.csv          (80 rows)
manifests/phaseY/policy_validation_queries.csv     (41 rows)
manifests/phaseY/confirmatory_holdout_queries.csv  (40 rows)
manifests/phaseY/reference_pool.csv                (80 rows)
manifests/phaseY/reference_set_A.csv               (483 rows)
manifests/phaseY/reference_set_B.csv               (483 rows)
reports/phaseY_data_audit/split_summary.json
reports/phaseY_data_audit/data_audit_summary.md
```

## 10. Insufficient-Data Rule

If available independent cases had been < 60, the protocol requires pausing
and NOT claiming a policy study. 161 >= 60, so the study proceeds. The
80/41/40 split is within the preferred 80/40/40 target.
