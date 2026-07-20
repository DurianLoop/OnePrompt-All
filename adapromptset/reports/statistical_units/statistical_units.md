# Statistical Unit Audit

Status: `PASS`

Multi-reference episodes are repeated measurements for a query and are not independent samples.
Metrics must be aggregated to query-level before patient/query bootstrap; use patient-cluster bootstrap when multiple queries share a patient.

| Manifest | Episodes | Queries | Patients | Max refs/query | Independent unit |
|---|---:|---:|---:|---:|---|
| manifests/episodes/monusac/episodes_train.csv | 810 | 162 | 35 | 5 | patient_cluster |
| manifests/episodes/monusac/episodes_val.csv | 190 | 38 | 9 | 5 | patient_cluster |
| manifests/episodes/monusac/episodes_test.csv | 505 | 101 | 25 | 5 | patient_cluster |
| manifests/episodes/monuseg/episodes_val.csv | 8 | 8 | 8 | 1 | query |
| manifests/episodes/glas/episodes_val.csv | 1 | 1 | 1 | 1 | query |

Decision: report query-level metrics; do not treat reference episodes as independent samples.
