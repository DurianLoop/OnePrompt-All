# Phase R Data Roles

Date: 2026-07-22

## MoNuSAC (Primary)

| Role | Count | Patients | Usage |
|---|---:|---:|---|
| Policy Train Queries | 80 | 18 | R1 instability replication (5 seeds); R2 β/τ selection; R3 reliability model training |
| Policy Validation Queries | 41 | 9 | R2 consensus evaluation; R3 reliability evaluation; frozen A/B |
| Confirmatory Holdout | 40 | 9 | R4 one-shot evaluation (only if R2 or R3 passes) |
| Formal Test | 101 | — | LOCKED. No access. |

## GlaS (Secondary Diagnostic)

| Role | Count | Usage |
|---|---:|---|
| Development | 55 | β/τ config selection (only if MoNuSAC R2 passes) |
| Strict Validation | 30 | Directional evaluation only |
| testA/testB | — | LOCKED. No access. |

## DDR (Optional)

Default: NOT RUN.
Only if MoNuSAC R2 passes AND DDR protocol directly reusable.

## Reference Pool

- Size: 80 cases (same as MoNuSAC train)
- Patients: 18
- Patient-disjoint from all queries
- 6 references per query (A=3, B=3, non-overlapping)

## Data Discipline

- Phase X 10 queries: excluded from primary statistics;
- Phase Y holdout: still unobserved;
- Validation queries: only 1 fixed A/B observation (no multiple observations);
- All splits from Phase Y are reused without modification.
