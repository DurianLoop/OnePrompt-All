# Phase Y — Reference Protocol

Date: 2026-07-21

Directive section: 5.3.

## 1. Two Non-Overlapping Reference Sets Per Query

For every query in train + validation + holdout, pre-register two
mutually-exclusive reference sets:

```text
Reference Set A: 3 references
Reference Set B: 3 references
```

6 distinct references per query total.

## 2. Constraints

| Constraint | Rule |
|---|---|
| Reference source | train fold only (no val/holdout/test as reference) |
| Reference case != query case | enforced |
| Reference patient != query patient | enforced (patient-disjoint) |
| Set A and Set B disjoint per query | enforced |
| Sampling seed | `20260721` (frozen) |
| Result-based reference selection | FORBIDDEN |

## 3. Why Two Sets

- **Set A**: primary action-utility matrix and policy feature source.
- **Set B**: reference-stability and cross-reference oracle evaluation.

The cross-reference design tests whether oracle gaps survive independent
references. If the oracle gap collapses when actions chosen on Set A are
evaluated on Set B, the gap was reference-sampling noise (Winner's Curse).

## 4. Aggregation

Multiple references are first aggregated WITHIN a query (mean Dice over the 3
references in the set). The statistical unit remains the QUERY, not the
reference episode. Reference episodes are NOT independent samples.

## 5. Phase X Reference Audit

Phase X used `train_004` and `train_005` as references for all 10 episodes.
Both come from patient `TCGA-55-1594` — a single patient. This is recorded
as a historical limitation. Phase Y's reference pool has 80 cases across 18
patients, and every query's references are patient-disjoint from the query.

## 6. Output Files

```text
manifests/phaseY/reference_pool.csv     (80 rows — all train cases)
manifests/phaseY/reference_set_A.csv    (483 rows — 3 per query × 161 queries)
manifests/phaseY/reference_set_B.csv    (483 rows — 3 per query × 161 queries)
reports/phaseY_reference_audit.json     (constraint verification)
```

## 7. Reference Audit Status

```text
PHASEY_REFERENCE_AUDIT_PASS
```

Verified:

- 0 Set A / Set B overlaps per query
- 0 reference-patient == query-patient violations
- all references sourced from train fold
- fixed seed recorded

## 8. Re-Sampling Forbidden

Once the preregistration commit lands, the reference sets are frozen. No
re-sampling of references based on results. If a reference is found
corrupt (e.g., image file missing), it is replaced using the same frozen
seed with the next eligible candidate, and the replacement is logged in
`docs/decision_log.md`.
