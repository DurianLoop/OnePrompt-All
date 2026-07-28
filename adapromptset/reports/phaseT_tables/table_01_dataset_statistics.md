# Table 1 — Dataset statistics

Datasets, splits, and roles used in the Phase T reference-instability benchmark and risk-aware mitigation study.

| Dataset | Role | Cases | Query split | Reference pool | Prior exposure | Phase T status |
| --- | --- | --- | --- | --- | --- | --- |
| MoNuSAC | Primary benchmark | 80+41+40+101 | Holdout (40) | Train (80) | Phase X/Y/R/S training+val | OBSERVED (WP1 PASS) |
| MoNuSAC (test) | Locked formal test | 101 | Locked | — | Never accessed | LOCKED_UNUSED |
| MoNuSeg (KMoNuSeg) | Secondary diagnostic (WP2) | 82 | Validation (30) | 52 (patient-disjoint) | Not used under ref-resampling protocol | UNEXPOSED until WP2 |
| IDRID | Development replication (WP2) | 54 | Validation (30) | 24 (patient-disjoint) | Phase 4+ Route B matcher smoke only | UNEXPOSED until WP2 |
| GlaS | Development replication (planned) | 0 (data missing) | — | — | Phase 2D reference pool | DATA_MISSING |

**Notes:**
- Statistical unit is the query / case (never the reference episode).
- MoNuSAC holdout (40 queries) was unobserved prior to WP1; one-shot confirmation only.
- MoNuSeg and IDRID were not used in any prior reference-resampling phase; their reference pools are patient-disjoint from their query splits.
- GlaS directory is empty on disk; GlaS is documented as DATA_MISSING.
- Formal test sets (MoNuSAC test, MoNuSeg official test, IDRID test, GlaS testA/testB) remain LOCKED_UNUSED throughout Phase T.
