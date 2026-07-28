# Fresh Confirmatory Cohort Decision

**Date**: 2026-07-27
**Gate**: WP1

## Audit Result

After auditing all available and candidate datasets:

### No Fresh Cohort Available

All datasets that have been evaluated:
- **MoNuSAC 80 queries**: Development Cohort — fully observed
- **MoNuSAC 41 queries**: Validation Cohort — used for M2 model selection
- **MoNuSeg 30 queries**: External — observed in Phase T, CFR=0 (no failures to detect)
- **IDRID 30 queries**: External — observed in Phase T, segmentation quality too low

### Candidate Datasets Requiring User Action

- **GlaS**: Gland segmentation, different organ but same modality. Need to check availability, exposure, and labels.
- **CPM17**: Nuclei segmentation, same task. Small dataset (17 images). Need to check availability.
- **Formal Test**: Locked — requires explicit user authorization.

## Decision

### PHASEAB_BLOCKED_NO_FRESH_CONFIRMATORY_DATA

**Consequences**:
1. Cannot run WP2 (Confirmatory Experiment)
2. Paper is exploratory/benchmark, not confirmatory
3. Cannot use "holdout-confirmed" or "confirmatory" language
4. Target venue expectations must be adjusted
5. External datasets serve as secondary validation only

## Recommended Path Forward

1. **Option A**: User provides fresh data → resume confirmatory pipeline
2. **Option B**: Proceed with exploratory/benchmark paper → honesty about limitations becomes a strength
3. **Option C**: Wait for formal test access → may delay submission

**Current recommendation**: Option B. The method is simple, the evidence is honest, and the problem formulation is novel. This is acceptable for venues that accept benchmark/analysis papers.