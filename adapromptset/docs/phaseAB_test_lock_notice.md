# Phase AB Test Lock Notice

**Date**: 2026-07-27
**Branch**: phaseAB-fresh-confirmation-paper-assembly

## Locked Resources

The following data must NOT be accessed for confirmatory purposes:

1. **MoNuSAC 80 queries** (train_001-train_080): Development Cohort — fully observed
2. **MoNuSAC 41 queries** (train_081+): Validation Cohort — used for M2 model selection
3. **MoNuSeg**: External, observed in Phase T
4. **IDRID**: External, observed in Phase T
5. **Formal test data**: LOCKED — requires user authorization

## What Remains Fresh

No data currently qualifies as fresh confirmatory cohort.

## If New Data Becomes Available

Before accessing:
1. Create separate test_lock_notice for that dataset
2. Freeze all method parameters
3. Preregister hypothesis and analysis plan
4. Get user approval
5. Run ONCE, no tuning