# Phase AA Data Role Protocol

**Date**: 2026-07-27

## Data Cohorts

### Development Cohort (80 queries)
- **Source**: MoNuSAC training set
- **Query IDs**: train_001 through train_080
- **Role**: All method development, feature design, threshold selection, model training
- **Status**: NOT a holdout; renamed from "holdout" to "Development Cohort"
- **Usage**: Grouped 5-fold CV for development evaluation only

### Validation Cohort (41 queries)
- **Source**: M2 validation split
- **Role**: Held out from M2 training; never used in paper results
- **Status**: Candidate for confirmatory evaluation (pending AA1 audit)

### Confirmatory Cohort
- TBD after AA1 audit
- Must be data never observed during method design

## Rules

1. No method development on confirmatory data
2. Confirmatory evaluation is one-shot only
3. All thresholds and hyperparameters frozen before confirmatory run
4. No result-dependent method changes after confirmatory run