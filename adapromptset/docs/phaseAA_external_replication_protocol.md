# Phase AA External Replication Protocol

**Date**: 2026-07-27

## SAM-Med2D Audit

Required documentation:
- Checkpoint source URL
- SHA256 hash
- Loading log
- Inference log
- Per-query predictions
- Reproduction command

If any of the above cannot be provided: REMOVE from main table.

## External Baseline Fairness

### Table A: Shared Backbone (SAM-B)
Only methods using the same segmenter backbone as our primary pipeline.

### Table B: Official Default
Methods using their original backbones. Backbone differences must be explicitly annotated.

### Rules
- No "identical conditions" claims across tables
- No bold "best" across backbone boundaries
- Evaluation protocol must be identical