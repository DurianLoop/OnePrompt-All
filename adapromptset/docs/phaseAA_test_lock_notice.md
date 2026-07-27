# Phase AA Test Lock Notice

**Date**: 2026-07-27

## WARNING

This document serves as a formal notice that:

1. All confirmatory data must be locked before any confirmatory experiment runs
2. No method parameters may be changed after confirmatory data is observed
3. Any violation of the test lock invalidates the confirmatory status of the data

## Locked Items

- Risk score definition (Prediction Consistency)
- Reference count (m)
- Prompt configuration
- Consensus method (mean-consensus)
- Coverage levels
- Catastrophic failure threshold
- All evaluation metrics
- Confidence interval method

## Preregistration

The preregistration commit (to be created) serves as the official lock point.
No confirmatory experiments may run before this commit.