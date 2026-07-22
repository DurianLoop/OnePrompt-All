# Phase R Problem Statement

Date: 2026-07-22

## Problem

Phase Y demonstrated that the dominant bottleneck in the AdaPromptSet pipeline
is `REFERENCE_INSTABILITY` — different valid reference images produce substantially
different segmentation quality for the same query, even with the same fixed action.

## Evidence

From Phase R0 historical audit:
- Mean best-worst Dice gap across 6 references: 0.1874
- Mean reference std: 0.0687
- 82.9% of validation queries have best-worst gap >= 0.05
- Exact action agreement rate (A vs B): 4.88%
- Cross-reference search Dice: 0.4105 (below fixed baseline 0.4338)

## Hypothesis

Reference-stable aggregation (mean/median/trimmed/penalized/quantile consensus)
and test-time-safe reference reliability estimation can reduce this variance and
improve cross-reference reliability.

## Research Questions

1. RQ1: Is reference instability stable and replicable on train data?
2. RQ2: Is simple deterministic consensus sufficient to reduce instability?
3. RQ3: Is reference reliability learnable from test-time-safe features?

## Scope

- MoNuSAC only (train=80, validation=41, holdout=40);
- GlaS secondary diagnostic (only if R2 passes);
- DDR optional (only if R2 passes and protocol reusable);
- No formal test;
- No SAM2/MedSAM;
- No RL;
- 1 SAM call per query.
