# Phase AA Research Question

**Date**: 2026-07-27

## Primary Research Question

> In multi-reference visual prompting segmentation, can the inconsistency between predictions generated from different references serve as a query-level risk signal that reliably identifies catastrophic failures and supports controlled-coverage selective prediction, without requiring additional learning or complex calibration?

## English Version

> Can cross-reference prediction inconsistency serve as a simple, model-light risk signal for identifying catastrophic failures and enabling selective prediction in multi-reference visual prompting?

## What is NOT Being Studied

- Full RAMP weighting mechanisms
- Adaptive fusion strategies
- RRN as a main-text method
- Action bank or point-subset policies
- Reinforcement learning approaches
- Dynamic reference budget allocation
- GT-guided oracle methods
- Large risk models
- Large set encoders

## Hypotheses

1. **H1 (Primary)**: Prediction Consistency (PC) risk score produces AURC significantly lower than random ranking on the primary pipeline.
2. **H2 (Generality)**: PC risk score is effective (AURC < random, CFR@80 reduced) on at least 3/5 tested pipelines.
3. **H3 (Cross-dataset)**: PC risk score transfers to at least one secondary dataset with AUROC >= 0.70.
4. **H4 (Simplicity)**: PC risk score is not outperformed by learned baselines (M2-Instability, Dual-head M2) on the Development Cohort.