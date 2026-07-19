# Research Track Registry

Date: 2026-07-19

## Route A: Promptability-Aware Routing

Status: `REGISTERED_PENDING_PHASE4_AUDIT`

Core question:

Can test-time-safe evidence predict whether a query is suitable for point, box, mask, direct matching, or abstention, and route accordingly under risk/cost constraints?

Potential contribution:

- Reference-conditioned promptability definition.
- Promptability labels and upper bounds.
- Calibrated predictor with abstention.
- Risk-coverage and cost-quality analysis.

Key gates:

- A1 upper-bound separability.
- A2 promptability prediction.
- A3 router utility.

## Route B: Reference Promptability Benchmark

Status: `REGISTERED_PENDING_PHASE4_AUDIT`

Core question:

When do matching evidence, privileged promptability, and automatic prompting align or fail across datasets, target morphologies, matchers, segmenters, and prompt representations?

Potential contribution:

- Benchmark protocol.
- Promptability metrics: PG, LA, IG, OAV, reliability.
- Failure taxonomy and diagnostic baseline.

Key gate:

- Benchmark breadth, reproducibility, and non-trivial finding.

## Route C: Direct Reference Segmentation

Status: `REGISTERED_PENDING_PHASE4_AUDIT`

Core question:

If sparse prompt interfaces are the bottleneck, can dense reference-query matching evidence directly produce useful segmentation masks?

Potential contribution:

- Multi-prototype reference encoder.
- Multi-reference uncertainty fusion.
- Lightweight dense refiner.
- Component/topology-aware objective.

Key gate:

- Direct method improvement over raw matching and fixed prompt pipelines across at least two datasets.

## Selection Rule

After Phase 4 read-only audit, produce:

- `reports/research_track_selection.md`
- `reports/research_track_selection.json`

The selection must choose one primary, one fallback, and one deferred track using the weighted scoring table in the Phase 4+ instruction.
