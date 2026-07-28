# Route B Protocol v1: Reference Promptability Benchmark

Date: 2026-07-19

Primary track: `Reference Promptability Benchmark`

Working title: `MedRefPromptBench: A Promptability Atlas for Reference-Guided Medical Segmentation`

## Core Claim Candidate

Reference-guided promptability must be measured separately from candidate precision, privileged upper bounds, and absolute proposal quality. Matched controls and prompt-interface diagnostics reveal when automatic prompting is not justified.

This is a benchmark/diagnostic claim, not a new automatic prompting success claim.

## Phase B0: Read-Only Foundation

Use existing artifacts:

- MoNuSAC Gate F / oracle-analysis / representation artifacts.
- GlaS Phase 2D grouped-point confirmation.
- GlaS Phase 3 R1 region proposal artifacts.
- Phase 4 read-only audit.

No new model inference occurs in B0.

## Phase B1: Fresh Development Expansion

Candidate fresh data:

- DDR train/valid, four-intent EX/HE/MA/SE positive-query protocol.
- MoNuSeg secondary pathology validation after split/exposure re-audit.
- GlaS testA/testB remain locked until final benchmark protocol freeze.

Before B1 inference:

- create dataset-specific protocol files;
- create manifests and hashes;
- run leakage/exposure audits;
- freeze prompt representations and controls;
- update compute ledger.

## Required Prompt Representations

- point / top-k point;
- grouped point if fresh protocol allows;
- box;
- multi-box;
- hard mask;
- soft mask where legal;
- direct matching mask;
- privileged GT point/box/mask upper bounds separated from automatic methods;
- matched random, size-matched random, full-image, and alternative-reference controls.

## Benchmark Metrics

- Promptability Gap: privileged utility minus automatic utility.
- Localization Advantage: automatic utility minus matched-control utility.
- Interface Gain: segmenter output minus raw proposal.
- Oracle Adaptation Value: per-query best representation minus global best representation.
- Promptability Reliability: calibration between predicted viability and observed viability.

## Submission Gate for Route B

Route B cannot be called benchmark-ready until it has:

1. at least three datasets or task domains;
2. at least six model/representation combinations;
3. at least 300 independent query units, or an explicit limitation if the local data cannot support this;
4. matched controls and privileged upper bounds for every main prompt type;
5. a failure taxonomy that is reproducible at query/case level;
6. at least one calibration or diagnostic baseline;
7. a non-trivial cross-domain finding;
8. clean provenance and no formal-test leakage.

If these conditions cannot be met with local data and approved compute, Route B status becomes `TRACK_B_INSUFFICIENT_BREADTH` or `TRACK_B_INSUFFICIENT_FINDING`, and the state machine pivots to Route A.
