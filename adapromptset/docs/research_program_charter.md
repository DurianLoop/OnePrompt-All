# Research Program Charter

Program: Phase 4+ -- From Evidence Audit to Submission-Ready Research

Working directory: `D:\SAM`

Project directory: `D:\SAM\adapromptset`

## Mission

Advance AdaPromptSet from evidence audit toward a submission-ready research result while preserving formal-test locks, preventing validation overfitting, and allowing honest negative, benchmark, deterministic, promptability, or direct-segmentation outcomes.

The program is not complete when a paper compiles. It is complete only when the submission-readiness gate is satisfied or when all legal registered routes are exhausted and the state becomes `RESEARCH_PROGRAM_EXHAUSTED_NOT_SUBMISSION_READY`.

## Frozen Historical Boundaries

| Prior route | Frozen state | Consequence |
| --- | --- | --- |
| MoNuSAC adaptive prompt-set policy | `PIVOT_ORACLE_ANALYSIS` | No MoNuSAC policy training, no Gate-F threshold lowering, no post-hoc action-bank redesign. |
| GlaS grouped sparse points | `STOP_GLAS_GROUPED_POINT_PROMPT_PATH_CONFIRMED` | No P3 point-count/fusion rescue, no second Gate-D revision, no third segmenter rescue. |
| GlaS automatic regions | `STOP_GLAS_REGION_LOCALIZATION_ADVANTAGE_NOT_ESTABLISHED` | R1 failed; no same-validation action/grid expansion or test-driven region selection. |
| GlaS formal tests | `LOCKED_UNUSED` | testA/testB only after a frozen final method and test-freeze protocol. |

## Allowed Research Forms

- New method paper.
- Benchmark or diagnostic framework paper.
- Failure-boundary / negative empirical study.
- Deterministic method paper.
- Reliability-aware routing method.
- Frozen-feature direct reference segmentation method.

## Initial Candidate Tracks

Primary candidate by default, pending Phase 4 audit:

- Route A: Promptability-Aware Routing / AdaPromptability.

Fallback candidates:

- Route B: Reference Promptability Benchmark / MedRefPromptBench.
- Route C: Direct Reference Segmentation / Match2Mask.

Track selection is not decided by this charter. It must be scored after Phase 4 read-only audit.

## Non-Negotiable Constraints

- No formal test access for route selection.
- No testA/testB use before method freeze.
- No privileged GT prompt mixed with automatic method rows.
- No lowering frozen gates.
- No post-hoc claim expansion from negative evidence.
- No network/download/new model without user approval.
- No `SUBMISSION_READY` state before novelty, baseline, ablation, statistics, formal test, reproducibility, paper, supplementary, and mock-review gates are satisfied.
