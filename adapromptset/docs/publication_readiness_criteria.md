# Publication Readiness Criteria

Status names:

- `SUBMISSION_READY`
- `RESEARCH_PROGRAM_EXHAUSTED_NOT_SUBMISSION_READY`
- intermediate registered states from the Phase 4+ state machine

## Submission-Ready Requirements

`SUBMISSION_READY` requires strong evidence across most of the following, and no core leakage or claim-boundary failure:

| Area | Required evidence |
| --- | --- |
| Research question | Clear, falsifiable, and generally meaningful beyond one threshold heuristic. |
| Novelty | Difference from GF-SAM, SynPo, AlignSAM, Point Prompt Defender, reference-guided SAM, prompt routing, and selective prediction is explicit. |
| Formal testing | At least one untouched formal test set evaluated only after freeze. |
| Breadth | At least three meaningful datasets/tasks, or two datasets plus strong domain/model transfer. |
| Baselines | Strong deterministic, matched-control, privileged upper-bound, and route-specific baselines. |
| Ablation | Controlled feature/objective/representation/cost/reference ablations. |
| Statistics | Query/case-level paired analysis, confidence intervals, win/tie/loss, cost and failure analysis. |
| Seeds | Main learned conclusions do not rely on a single seed. |
| Claims | Every claim maps to structured artifact evidence. |
| Reproducibility | One-command pipeline, manifests, hashes, configs, checkpoints, and compute ledger. |
| Paper | Clean 8-page main paper plus supplementary; figures/tables readable and provenance-audited. |
| Mock review | At least two rounds; major reviewer concerns addressed or converted to limitations. |

## Explicit Non-Completion Conditions

The following are not sufficient:

- A compiled PDF.
- A single-dataset validation result.
- A privileged GT upper bound.
- A small pilot without formal test.
- An unverified reviewer-style narrative.
- A green test suite that does not cover leakage, claims, and result provenance.
