# Phase T Goal Read Receipt

Date: 2026-07-23
Branch: `phaseT-publication-completion`
Base branch: `phaseS-multitrack-reference-robust-research`
Base commit: `2c1f7d2`
Directive: `D:\SAM\adapromptset\reports\AdaPromptSet_PhaseT_Publication_Completion.md`

## Acknowledgement

CX acknowledges receipt of the Phase T directive and the `/goal` mode execution
mandate. Phase T is a publication completion program that advances the Phase S
core finding — reference-induced instability is real, predictable, but cannot
be mitigated through reference weighting (Track B NO_GAIN) or set conditioning
(Track C NO_GAIN) — into a submission-ready Benchmark + risk-aware mitigation
study with confirmatory holdout, cross-dataset replication, external baseline
comparison, and complete paper materials.

## Goal

> Use an unobserved holdout and multiple real datasets to confirm
> reference-induced instability, establish a reproducible evaluation protocol,
> demonstrate the strong-baseline value of simple mean consensus, and provide
> at least one risk-aware practical mitigation or reliable failure detection,
> so that the whole study forms a clear, credible, submittable Benchmark or
> Hybrid paper.

## Four Registered Work Packages

1. **WP1 — Confirmatory Benchmark** — Confirm the Phase S finding on the
   unobserved MoNuSAC holdout (n=40) under a frozen protocol.
2. **WP2 — Cross-Dataset and Cross-Method Replication** — Replicate the
   phenomenon on GlaS / MoNuSeg / DDR and build an external comparator matrix.
3. **WP3 — Risk-Aware Mitigation** — Two parallel routes:
   - **M1** Dynamic Reference Budget (k∈{1,3,6} selection).
   - **M2** Instability-Aware Selective Prediction (stable / unstable /
     catastrophic-risk classification).
4. **WP4 — Paper Construction and Mock Review** — Full paper draft, novelty
   audit, mock review, and Publication Readiness Gate.

## Compute Budget Acknowledgement

- Total GPU-hour cap: **195 GPU-hours**.
- Used at preregistration: **0 GPU-hours**.
- Checkpoints: 40 / 90 / 140 / 195 GPU-hours.
- At 195 GPU-hours the program stops and user approval is required to proceed.
- Compute reuse is mandatory: DINOv2 features, matching maps, candidate pools,
  SAM output cache, and Phase R/S manifests must be reused wherever possible.

## Execution Order (directive §26)

1. Read Phase X / Y / R / S artifacts.
2. Create `phaseT-publication-completion` branch.
3. Create publication charter and claim registry.
4. Freeze MoNuSAC holdout benchmark protocol.
5. Run WP1.
6. If WP1 fails → STOP; do not access formal test.
7. Audit GlaS / MoNuSeg / DDR.
8. Run at least two additional datasets.
9. Build external comparator matrix.
10. Reproduce available external methods.
11. Train M2 selective prediction.
12. Train M1 dynamic reference budget.
13. Freeze final mitigation after validation pass.
14. Run one-shot confirmation on legal holdout.
15. Complete cross-dataset results.
16. Generate all statistics, tables, and figures.
17. Update novelty audit.
18. Complete mock review.
19. Generate full paper draft.
20. Execute Publication Readiness Gate.
21. Do NOT access formal test.
22. Output the unique final state.

## Commitment

CX commits to the execution order in directive §26. No new inference will run
before the MoNuSAC holdout protocol is frozen and the preregistration commit
is made. The success criterion is NOT "training a more complex model"; the
success criterion is producing a clear, evidence-backed, submittable Benchmark
or Hybrid paper — or honestly documenting why that is not possible.

## Starting Scientific State

```text
PHASES_BENCHMARK_CORE_FINDING_ESTABLISHED
```

Phase S established:

- Reference instability is real: corrected `V_ref_frac` 0.1376–0.3312.
- Reference instability is predictable: validation Spearman 0.478.
- Reference instability CANNOT be mitigated through reference weighting
  (Track B `NO_GAIN`) or set conditioning (Track C `NO_GAIN`).
- Mean consensus is the current strongest automatic method (+0.046 Dice over
  random single, single SAM call, but multiple matching passes).
- Final Phase S state: `PHASES_BENCHMARK_PAPER_READY`.

Phase T must NOT be written as `SUBMISSION_READY`, `PUBLICATION_COMPLETED`,
or `SOTA` in any new report until the Publication Readiness Gate (§20) passes.
