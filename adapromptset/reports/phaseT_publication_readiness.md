# Phase T Publication Readiness Checklist

Date: 2026-07-23
Branch: `phaseT-publication-completion`
Directive: `D:\SAM\adapromptset\reports\AdaPromptSet_PhaseT_Publication_Completion.md`
Related: directive §20 (Publication Readiness Gate)

## Publication Readiness Gate

Only when ALL 12 conditions below are satisfied may the state
`PHASET_PAPER_CANDIDATE_READY` be written. Otherwise the state is one of:

```
PHASET_BENCHMARK_NEEDS_MORE_EVIDENCE
PHASET_METHOD_NOT_READY_BENCHMARK_CONTINUES
PHASET_RESEARCH_DIRECTION_NOT_SUPPORTED
```

All conditions are currently **PENDING**.

| # | Condition | Status | Evidence Required |
|---|---|---|---|
| 1 | MoNuSAC holdout confirmed | PENDING | WP1 Gate (§5.6) all 10 conditions pass; `reports/phaseT_wp1_holdout_summary.json` |
| 2 | ≥ 2 extra datasets directional support | PENDING | WP2 cross-dataset Gate (§6.4); `reports/phaseT_cross_dataset_audit/` |
| 3 | ≥ 1 external method reproduced under unified protocol (OR documented justification + strong internal baselines) | PENDING | `reports/phaseT_external_baselines/reproduced_results.csv`; directive §7.3 |
| 4 | Benchmark metrics implementation tested | PENDING | Unit tests for V_ref, CVaR, CFR, J_pred; Phase S Track A audit already PASS, Phase T must reverify |
| 5 | Mean consensus gain cross-dataset directionally stable | PENDING | WP2 cross-dataset consensus-vs-random-single Δ Dice ≥ 0 on each directional-support dataset |
| 6 | Failure taxonomy complete | PENDING | `reports/phaseT_failure_taxonomy.{md,json}` covering small-sparse-target, diffuse-map, prompt-interface, reference-catastrophic modes |
| 7 | M1 OR M2 has validation positive signal (Benchmark-only route: not required, but submission positioning is lowered) | PENDING | WP3 M1 Gate (§9.6) OR M2 Gate (§10.5) pass on validation |
| 8 | All main tables and figures complete | PENDING | 18 tables + 18 figures per `docs/phaseT_paper_outline.md` |
| 9 | Full paper draft complete | PENDING | `paper/phaseT/full_draft.md` and section files per directive §19 |
| 10 | Mock review with no fatal unresolved issues | PENDING | `reports/phaseT_mock_review.md` and `reports/phaseT_rebuttal_plan.md` |
| 11 | Formal test still locked | PENDING (LOCKED) | `docs/phaseT_test_lock_notice.md`; lock audit PASS |
| 12 | Claim registry clean (no prohibited claims) | PENDING | `docs/phaseT_claim_registry.md`; claim-boundary audit PASS |

## Submission Route Decision

The submission route is decided at WP4 based on which Gates passed:

- **Benchmark-only route**: Conditions 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12 pass.
  Condition 7 is NOT required, but the submission positioning is lowered.
- **Hybrid route**: All 12 conditions pass (including condition 7).

The route decision is recorded in `reports/phaseT_final_publication_report.md`.

## Prohibited State Transitions

The following state transitions are PROHIBITED regardless of Gate outcomes:

- Writing `SUBMISSION_READY` or `PUBLICATION_COMPLETED`.
- Writing `SOTA` in any report.
- Writing `PHASET_PAPER_CANDIDATE_READY` before all 12 conditions pass.
- Lifting the formal test lock based on a `PHASET_PAPER_CANDIDATE_READY`
  state.

## Audit

Before the final state transition, the claim-boundary audit scans all Phase T
artifacts for prohibited tokens. Any hit blocks the final state. The audit
covers `docs/phaseT_*.md`, `reports/phaseT_*.{md,json,csv}`,
`paper/phaseT/*.md`, and terminal replies.
