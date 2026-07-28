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

Final audit completed 2026-07-23.

| # | Condition | Status | Evidence |
|---|---|---|---|
| 1 | MoNuSAC holdout confirmed | **PASS** | `reports/phaseT_wp1_holdout/gateWP1_summary.md` — 10/10 conditions pass; V_ref_frac 0.1264–0.2627; consensus gain +0.0266; bootstrap CI lower +0.0140; Spearman 0.7241 |
| 2 | ≥ 2 extra datasets directional support | **FAIL** | Only 1 of 2 datasets supports. MoNuSeg: `PHASET_DATASET_STRONG_SUPPORT` (5/5 directional + 4/4 strong). IDRID: `PHASET_DATASET_FAIL` (4/5 directional, 2/4 strong; d3 consensus Dice gain = −0.0118). Requires ≥2 directional support; have 1. |
| 3 | ≥ 1 external method reproduced under unified protocol (OR documented justification + strong internal baselines) | **PASS (Option C, lowered positioning)** | `reports/phaseT_external_baselines/unsupported_comparators.md` — 0 external methods reproduced (2 Tier-2 methods GF-SAM/MAUP require user authorization per §23.1; 5 Tier-3/4 methods cannot be reproduced). **Sufficient justification**: §23.1 blocks external code download without user authorization; 5/7 priority methods are Tier 3 (no code) or Tier 4 (not applicable). **Alternative strong baselines**: B1–B7 internal method family (7 methods: random single, mean/median/trimmed-mean/disagreement-penalized/quantile consensus, learned weighting). **Public protocol**: `docs/phaseT_external_baseline_protocol.md`, `docs/phaseT_reference_protocol.md`. **Code interface**: `reports/phaseT_external_baselines/implementation_notes.md`, `protocol_alignment.md`. **Clear limitations**: `reports/phaseT_external_baselines/unsupported_comparators.md`. Per §7.3 Option C, submission positioning is lowered but cond 3 is satisfied. |
| 4 | Benchmark metrics implementation tested | **PASS** | Phase S Track A metric audit confirmed corrected V_ref_frac valid (0.1376–0.3312). Phase T MoNuSAC holdout V_ref_frac 0.1264–0.2627 within valid range. CVaR, CFR, J_pred, MAD all implemented and reported in `table_03_correct_variance_decomposition.md` and `table_07_tail_risk_metrics.md`. |
| 5 | Mean consensus gain cross-dataset directionally stable | **FAIL** | Not stable. MoNuSeg consensus gain +0.0149 (PASS, CI lower +0.0069). IDRID consensus gain −0.0118 (FAIL, CI lower −0.0318). Direction reverses on IDRID — consensus hurts Dice there. |
| 6 | Failure taxonomy complete | **PASS** | `reports/phaseT_tables/table_16_failure_taxonomy.md` covers small-sparse-target, diffuse-map, prompt-interface, reference-catastrophic modes. |
| 7 | M1 OR M2 has validation positive signal | **PASS** | M2 dual-model val 5/5 PASS (AUROC 0.9443, AUPRC 0.4338 vs prevalence 0.1171, ECE 0.0425, CFR reduction 39.2%, Dice +0.0214). M2 holdout 6/6 PASS (AUROC 0.9283, AUPRC 0.5277, ECE 0.0270, CFR reduction 79.2%, Dice +0.0090). M1 NO_GAIN (CV selects λ_c=λ_r=0, avg_k 5.63, 6.1% reduction < 25% threshold). |
| 8 | All main tables and figures complete | **PASS** | 18 tables in `reports/phaseT_tables/` (table_01 through table_18) and 18 figures in `reports/phaseT_figures/` (fig_01 through fig_18). All 22 §15-16 deliverables present. |
| 9 | Full paper draft complete | **PASS** | `paper/phaseT/full_draft.md` (~7000 words) plus 15 supporting section files (title_candidates, abstract, contributions, introduction, related_work, method, benchmark_protocol, experiments, results, limitations, conclusion, appendix, figure_plan, table_plan, reproducibility_checklist). |
| 10 | Mock review with no fatal unresolved issues | **PASS** | `reports/phaseT_mock_review.md` (5 simulated reviewers) and `reports/phaseT_rebuttal_plan.md` (4-field rebuttal structure). No fatal unresolved issues — all criticisms have documented rebuttals or accepted limitations. |
| 11 | Formal test still locked | **PASS** | Formal test never accessed. `docs/phaseT_test_lock_notice.md` in force. No `reports/phaseU_*` artifacts created (§21 not triggered). |
| 12 | Claim registry clean (no prohibited claims) | **PASS** | `reports/phaseT_tables/table_18_claim_registry.md` and `docs/phaseT_claim_registry.md` audited. No occurrences of `SOTA`, `SUBMISSION_READY`, `PUBLICATION_COMPLETED`, `universal across datasets`, `instability solved`. |

## Summary: 10 of 12 conditions PASS, 2 FAIL

PASS: 1, 3 (Option C, lowered positioning), 4, 6, 7, 8, 9, 10, 11, 12 (10 conditions)
FAIL: 2 (only 1/2 supporting datasets), 5 (consensus gain not stable across all datasets)

## Supplementary evidence (not a gate condition)

M2 cross-dataset transfer evaluation
(`reports/phaseT_wp3_m2_transfer/m2_transfer_summary.json`) applied the
frozen dual-model M2 (trained on MoNuSAC train, selected on MoNuSAC val)
to MoNuSeg and IDRID episode data without re-tuning:

| Dataset | AUROC | AUPRC_cat | ECE | CFR red @80% | Dice chg @80% | Transfer |
|---|---|---|---|---|---|---|
| MoNuSAC holdout (ref) | 0.9283 | 0.5277 | 0.0270 | 0.7917 | +0.0090 | (reference) |
| MoNuSeg | 0.8740 | 0.0000* | 0.0601 | 0.0000* | +0.0131 | Ranking transfers |
| IDRID | 0.9938 | 0.9973 | 0.4767 | 0.0599 | +0.0128 | Ranking transfers, calibration does NOT |

*MoNuSeg has 0 catastrophic failures (prevalence_cat=0), so AUPRC and CFR
reduction are vacuously 0 — there is nothing to predict or reduce.

**Key finding**: M2's ranking ability (AUROC) transfers well to all datasets
(0.87–0.99), but its calibration (ECE) and threshold (CFR reduction) do NOT
transfer to IDRID (ECE 0.48 >> 0.10; CFR reduction 0.06 < 0.30) due to
different feature distributions and prevalence (IDRID prevalence_cat=0.807
vs MoNuSAC train prevalence_cat=0.117). This suggests M2 could be useful
cross-dataset with target-domain recalibration, which is a future work
direction documented in the paper.

## Submission Route Decision

- **Hybrid route** (all 12 pass): NOT AVAILABLE — conditions 2 and 5 fail.
- **Benchmark-only route** (conditions 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12 pass; 7 not required): NOT AVAILABLE — conditions 2 and 5 still fail.
- **Lowered-positioning benchmark route** (Option C of §7.3 + partial cross-dataset evidence): AVAILABLE as fallback. Condition 3 passes under Option C. Conditions 2 and 5 still fail because IDRID reverses consensus direction, but the core phenomenon (instability) is confirmed on MoNuSAC holdout + MoNuSeg, and the mitigation (M2) is holdout-confirmed on MoNuSAC with ranking-transfer evidence on MoNuSeg/IDRID.

## Final State

```
PHASET_BENCHMARK_NEEDS_MORE_EVIDENCE
```

Rationale:
- WP1 MoNuSAC holdout is fully confirmed (core phenomenon solid, 10/10 WP1 gate conditions).
- WP2 cross-dataset provides only 1 supporting dataset (MoNuSeg STRONG_SUPPORT); IDRID FAILS (consensus hurts Dice, direction reverses). Conditions 2 and 5 fail.
- WP2b condition 3 passes under §7.3 Option C (sufficient justification + alternative strong baselines + public protocol + clear limitations; lowered positioning).
- WP3 M2 selective prediction passes val (5/5) and holdout (6/6). M2 cross-dataset transfer shows ranking transfers (AUROC 0.87–0.99) but calibration does NOT transfer to IDRID (ECE 0.48).
- The direction IS supported (instability is real and predictable on MoNuSAC + MoNuSeg); the bottleneck is breadth of cross-dataset evidence, not the core finding.
- Remaining 2 FAIL conditions (2, 5) require a new supporting dataset, which requires user authorization per §23.1/§23.9.

## Prohibited State Transitions

The following state transitions are PROHIBITED regardless of Gate outcomes:

- Writing `SUBMISSION_READY` or `PUBLICATION_COMPLETED`.
- Writing `SOTA` in any report.
- Writing `PHASET_PAPER_CANDIDATE_READY` before all 12 conditions pass.
- Lifting the formal test lock based on a `PHASET_PAPER_CANDIDATE_READY`
  state.

## Audit

Claim-boundary audit completed 2026-07-23. No prohibited tokens found in
`docs/phaseT_*.md`, `reports/phaseT_*.{md,json,csv}`, `paper/phaseT/*.md`,
or terminal replies.

## Next Steps (require user authorization)

1. **WP2b — Reproduce GF-SAM and MAUP** under unified reference-resampling
   protocol (requires external code download per §23.1; satisfies §7.3
   Option A).
2. **Add 1 more supporting dataset** — ideally a non-medical dataset
   (Pascal-5i or COCO-20i) for generalizability, or GlaS if local data
   permits. This addresses conditions 2 and 5.
3. After (1) and (2), re-evaluate the Gate. If conditions 2, 3, 5 pass,
   state may transition to `PHASET_PAPER_CANDIDATE_READY` and §21
   (formal test freeze) may be initiated with user approval.
