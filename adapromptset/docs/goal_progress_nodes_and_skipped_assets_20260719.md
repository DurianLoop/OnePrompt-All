# AdaPromptSet Goal Progress Nodes and Deferred Figure/Table Audit

Date: 2026-07-19

Update scope: this document now covers both the earlier eight-page oracle-analysis paper work and the current Phase 2 GlaS strict continuation. It is written as an accountability note, not as a success report. Items marked deferred below were not completed as paper-grade experimental evidence during the goal run.

This note answers why the goal run did not complete every important experiment table and figure in `checklist.md`, and records the process nodes reached during the run. The short version is that the work did not have legal evidence to advance into the full adaptive-policy table/figure set after Gate F failed. I then narrowed the paper work to a claim-safe eight-page oracle-analysis artifact, which is useful but not the same as completing the full Phase 15/16 paper package.

## Why Important Tables and Figures Were Not Completed

I did not complete the heavy experimental tables and figures for five concrete reasons.

1. Gate F failed after the one allowed MoNuSAC rescue bank.
   - Evidence: `reports/gates/gate_f_oracle_gap.json` records `status=FAIL_ORACLE_GAP_SMALL`, Dice oracle gap `0.0050113301616486194`, AJI oracle gap `0.0038202062471413`, and minimum Dice gap `0.010`.
   - Effect: MoNuSAC policy training and post-hoc MoNuSAC action-bank redesign were prohibited. Any table that needs a trained policy, formal policy comparison, or final adaptive-method row could not be generated honestly.

2. The current result is an oracle-analysis pivot, not a successful adaptive-policy result.
   - Evidence: `reports/decision_checkpoint/phase5_decision.json` and `reports/final_project_report.json` both record `PIVOT_ORACLE_ANALYSIS`.
   - Effect: I generated fixed-rule/oracle diagnostic assets, but not the full main comparison, formal budget table, frozen-transfer table, or policy ablation tables.

3. External datasets were locally useful but not formal-gate ready.
   - Evidence: `reports/new_domain_pilot_readiness.json` records GlaS as `BLOCKED_INSUFFICIENT_EPISODES` with `episode_count=1` versus `gate_min_episodes=10`; DDR as `READY_FOR_INFRASTRUCTURE_ONLY`; IDRID as `READ_ONLY_NON_TUNABLE`; and CPM-17 as `DEFERRED_MISSING_OR_INCOMPLETE`.
   - Effect: cross-dataset and frozen-transfer tables would have overstated the evidence. GlaS could only support a smoke/readiness result, DDR only infrastructure checks, IDRID only read-only boundary reporting, and CPM-17 only missing-asset reporting.

4. Statistical support was insufficient for promotion.
   - Evidence: `reports/final_project_report.json` records bootstrap/statistical status as insufficient/pending, and the current MoNuSAC Gate F decision is based on 10 validation episodes and 120 utility records.
   - Effect: figures that imply a stable policy advantage, confidence-supported main result, or reliable formal comparison were not valid to produce as final paper evidence.

5. Several requested figure/table categories require artifacts that do not exist yet.
   - Main comparison requires valid automatic-method rows beyond the current fixed/oracle diagnostic state.
   - Factorial and architecture ablations require trained or evaluated model/action variants that Gate F did not permit.
   - Reference sensitivity requires multi-reference robustness artifacts beyond the current partial diagnostics.
   - Efficiency tables and Dice-runtime Pareto require same-hardware quality and wall-clock comparisons under a frozen protocol; the current compute ledger is provenance, not an efficiency result.
   - Qualitative success/failure cases and behavior plots require a stable selection rule and a registered qualitative sampling rule, otherwise they risk cherry-picking.

6. The current Phase 2 GlaS continuation crossed Gate C only after one predeclared redesign.
   - Evidence: the frozen candidate generator failed component coverage in `reports/phase2_glas/candidates/gate_c_summary.json`; the predeclared redesign in `docs/glas_candidate_redesign_v1.md` passed in `reports/phase2_glas/candidates_redesign_v1/gate_c_summary.json`.
   - Effect: Phase 2 candidate-quality table/plots now exist, but Gate D later failed. No GlaS oracle-gap table, policy-reliability table, action-bank result table, or formal testA/testB figure can be claimed as completed.

## Checklist Figure/Table Status

| Checklist asset | Current status | Reason |
| --- | --- | --- |
| Related-work table | Completed | `paper/tables/table_related_work.tex` is generated from the boundary registry. |
| Main comparison | Deferred | No valid trained/adaptive policy or formal test path after Gate F failure. |
| Budget table | Deferred / partial diagnostic only | `table_oracle_pivot.tex` and `budget_pareto.csv` exist, but only for oracle-analysis pilot evidence. |
| Oracle heterogeneity | Deferred / partial diagnostic only | `table_action_summary.tex`, `table_representation_pivot.tex`, and oracle figures exist, but do not prove policy-ready heterogeneity. |
| Factorial ablation | Deferred | Would require prohibited or missing policy/action variants. |
| Architecture ablation | Deferred | Would require model/policy training beyond the pivot boundary. |
| Cross-dataset table | Deferred | GlaS has only one strict validation episode; DDR/IDRID/CPM-17 are not formal policy datasets in the current state. |
| Frozen transfer table | Deferred | No legal frozen transfer protocol/result set exists after the pivot. |
| Prompt statistics | Deferred | Some candidate/action counts exist, but no complete paper-grade prompt-statistics table was exported. |
| Reference sensitivity | Deferred | Reference robustness is not yet quantified enough for a formal table. |
| Efficiency table | Deferred | Current GPU-hour/runtime records are provenance only, not a quality-plus-cost claim. |
| TikZ method overview | Completed | `paper/figures/method_overview.tex` generated and audited. |
| TikZ action-bank illustration | Completed | `paper/figures/action_bank.tex` generated from `prompt_set_bank_v2`; deferred families are disclosed. |
| Dice-calls Pareto plot | Partial diagnostic only | `paper/figures/oracle_pareto.png` exists, but it is not the full policy/CI/formal-test Pareto requested by the checklist. |
| Dice-runtime Pareto plot | Deferred | No valid same-hardware quality-runtime formal comparison exists. |
| Qualitative success cases | Deferred | No registered qualitative sampling/export path was completed. |
| Qualitative failure cases | Deferred | No registered qualitative sampling/export path was completed. |
| Behavior plots | Deferred | No valid trained policy behavior artifact exists after Gate F. |
| Gate-routing table | Added during eight-page paper expansion | `paper/sections/06_limitations.tex` now includes a failure-to-experiment routing table; this is a useful audit table, not an original checklist completion item. |

## Process Nodes Reached During the Goal Run

| Node | Span | What was done | Main artifacts | Result and boundary |
| --- | --- | --- | --- | --- |
| 0. Scope and priority lock | 2026-07-17 | Read and operationalized the project docs, with the response/decision docs taking priority over long-range checklist ambitions. | `reports/phase0_scope_structure/*` | Research question narrowed to frozen DINOv2-L/SAM-B prompt-set selection; no broad novelty claim. |
| 1. Reproducibility metadata | 2026-07-17 | Added config hashing, source/input hashes, and compact run metadata. | `reports/run_metadata/phase0_reproducibility/run_metadata.json` | Reproducibility path established before reporting. |
| 2. Data and exposure audit | 2026-07-17 | Audited local MoNuSAC, MoNuSeg, and GlaS manifests plus reference exposure. | `reports/data_profile_audit.*`, `reports/reference_exposure_audit.*` | Data/profile checks passed; reference leakage check passed. |
| 3. DDR/other local dataset readiness | 2026-07-17 to 2026-07-18 | Audited DDR protocol role and later consolidated GlaS/DDR/IDRID/CPM-17 readiness. | `reports/ddr_protocol_audit/*`, `reports/new_domain_pilot_readiness.*` | DDR infrastructure-only; GlaS insufficient episodes; IDRID read-only; CPM-17 incomplete. |
| 4. Candidate and SAM smoke | 2026-07-17 | Ran candidate and SAM chain checks. | `reports/candidate_pilot/*`, `reports/sam_smoke/*`, `reports/gate_c_status.md`, `reports/gate_d_status.md` | Pipeline ran, but candidate coverage later became a limiting diagnostic. |
| 5. Action-bank MVP and rescue bank | 2026-07-17 | Built MVP action bank and the one allowed Rescue-F1 v2 subset. | `configs/actions/prompt_set_bank_v1.*`, `configs/actions/prompt_set_bank_v2.*`, `reports/action_bank/*` | v2 bank frozen with 13 actions, 12 valid; still not a complete final action-bank universe. |
| 6. Gate F oracle-gap pilots | 2026-07-17 to 2026-07-18 | Ran v1 and v2 oracle-gap pilots, later adding AJI recording. | `reports/oracle_gap_pilot/*`, `reports/oracle_gap_pilot_v2/*`, `reports/gate_f_status.md` | Gate F failed: Dice gap `0.005011` below `0.010`; AJI gap `0.003820` below auxiliary threshold. |
| 7. Phase 5 decision checkpoint | 2026-07-18 | Wrote the authoritative pivot decision. | `reports/decision_checkpoint/phase5_decision.*` | Unique state became `PIVOT_ORACLE_ANALYSIS`; allowed next work limited to fixed-rule/oracle analysis or predeclared new-domain pilot. |
| 8. Gate-F failure diagnostics | 2026-07-18 | Audited why Gate F failed. | `reports/gate_f_failure_diagnostics/*` | Coverage low, actions separated but insufficient, no high-call gain. |
| 9. Representation pivot analysis | 2026-07-18 | Joined v2 utility matrix and candidate diagnostics. | `reports/representation_analysis/*` | `PIVOT_REPRESENTATION_ANALYSIS`; claim stayed `oracle_analysis_only_insufficient_separation`. |
| 10. GlaS new-domain smoke | 2026-07-17 to 2026-07-18 | Registered and ran a small GlaS strict validation smoke. | `reports/new_domain_pilot_glas_*`, `reports/candidate_pilot_glas/*`, `reports/oracle_gap_pilot_glas/*` | GlaS wiring worked, but only one validation episode; not enough for Gate F or cross-dataset claim. |
| 11. Gate report export | 2026-07-18 | Exported Gates A-I with allowed/prohibited next actions. | `reports/gates/gate_*.json`, `reports/gates/gate_*.md` | A-D passed; E passed as rescue subset; F failed; G-I blocked/prohibited by pivot. |
| 12. Bootstrap/statistics boundary | 2026-07-18 | Added statistics/bootstrap reporting. | `reports/bootstrap/bootstrap_summary.*` | Insufficient independent clusters; no statistical promotion. |
| 13. Oracle-analysis paper tables | 2026-07-17 to 2026-07-18 | Generated partial paper tables from real artifacts. | `paper/tables/table_oracle_pivot.tex`, `table_action_summary.tex`, `table_representation_pivot.tex` | Useful for oracle-analysis paper only; not full Phase 15 main/budget/heterogeneity table set. |
| 14. Related-work boundary table | 2026-07-17 | Generated related-work boundary registry and table. | `reports/related_work_boundaries/*`, `paper/tables/table_related_work.tex` | Completed checklist item for related-work table. |
| 15. Oracle-analysis figures | 2026-07-18 | Generated diagnostic oracle figures from `budget_pareto.csv` and `action_summary.csv`. | `paper/figures/oracle_pareto.png`, `paper/figures/oracle_action_wins.png`, `reports/oracle_analysis_figures.*` | Partial diagnostic figures only; not full policy/CI/formal Pareto. |
| 16. Method and action-bank figures | 2026-07-18 | Generated TikZ method overview and action-bank illustration. | `paper/figures/method_overview.tex`, `paper/figures/action_bank.tex`, reports for both figures | Completed current oracle-analysis figure assets; action-bank figure discloses deferred families. |
| 17. Claim-boundary audits | 2026-07-18 | Added scientific-claim and paper-section audits. | `reports/scientific_claim_audit.*`, `reports/paper_section_audit.*` | Current reader-facing text audited as oracle-analysis-only. |
| 18. Final handoff report | 2026-07-18 | Exported consolidated project report. | `reports/final_project_report.*` | Status remained `PIVOT_ORACLE_ANALYSIS`; not final sign-off. |
| 19. PDF metadata audit | 2026-07-18 | Added PDF info/visible-title metadata check. | `reports/pdf_metadata_audit.*` | Metadata passed for the current PDF. |
| 20. Eight-page paper expansion | 2026-07-18 | Expanded the paper from a skeleton into an eight-page oracle-analysis draft, adding discussion, gate-routing, reproducibility, and limitations text. | `paper/main.pdf`, `paper/sections/*.tex` | PDF reached 8 pages, but it remains an oracle-analysis paper, not a full positive-result submission. |
| 21. Clean latexmk audit | 2026-07-18 | Added clean-build audit script and tests; attempted real clean `latexmk`. | `src/adapromptset/scripts/audit_latexmk_build.py`, `scripts/audit_latexmk_build.py`, `reports/latexmk_audit.*` | Script tests passed, but the real audit was interrupted/failed with MiKTeX log permission issues; checklist item remains open. |
| 22. Phase 2 objective reset | 2026-07-19 | Switched from MoNuSAC oracle-pivot paper work to GlaS strict diagnostic and policy-gate re-evaluation under the pasted Phase 2 instructions. | Active branch `phase2-glas-strict`; `docs/phase2_goal_read_receipt.md` | Phase 2 scope became GlaS train/strict-val diagnostic only; MoNuSAC remained frozen. |
| 23. MoNuSAC freeze guard | 2026-07-19 | Recorded that the prior MoNuSAC result is frozen at `PIVOT_ORACLE_ANALYSIS` and cannot be reopened for policy/action-bank changes. | `docs/monusac_freeze_notice.md`, `reports/phase2_gate_status.md` | Prevented further MoNuSAC policy training, post-hoc action-bank edits, or Gate-F threshold lowering. |
| 24. GlaS exposure and protocol audit | 2026-07-19 | Audited the local GlaS official train/testA/testB layout, grade metadata, and prior exposure boundary. | `docs/glas_protocol_v2.md`, `docs/glas_exposure_audit.md` | testA/testB were locked and marked not fully blind because prior manifest/profile exposure existed, but no audited method-selection use was found. |
| 25. GlaS 55/30 strict split construction | 2026-07-19 | Built Phase 2 manifests from official train only: 55 utility-training queries and 30 strict-validation queries, with locked testA/testB case listing. | `manifests/phase2/glas/*`, `reports/phase2_glas/split/split_report.json` | Manifest gate passed, but this is only a data-split artifact; it does not complete any policy or formal-result table. |
| 26. Path-style correction for GlaS manifests | 2026-07-19 | Corrected generated GlaS paths from workspace-relative to GlaS-root-relative and made the real matcher smoke resolver compatible with both styles. | Modified `build_glas_phase2_splits.py`, `run_real_matcher_smoke.py`, related tests | Reduced rerun fragility before Gate C; manifest hashes changed and docs need hash refresh before committing. |
| 27. Phase 2 Gate C strict audit implementation | 2026-07-19 | Added a candidate-quality audit layer intended to aggregate 90 reference episodes back to 30 validation queries and check precision enrichment, component coverage, negative blank fraction, and pool-size curves. | `src/adapromptset/scripts/audit_glas_phase2_gate_c.py`, `scripts/audit_glas_phase2_gate_c.py`, `tests/test_audit_glas_phase2_gate_c.py` | Implementation exists but has not yet been validated by focused tests or run on real GlaS candidate outputs in this current node. |
| 28. Current stop point after user challenge | 2026-07-19 | Re-read `checklist.md`, `reports/phase2_gate_status.md`, current paper table/figure files, and git status to answer why the requested assets were not all completed. | This document | Confirmed status at that moment: Phase 2 was at `PHASE2_GATE_C_READY`; the important experimental figure/table set was not complete. |
| 29. Phase 2 Gate C real candidate pilot | 2026-07-19 | Ran DINOv2-L candidate diagnostics on 30 GlaS strict-validation queries with 3 references each. | `reports/phase2_glas/candidates/candidate_pilot_metrics.csv`, `candidate_pilot_candidates.csv` | Produced 90-reference-episode evidence with GPU estimate `0.017943`; the old summary failed, so the strict query-level audit was used as authoritative. |
| 30. Phase 2 strict Gate C audit | 2026-07-19 | Aggregated candidate metrics to query level and generated Gate C tables/figures. | `reports/phase2_glas/candidates/gate_c_summary.json`, `query_level_candidate_metrics.csv`, `candidate_pool_curve.csv`, `paper_phase2/figures/candidate_diagnostics/*.png` | Gate C failed: precision enrichment passed, coordinate validation passed, negative blank fraction passed, but component coverage was `0.418820 < 0.600000`; next state is `REDESIGN_GLAS_CANDIDATES_ONCE_ALLOWED`. |
| 31. Phase 2 candidate redesign pre-registration | 2026-07-19 | Pre-registered the only allowed GlaS candidate redesign after Gate C failure. | `docs/glas_candidate_redesign_v1.md` | Allowed change limited to `local_peaks`, diagnostic pool sizes `5,10,20`, and spacing `2.0`; no testA/testB, action-bank, threshold, or MoNuSAC changes allowed. |
| 32. Phase 2 redesigned Gate C rerun | 2026-07-19 | Ran and audited the predeclared `local_peaks` candidate redesign on the same 30 query / 90 reference episode validation manifest. | `reports/phase2_glas/candidates_redesign_v1/gate_c_summary.json`, `paper_phase2/figures/candidate_diagnostics_redesign_v1/*.png` | Gate C passed after redesign: precision delta `0.433889`, precision ratio `1.806818`, component coverage `0.630221`, negative blank fraction `0.135556`, coordinate validation `PASS`; current state is `PHASE2_GATE_D_READY`. |
| 33. Phase 2 Gate D SAM-B smoke | 2026-07-19 | Ran GlaS Gate D privileged/control SAM-B smoke on 6 selected strict-validation cases. | `reports/phase2_glas/gate_d_segmenter_smoke/gate_d_summary.json`, `gate_d_case_metrics.csv` | Gate D failed: mean privileged best Dice `0.733908` vs control `0.638274`, but at least one case did not clearly exceed controls. |
| 34. Phase 2 Gate D SAM-L alternative | 2026-07-19 | Pre-registered and ran the only allowed alternative segmenter, SAM-L. | `docs/glas_gate_d_segmenter_alternative_v1.md`, `reports/phase2_glas/gate_d_segmenter_smoke_saml/gate_d_summary.json` | SAM-L also failed the per-case privileged-control margin check despite better mean margin; current state is `STOP_GLAS_SEGMENTER_FAILURE`. |

## Current Paper Assets Actually Present

Current generated paper assets include:

- `paper/main.pdf`: 8 pages as of the last successful `pdflatex` build.
- `paper/figures/method_overview.tex`
- `paper/figures/action_bank.tex`
- `paper/figures/oracle_pareto.png`
- `paper/figures/oracle_action_wins.png`
- `paper/tables/table_related_work.tex`
- `paper/tables/table_oracle_pivot.tex`
- `paper/tables/table_action_summary.tex`
- `paper/tables/table_representation_pivot.tex`
- Gate-routing table embedded in `paper/sections/06_limitations.tex`

Current Phase 2 Gate C assets include:

- `reports/phase2_glas/candidates/gate_c_summary.json`
- `reports/phase2_glas/candidates/gate_c_summary.md`
- `reports/phase2_glas/candidates/query_level_candidate_metrics.csv`
- `reports/phase2_glas/candidates/candidate_pool_curve.csv`
- `paper_phase2/figures/candidate_diagnostics/precision_vs_pool.png`
- `paper_phase2/figures/candidate_diagnostics/coverage_vs_pool.png`
- `paper_phase2/figures/candidate_diagnostics/negative_blank_vs_pool.png`
- `docs/glas_candidate_redesign_v1.md`
- `reports/phase2_glas/candidates_redesign_v1/gate_c_summary.json`
- `reports/phase2_glas/candidates_redesign_v1/gate_c_summary.md`
- `reports/phase2_glas/candidates_redesign_v1/query_level_candidate_metrics.csv`
- `reports/phase2_glas/candidates_redesign_v1/candidate_pool_curve.csv`
- `paper_phase2/figures/candidate_diagnostics_redesign_v1/precision_vs_pool.png`
- `paper_phase2/figures/candidate_diagnostics_redesign_v1/coverage_vs_pool.png`
- `paper_phase2/figures/candidate_diagnostics_redesign_v1/negative_blank_vs_pool.png`
- `reports/phase2_glas/gate_d_segmenter_smoke/gate_d_summary.json`
- `reports/phase2_glas/gate_d_segmenter_smoke/gate_d_case_metrics.csv`
- `docs/glas_gate_d_segmenter_alternative_v1.md`
- `reports/phase2_glas/gate_d_segmenter_smoke_saml/gate_d_summary.json`
- `reports/phase2_glas/gate_d_segmenter_smoke_saml/gate_d_case_metrics.csv`

## What I Should Have Made Clearer During the Goal

I should have explicitly separated these two tracks earlier:

1. Full checklist completion track: requires more experiments, legal gates, external dataset readiness, and formal comparison artifacts.
2. Eight-page claim-safe paper track: can be completed from current oracle-analysis evidence, but only as a bounded pivot paper.

When the work reached Gate F failure, the correct phrase was not "finish every table and figure"; it was "finish the tables and figures that are legal under the oracle-analysis pivot, and list the rest as deferred with reasons." The eight-page PDF work followed the second track.

## Next Valid Work If You Reopen the Goal

The next valid work is not to fill blank tables by hand. It should be one of these registered continuations:

1. Run a predeclared GlaS train/validation diagnostic with enough query-level units, while preserving official testA/testB.
2. Run DDR stage-one infrastructure smoke only, then predeclare any four-intent DDR protocol before formal claims.
3. Write a CPM-17 missing-asset report or repair the manifest with explicit approval and license/source checks.
4. Add prompt-statistics and reference-sensitivity exporters from existing artifacts if they can be clearly labeled diagnostic-only.
5. Repair/rerun the clean `latexmk` audit and update checklist Phase 16 only after it passes.

## Immediate Corrections I Should Make Before Continuing

1. Treat Phase 2 Gate C as passed only after the predeclared redesign; do not describe the frozen candidate generator as passing.
2. Treat Phase 2 Gate D as failed after SAM-B and the one predeclared SAM-L alternative.
3. Do not proceed to Gate E/F/H/I in this GlaS branch.
4. Do not use GlaS testA/testB; they remain locked and unused.
5. Keep `reports/phase2_gate_status.md`, `reports/phase2_compute_ledger.csv`, and `docs/phase2_goal_read_receipt.md` synchronized with every rerun hash.
6. The next deliverable should be `reports/phase2_final_report.{md,json}` with final state `STOP_GLAS_SEGMENTER_FAILURE`.
