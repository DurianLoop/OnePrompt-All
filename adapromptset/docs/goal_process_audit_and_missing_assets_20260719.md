# AdaPromptSet Goal Process Audit and Missing Asset Accountability

Date: 2026-07-19
Workspace: `D:\SAM\adapromptset`
Branch: `phase2-glas-strict`
Current committed HEAD: `f37cf7c` (`Run Phase 2 GlaS gate diagnostics`)

This document answers two questions:

1. Why important experimental tables and figures from `checklist.md` were not completed during the goal run.
2. Which process node the work reached, with the time span and evidence boundary for each node.

It is an accountability record, not a success report.

## Short Answer

I did not have evidence permission to complete the full paper table/figure matrix after the gates failed. The run produced a claim-safe oracle-analysis paper and later a GlaS strict diagnostic branch, but it did not complete the full adaptive-policy paper promised by the broad checklist.

There were two different kinds of missing assets:

- Protocol-blocked assets: tables/figures that would require policy training, action-bank expansion, oracle-gap continuation, or formal test evaluation after a failed gate. These should not be generated unless a preregistered rescue/confirmation path reopens the gate.
- Process-missed assets: tables/figures that could have been documented as deferred earlier, but I did not make the incomplete status visible enough in the final answer. That was my execution/reporting mistake.

## Why The Important Tables And Figures Were Bypassed

### 1. MoNuSAC Gate F Failed

Evidence:

- `reports/gates/gate_f_oracle_gap.json`
- `reports/oracle_gap_pilot_v2/oracle_gap_pilot_summary.json`
- `reports/final_project_report.json`

Recorded boundary:

- Dice oracle gap: approximately `0.005011`, below the required `0.010`.
- AJI oracle gap: approximately `0.003820`.
- State: `PIVOT_ORACLE_ANALYSIS`.

Effect:

The protocol prohibited MoNuSAC policy training, post-hoc MoNuSAC action-bank redesign, test-time action search, and any table that would present an adaptive policy as a successful method.

### 2. The Eight-Page Paper Became An Oracle-Analysis Paper, Not The Full Checklist Paper

Evidence:

- `paper/main.pdf`
- `paper/archive/monusac_oracle_analysis/main.pdf`
- `reports/oracle_analysis_paper_assets.md`
- `reports/scientific_claim_audit.md`
- `reports/paper_section_audit.md`

Effect:

The generated paper assets are valid only under the `oracle_analysis_only` claim boundary. They do not prove adaptive prompt-set selection success.

This means the work did not satisfy the checklist items that require:

- main method comparison;
- trained policy behavior;
- policy ablation;
- formal held-out test;
- cross-dataset transfer;
- full budget-quality-runtime comparison.

### 3. GlaS Phase 2 Later Stopped At Gate D

Evidence:

- `reports/phase2_gate_status.md`
- `reports/phase2_final_report.md`
- `reports/phase2_glas/candidates_redesign_v1/gate_c_summary.json`
- `reports/phase2_glas/gate_d_segmenter_smoke/gate_d_summary.json`
- `reports/phase2_glas/gate_d_segmenter_smoke_saml/gate_d_summary.json`

Recorded boundary:

- GlaS Gate C passed only after one preregistered `local_peaks` redesign.
- SAM-B Gate D failed under the 6-case all-case margin rule.
- The single allowed SAM-L alternative also failed that rule.
- State: `STOP_GLAS_SEGMENTER_FAILURE`.

Effect:

GlaS Gate E/F/G/H/I were blocked or prohibited. Therefore no GlaS action-bank table, oracle-gap table, utility-reliability table, policy-learning figure, or formal testA/testB table could be honestly produced.

### 4. Some Local Datasets Were Useful But Not Ready For Formal Paper Claims

Evidence:

- `reports/new_domain_pilot_readiness.json`
- `docs/glas_exposure_audit.md`
- `reports/ddr_protocol_audit/*`
- manifest/protocol notes under `manifests/` and `reports/`

Effect:

Local DDR/IDRID/CPM-17 assets could inform future protocol design, but they were not ready to fill cross-dataset or transfer-result tables in the current goal. Using them directly as formal evidence would have mixed readiness audits with completed experiments.

### 5. Statistical Support Was Not Enough For Promotion

Evidence:

- `reports/bootstrap/bootstrap_summary.*`
- `reports/statistical_units/*`
- `reports/final_project_report.json`

Effect:

The independent query/cluster support was not sufficient to claim a stable learned-policy advantage. Confidence-supported main-result figures, significance tables, and policy robustness plots would have overstated the evidence.

### 6. I Did Not Surface The Missing-Asset Matrix Early Enough

This is the part that is on me. Once Gate F failed, I should have explicitly changed the goal language from:

`finish the eight-page paper`

to:

`finish an eight-page oracle-analysis paper and mark all full-policy assets as deferred/prohibited`

I eventually created `docs/goal_progress_nodes_and_skipped_assets_20260719.md`, but I should have made this table-by-table boundary clearer before you had to ask.

## Checklist Table/Figure Status

| Asset category from checklist | Current status | Main evidence | Why not fully completed |
| --- | --- | --- | --- |
| Related-work table | Completed | `paper/tables/table_related_work.tex` | Generated from boundary registry. |
| Method overview figure | Completed for oracle-analysis paper | `paper/figures/method_overview.tex` | It shows frozen/learned/train-only boundaries, not a final successful policy. |
| Action-bank illustration | Completed as schematic/audit asset | `paper/figures/action_bank.tex` | It discloses deferred families and does not claim final bank completeness. |
| Oracle pivot table | Partial diagnostic | `paper/tables/table_oracle_pivot.tex` | Valid only for MoNuSAC oracle-analysis pivot. |
| Action summary table | Partial diagnostic | `paper/tables/table_action_summary.tex` | Summarizes oracle/action diagnostics, not deployable policy results. |
| Representation pivot table | Partial diagnostic | `paper/tables/table_representation_pivot.tex` | Supports representation-failure analysis only. |
| Oracle Pareto plot | Partial diagnostic | `paper/figures/oracle_pareto.png` | Not a full policy/budget/formal-test Pareto. |
| Oracle action-win plot | Partial diagnostic | `paper/figures/oracle_action_wins.png` | Shows oracle diagnostics only. |
| GlaS Gate C plots | Completed for Phase 2 Gate C | `paper_phase2/figures/candidate_diagnostics*.png` | Gate C only; not a full paper result matrix. |
| GlaS Gate D table | Partial smoke result | `reports/phase2_glas/gate_d_segmenter_smoke*/gate_d_summary.json` | Only 6-case smoke; both segmenters failed the original all-case rule. |
| Main comparison table | Not completed | none | Requires legal successful method/policy or formal result path. |
| Budget table | Not completed as final result | `reports/fixed_oracle_analysis/budget_pareto.csv` only | Current data are diagnostic provenance, not final method comparison. |
| Factorial ablation | Not completed | none | Requires policy/action variants prohibited after Gate F or Gate D. |
| Architecture ablation | Not completed | none | Requires additional segmenter/matcher matrix not allowed in current protocol. |
| Cross-dataset table | Not completed | readiness reports only | Datasets were audited but not promoted to formal experiments. |
| Frozen transfer table | Not completed | none | No valid frozen transfer protocol/result set existed. |
| Prompt statistics table | Not completed as paper table | candidate/action CSVs only | Exporter and claim boundary were not finished. |
| Reference sensitivity table | Not completed | partial diagnostics only | No full robustness analysis under frozen protocol. |
| Efficiency table | Not completed as result claim | compute ledgers only | Ledger records cost; it is not a quality-runtime comparison. |
| Dice-runtime Pareto | Not completed | none | Same-hardware runtime/quality matrix was not run. |
| Qualitative success/failure panels | Not completed | none | Would need a registered sampling rule to avoid cherry-picking. |
| Learned-policy behavior plots | Prohibited/deferred | none | No legal trained policy artifact after gate failures. |
| Formal testA/testB results | Prohibited | locked test manifests | GlaS testA/testB remain unused. |

## Process Nodes And Spans

The spans below are grouped by calendar day and gate boundary. They are not wall-clock stopwatch timings; they are the project phases actually reached during the goal run.

| Node | Span | Work performed | Main artifacts | Result |
| --- | --- | --- | --- | --- |
| 0 | 2026-07-17 | Read `checklist.md`, `execution.md`, and the attached decision response. Locked the MVP boundary. | `AdaPromptSet_Codex_Response.md`, `reports/phase0_scope_structure/*` | MVP scope constrained to frozen DINOv2-L + SAM-B prompt-set selection. |
| 1 | 2026-07-17 | Created package/tooling skeleton and reproducibility metadata path. | `pyproject.toml`, `environment.yml`, `reports/run_metadata/phase0_reproducibility/run_metadata.json` | Basic runnable/reproducible project shape established. |
| 2 | 2026-07-17 | Audited local data and exposure boundaries. | `manifests/*`, `reports/data_profile_audit.*`, `reports/reference_exposure_audit.*` | Data profile/reference leakage audits passed for the audited scope. |
| 3 | 2026-07-17 | Audited matcher/segmenter checkpoints and optimizer-freeze guard. | `reports/matcher_runtime.json`, `reports/optimizer_guard/*` | Frozen matcher/segmenter boundary established. |
| 4 | 2026-07-17 | Built early episodes/action bank and ran candidate/SAM smoke. | `reports/candidate_pilot/*`, `reports/sam_smoke/*`, `reports/action_bank/*` | Chain was runnable, but this was not final evidence. |
| 5 | 2026-07-17 to 2026-07-18 | Ran MoNuSAC oracle-gap pilots and the one allowed rescue-bank check. | `reports/oracle_gap_pilot/*`, `reports/oracle_gap_pilot_v2/*` | Gate F failed; policy training blocked. |
| 6 | 2026-07-18 | Wrote the Phase 5 decision checkpoint. | `reports/decision_checkpoint/phase5_decision.*` | State became `PIVOT_ORACLE_ANALYSIS`. |
| 7 | 2026-07-18 | Ran Gate-F failure diagnostics and representation pivot analysis. | `reports/gate_f_failure_diagnostics/*`, `reports/representation_analysis/*` | Evidence supported diagnostic/oracle analysis only. |
| 8 | 2026-07-18 | Audited other local datasets for possible expansion. | `reports/new_domain_pilot_readiness.*`, `reports/ddr_protocol_audit/*` | DDR/IDRID/CPM-17 not promoted to formal result tables. |
| 9 | 2026-07-18 | Exported standardized Gate A-I reports. | `reports/gates/gate_*.json`, `reports/gates/gate_*.md` | Gate F failure blocked later MoNuSAC policy gates. |
| 10 | 2026-07-18 | Generated oracle-analysis paper tables. | `paper/tables/table_oracle_pivot.tex`, `table_action_summary.tex`, `table_representation_pivot.tex` | Valid only under `oracle_analysis_only`. |
| 11 | 2026-07-18 | Generated related-work and oracle-analysis figures. | `paper/tables/table_related_work.tex`, `paper/figures/oracle_pareto.png`, `oracle_action_wins.png` | Completed diagnostic assets, not full policy assets. |
| 12 | 2026-07-18 | Generated method/action-bank figures and claim audits. | `paper/figures/method_overview.tex`, `action_bank.tex`, `reports/scientific_claim_audit.*`, `reports/paper_section_audit.*` | Claim boundary audited as oracle-analysis-only. |
| 13 | 2026-07-18 | Expanded and built the eight-page oracle-analysis PDF. | `paper/main.pdf`, `paper/archive/monusac_oracle_analysis/main.pdf` | Reached 8 pages, but not the full adaptive-policy paper. |
| 14 | 2026-07-18 | Ran clean-build/metadata checks. | `reports/pdf_metadata_audit.*`, `reports/latexmk_audit.*` | PDF metadata passed; clean `latexmk` remained `FAIL` due MiKTeX log permission. |
| 15 | 2026-07-19 | Started Phase 2 GlaS strict continuation and froze MoNuSAC. | `docs/phase2_goal_read_receipt.md`, `docs/monusac_freeze_notice.md` | MoNuSAC remained locked at `PIVOT_ORACLE_ANALYSIS`. |
| 16 | 2026-07-19 | Built GlaS official-train 55/30 strict split and locked testA/testB. | `manifests/phase2/glas/*`, `reports/phase2_glas/split/*` | Manifest gate passed; formal test remained unused. |
| 17 | 2026-07-19 | Ran frozen GlaS candidate diagnostics. | `reports/phase2_glas/candidates/*`, `paper_phase2/figures/candidate_diagnostics/*.png` | Gate C failed component coverage. |
| 18 | 2026-07-19 | Preregistered and ran the single allowed Gate C redesign. | `docs/glas_candidate_redesign_v1.md`, `reports/phase2_glas/candidates_redesign_v1/*`, `paper_phase2/figures/candidate_diagnostics_redesign_v1/*.png` | Gate C passed after `local_peaks` redesign. |
| 19 | 2026-07-19 | Ran GlaS Gate D SAM-B 6-case smoke. | `reports/phase2_glas/gate_d_segmenter_smoke/*` | SAM-B failed original all-case margin rule. |
| 20 | 2026-07-19 | Preregistered and ran the only allowed SAM-L alternative. | `docs/glas_gate_d_segmenter_alternative_v1.md`, `reports/phase2_glas/gate_d_segmenter_smoke_saml/*` | SAM-L also failed the all-case rule. |
| 21 | 2026-07-19 | Closed Phase 2 with strict stop status. | `reports/phase2_gate_status.md`, `reports/phase2_final_report.md`, commit `f37cf7c` | State became `STOP_GLAS_SEGMENTER_FAILURE`; Gate E/F/G/H/I blocked/prohibited. |
| 22 | 2026-07-19 | Began Phase 2D Gate D confirmation preregistration in response to the concern that 6-case all-case Gate D may be too strict. | `docs/glas_gate_d_confirmation_protocol_v1.md`, `docs/glas_gate_d_amendment_rationale.md`, `reports/phase2_glas/gate_d_confirmation/*`, `configs/segmenter/glas_gate_d_confirmation_v1.yaml` | Preregistration files exist but are not yet committed; SAM confirmation has not been run. |
| 23 | 2026-07-19 current | Implemented part of the 30-query confirmation runner. | `src/adapromptset/scripts/run_glas_gate_d_confirmation.py` | File exists, but wrapper/tests/pyproject entry/checklist updates/preregistration commit are not yet complete. |

## Current Exact State

Committed state:

- `3c1695e`: added Phase 2 GlaS strict protocol setup.
- `f37cf7c`: ran Phase 2 GlaS gate diagnostics and committed `STOP_GLAS_SEGMENTER_FAILURE`.

Uncommitted/current Phase 2D state:

- Protocol and rationale files for a 30-query Gate D confirmation exist.
- Input hashes, protocol snapshot, and compute plan exist.
- The confirmation runner source file exists.
- SAM-B/SAM-L 30-query confirmation has not been run.
- No preregistration commit has been made yet.
- The final Gate D confirmation state is unknown.

## What Should Happen Next Before Any More Claims

1. Finish the Phase 2D preregistration package.
2. Add wrapper, pyproject entry, tests, and checklist/status/log updates.
3. Commit the preregistration before any SAM inference.
4. Run exactly the preregistered 30-query SAM-B/SAM-L confirmation.
5. Generate the Gate D confirmation tables and figures only from that frozen run.
6. Update final gate status and paper-asset status from the result.

Until that happens, the missing full-policy tables and figures remain legitimately incomplete, and the previous Phase 2 result remains `STOP_GLAS_SEGMENTER_FAILURE`.
