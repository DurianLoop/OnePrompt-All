# AdaPromptSet Final Project Report

Status: `PIVOT_ORACLE_ANALYSIS`
Claim boundary: `oracle_analysis_only`

## Decision

Gate F is `FAIL_ORACLE_GAP_SMALL`: oracle gap 0.005011 is below the predeclared 0.010000 threshold.
Policy training on MoNuSAC remains prohibited.

## Evidence

| Item | Value |
|---|---|
| Best fixed action | `B_top5_pos3neg` |
| Mean best fixed Dice | 0.371238 |
| Mean oracle Dice | 0.376250 |
| Mean best fixed AJI | 0.228341 |
| Mean oracle AJI | 0.232161 |
| AJI oracle gap | 0.003820 |
| Deterministic promotion | `NOT_PROMOTED` |
| Representation status | `PIVOT_REPRESENTATION_ANALYSIS` |
| Representation claim | `oracle_analysis_only_insufficient_separation` |
| Gate F failure diagnostics | `LOW_COMPONENT_COVERAGE`, `NO_HIGHER_CALL_GAIN` |
| GlaS pilot | `PENDING_INSUFFICIENT_EPISODES` |
| New-domain readiness | `READY_WITH_BLOCKED_FORMAL_DATASETS`; GlaS `BLOCKED_INSUFFICIENT_EPISODES`, DDR `READY_FOR_INFRASTRUCTURE_ONLY` |
| Oracle-analysis figures | `PASS_ORACLE_ANALYSIS_FIGURES`; 2 files |
| Method overview figure | `PASS_METHOD_OVERVIEW_FIGURE` |
| Action-bank figure | `PASS_ACTION_BANK_FIGURE`; `prompt_set_bank_v2` |
| Data profile | `PASS` (526 rows) |
| DDR protocol | `PASS_PROTOCOL_DEFINED`; `binary_union_of_EX_HE_MA_SE` |
| Phase 0 scope | `PASS_SCOPE_STRUCTURE`; `PASS` |
| Paper section audit | `PASS`; 0 violations |
| Oracle/privileged captions | `PASS` |
| PDF metadata | `PASS`; title `AdaPromptSet: Cost-Aware Prompt Set Selection for Reference-Guided Dense Medical Segmentation`; author `Internal MVP Draft` |
| Reference exposure | `PASS` |
| Bootstrap status | `PENDING_INSUFFICIENT_CLUSTERS` |
| PDF SHA256 | `B6E8E628BA04CA64E8A1E6E45BE3A2C749E88C9030A832C8A77EE0EC365CB9C1` |

## Prohibited Next Actions

- `monusac_policy_training`
- `policy_training_on_monusac`
- `posthoc_monusac_action_redesign`
- `test_time_action_search`

## Allowed Next Actions

- `fixed_rule_action_oracle_analysis`
- `predeclared_new_domain_pilot`

## Reproduction Commands

```powershell
cd D:\SAM\adapromptset
$env:PYTHONPATH = "D:\SAM\adapromptset\src"
D:\SAM\conda_envs\sam_gpu\python.exe -m unittest discover -s tests
D:\SAM\conda_envs\sam_gpu\python.exe scripts\audit_phase0_scope_structure.py --project-root D:\SAM\adapromptset
D:\SAM\conda_envs\sam_gpu\python.exe scripts\audit_data.py
D:\SAM\conda_envs\sam_gpu\python.exe scripts\audit_data_profile.py --project-root D:\SAM\adapromptset
D:\SAM\conda_envs\sam_gpu\python.exe scripts\build_episodes.py
D:\SAM\conda_envs\sam_gpu\python.exe scripts\audit_reference_exposure.py --project-root D:\SAM\adapromptset
D:\SAM\conda_envs\sam_gpu\python.exe scripts\audit_ddr_protocol.py --project-root D:\SAM\adapromptset
D:\SAM\conda_envs\sam_gpu\python.exe scripts\audit_new_domain_pilot_readiness.py --project-root D:\SAM\adapromptset
D:\SAM\conda_envs\sam_gpu\python.exe scripts\run_candidate_pilot.py --max-episodes 5
D:\SAM\conda_envs\sam_gpu\python.exe scripts\run_oracle_gap_pilot.py --max-episodes 10
D:\SAM\conda_envs\sam_gpu\python.exe scripts\build_action_bank.py --bank-version v2
D:\SAM\conda_envs\sam_gpu\python.exe scripts\run_oracle_gap_pilot.py --max-episodes 10 --action-bank D:\SAM\adapromptset\configs\actions\prompt_set_bank_v2.json --out-dir D:\SAM\adapromptset\reports\oracle_gap_pilot_v2
D:\SAM\conda_envs\sam_gpu\python.exe scripts\write_decision_checkpoint.py
D:\SAM\conda_envs\sam_gpu\python.exe scripts\analyze_fixed_oracle_pivot.py
D:\SAM\conda_envs\sam_gpu\python.exe scripts\analyze_representation_pivot.py --project-root D:\SAM\adapromptset
D:\SAM\conda_envs\sam_gpu\python.exe scripts\audit_gate_f_failure_diagnostics.py --project-root D:\SAM\adapromptset
D:\SAM\conda_envs\sam_gpu\python.exe scripts\run_bootstrap.py --episode-regret-csv D:\SAM\adapromptset\reports\fixed_oracle_analysis\episode_regret.csv --episode-manifest-csv D:\SAM\adapromptset\manifests\episodes\monusac\episodes_val.csv --out-dir D:\SAM\adapromptset\reports\bootstrap
D:\SAM\conda_envs\sam_gpu\python.exe scripts\export_oracle_analysis_tables.py --project-root D:\SAM\adapromptset
D:\SAM\conda_envs\sam_gpu\python.exe scripts\export_oracle_analysis_figures.py --project-root D:\SAM\adapromptset
D:\SAM\conda_envs\sam_gpu\python.exe scripts\export_method_overview_figure.py --project-root D:\SAM\adapromptset
D:\SAM\conda_envs\sam_gpu\python.exe scripts\export_action_bank_figure.py --project-root D:\SAM\adapromptset
D:\SAM\conda_envs\sam_gpu\python.exe scripts\export_gate_reports.py --project-root D:\SAM\adapromptset
D:\SAM\conda_envs\sam_gpu\python.exe scripts\audit_paper_sections.py --project-root D:\SAM\adapromptset
D:\SAM\conda_envs\sam_gpu\python.exe scripts\build_paper.py --project-root D:\SAM\adapromptset --mode submission --compile
D:\SAM\conda_envs\sam_gpu\python.exe scripts\audit_pdf_metadata.py --project-root D:\SAM\adapromptset
D:\SAM\conda_envs\sam_gpu\python.exe scripts\export_final_report.py --project-root D:\SAM\adapromptset
D:\SAM\conda_envs\sam_gpu\python.exe scripts\audit_scientific_claims.py --project-root D:\SAM\adapromptset
```

## Notes

- This report is generated from real local artifacts.
- It does not claim adaptive-policy success, transfer, efficiency, or leaderboard-level performance.
- Oracle rows are analysis upper bounds, not deployable method outputs.
- Phase 0 scope keeps sequential RL, continuous coordinate generation, and online SAM-response refinement outside the current mainline.
- Paper section audit confirms Results cost and accuracy wording, Limitations reference quality coverage, and oracle/privileged caption boundaries.
- Current table captions state oracle columns or labels are analysis upper bounds, not deployable outputs, and no privileged-input row is included.
- PDF metadata audit confirms the compiled PDF title, author, subject, and keywords match the internal MVP placeholder metadata.
- Background reference evidence is not explicit SAM negative point prompting.
- Multi-reference episodes are repeated measurements and not independent samples.
- No in-domain retraining or transfer claim is made.
- Failure results and limitations are disclosed.
- GlaS remains a smoke result until a predeclared protocol supplies enough episodes.
- Bootstrap/significance remain diagnostic because the current pilot has insufficient patient clusters.
