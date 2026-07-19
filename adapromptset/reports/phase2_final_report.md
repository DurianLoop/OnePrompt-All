# AdaPromptSet Phase 2 Final Report

Date: 2026-07-19
Branch: `phase2-glas-strict`
Commit at report time: `3c1695e8c1805aafbdd0a4834dee941cbafa81c5`
Final status: `STOP_GLAS_SEGMENTER_FAILURE`

This report closes the current Phase 2 GlaS strict diagnostic branch. It does not claim policy success, deterministic final-method success, or formal testA/testB results.

## Research Direction

Phase 2 asked whether GlaS gland segmentation contains enough query-level prompt-set heterogeneity to justify later cost-aware adaptive action selection.

The MoNuSAC branch remains frozen at `PIVOT_ORACLE_ANALYSIS`. Its Gate F Dice oracle gap was `0.0050113301616486194`, below the required `0.010`, so no MoNuSAC policy training or post-hoc action redesign was allowed.

## GlaS Data And Split

| Item | Value |
| --- | --- |
| Local root | `D:\SAM\external\research_data\extracted\GlaS\Warwick QU Dataset (Released 2016_07_08)` |
| Official train pairs | `85` |
| Utility-training queries | `55` |
| Strict-validation queries | `30` |
| Strict-validation reference episodes | `90` |
| Locked testA cases | `60` |
| Locked testB cases | `20` |
| testA/testB used | `false` |
| Manifest dir | `manifests/phase2/glas` |

## Gate Status

| Gate | Status | Evidence |
| --- | --- | --- |
| Gate C candidate quality | `PASS_AFTER_PREDECLARED_REDESIGN` | `reports/phase2_glas/candidates_redesign_v1/gate_c_summary.json` |
| Gate D segmenter smoke | `FAIL_SEGMENTER_CONTROLS_CLOSE` | `reports/phase2_glas/gate_d_segmenter_smoke*.json` |
| Gate E action bank | `BLOCKED_BY_GATE_D` | not generated |
| Gate F oracle gap | `BLOCKED_BY_GATE_D` | not run |
| Gate G utility reliability | `BLOCKED_BY_GATE_D` | not run |
| Gate H policy learning | `PROHIBITED_BY_GATE_D` | not run |
| Gate I formal test | `PROHIBITED_BY_GATE_D` | testA/testB unused |

## Gate C Result

The frozen candidate generator failed component coverage:

- component coverage: `0.418820 < 0.600000`;
- decision: `REDESIGN_GLAS_CANDIDATES_ONCE_ALLOWED`.

The single predeclared redesign used `local_peaks` with positive candidate pool sizes `5,10,20` and spacing `2.0`. It passed:

- validation queries: `30`;
- reference episodes: `90`;
- precision delta: `0.433889`;
- precision ratio: `1.806818`;
- component coverage: `0.630221`;
- negative blank fraction: `0.135556`;
- coordinate validation: `PASS`;
- decision: `CONTINUE_TO_GATE_D`.

## Gate D Result

SAM-B failed Gate D:

- cases: `6`;
- mean privileged best Dice: `0.733908`;
- mean control best Dice: `0.638274`;
- issue: privileged prompts did not clearly exceed controls on at least one case.

The only allowed alternative, SAM-L, was pre-registered and also failed:

- cases: `6`;
- mean privileged best Dice: `0.762905`;
- mean control best Dice: `0.412671`;
- issue: privileged prompts did not clearly exceed controls on at least one case.

Therefore the Phase 2 GlaS path stops at `STOP_GLAS_SEGMENTER_FAILURE`. No action bank, oracle-gap analysis, policy training, or formal test is allowed under the current protocol.

## Compute

Recorded Phase 2 GPU use: `0.039522` GPU-hours.

Ledger: `reports/phase2_compute_ledger.csv`

## Paper Claim Boundary

Allowed claim:

`GlaS Gate C can be passed by a predeclared local-peak candidate redesign, but the branch stops at Gate D because SAM-B and the one allowed SAM-L alternative fail the privileged-control margin rule.`

Not allowed:

- no GlaS policy success claim;
- no GlaS deterministic final-method claim;
- no formal testA/testB claim;
- no universal transfer or model-agnostic claim.

Archived MoNuSAC oracle-analysis PDF:

`paper/archive/monusac_oracle_analysis/main.pdf`

Clean latexmk status:

`FAIL_MIKTEX_LOG_PERMISSION`

Evidence: `reports/latexmk_audit.json` reports an 8-page clean-copy PDF was produced, but `latexmk_exit_zero=false` because MiKTeX could not write `latexmk.log`.

## Main Artifacts

- `docs/phase2_goal_read_receipt.md`
- `docs/glas_protocol_v2.md`
- `docs/glas_exposure_audit.md`
- `docs/monusac_freeze_notice.md`
- `docs/glas_candidate_redesign_v1.md`
- `docs/glas_gate_d_segmenter_alternative_v1.md`
- `reports/phase2_gate_status.md`
- `reports/phase2_compute_ledger.csv`
- `reports/phase2_glas/candidates/gate_c_summary.json`
- `reports/phase2_glas/candidates_redesign_v1/gate_c_summary.json`
- `reports/phase2_glas/gate_d_segmenter_smoke/gate_d_summary.json`
- `reports/phase2_glas/gate_d_segmenter_smoke_saml/gate_d_summary.json`
- `paper/archive/monusac_oracle_analysis/main.pdf`
- `reports/phase2_final_report.json`
- `reports/phase2_final_report.md`

## One-click Reproduction Commands

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe scripts\build_glas_phase2_splits.py --project-root D:\SAM\adapromptset --seed 20260719 --train-count 55 --val-count 30 --references-per-query 3
D:\SAM\conda_envs\sam_gpu\python.exe scripts\run_candidate_pilot.py --project-root D:\SAM\adapromptset --dataset-name GlaS --episodes-csv D:\SAM\adapromptset\manifests\phase2\glas\episodes_val.csv --out-dir D:\SAM\adapromptset\reports\phase2_glas\candidates --max-episodes 90 --device auto
D:\SAM\conda_envs\sam_gpu\python.exe scripts\audit_glas_phase2_gate_c.py --project-root D:\SAM\adapromptset
D:\SAM\conda_envs\sam_gpu\python.exe scripts\run_candidate_pilot.py --project-root D:\SAM\adapromptset --dataset-name GlaS --episodes-csv D:\SAM\adapromptset\manifests\phase2\glas\episodes_val.csv --out-dir D:\SAM\adapromptset\reports\phase2_glas\candidates_redesign_v1 --max-episodes 90 --device auto --positive-layout local_peaks --positive-limits 5 10 20 --min-positive-distance 2.0
D:\SAM\conda_envs\sam_gpu\python.exe scripts\audit_glas_phase2_gate_c.py --project-root D:\SAM\adapromptset --metrics-path D:\SAM\adapromptset\reports\phase2_glas\candidates_redesign_v1\candidate_pilot_metrics.csv --candidates-path D:\SAM\adapromptset\reports\phase2_glas\candidates_redesign_v1\candidate_pilot_candidates.csv --out-dir D:\SAM\adapromptset\reports\phase2_glas\candidates_redesign_v1 --figure-dir D:\SAM\adapromptset\paper_phase2\figures\candidate_diagnostics_redesign_v1 --cache-dir D:\SAM\adapromptset\cache\glas_candidates_redesign_v1
D:\SAM\conda_envs\sam_gpu\python.exe scripts\audit_glas_phase2_gate_d.py --project-root D:\SAM\adapromptset --episodes-csv D:\SAM\adapromptset\manifests\phase2\glas\episodes_val.csv --out-dir D:\SAM\adapromptset\reports\phase2_glas\gate_d_segmenter_smoke --device cuda:0
D:\SAM\conda_envs\sam_gpu\python.exe scripts\audit_glas_phase2_gate_d.py --project-root D:\SAM\adapromptset --episodes-csv D:\SAM\adapromptset\manifests\phase2\glas\episodes_val.csv --out-dir D:\SAM\adapromptset\reports\phase2_glas\gate_d_segmenter_smoke_saml --device cuda:0 --checkpoint D:\SAM\assets\checkpoints\sam_vit_l_0b3195.pth
```
