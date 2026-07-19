# AdaPromptSet Phase 2 Read Receipt

Date: 2026-07-19
Branch: `phase2-glas-strict`
Git commit at protocol start: `016daa85a785ed9c6c272326b93977f0f70cbdfa`
Workspace: `D:\SAM`
Project root: `D:\SAM\adapromptset`
Protocol version: `adapromptset.phase2.glas_strict.v2`

## Objective Read

The Phase 2 objective is `AdaPromptSet-v2: GlaS Strict Diagnostic and Policy Gate Re-evaluation`.

This is not a restart from zero and not a mechanical completion pass over all old unchecked boxes. The previous MoNuSAC result is frozen as `PIVOT_ORACLE_ANALYSIS`. Phase 2 asks whether GlaS gland segmentation contains enough query-level prompt-set heterogeneity to justify a later cost-aware adaptive action selector.

## Required Documents Read

| Path | SHA256 |
| --- | --- |
| `AdaPromptSet_Codex_Response.md` | `B9C5F6D4EAA3BB5481D062A943218718954415FDF6E3F78CD0EBA1E97818482E` |
| `open_questions_and_gate_extensions.md` | `4F0997D5C1B5F11C381E3C55C05D5D8E507217300AA61A2512249967029F2EFF` |
| `execution.md` | `006353509203DC77DEC6D918B3D714C1AA5C902966FB090AE0233E40E2BA94DE` |
| `checklist.md` | `AA0589B87995E25C883ACB826587DEF3B4AD51791F127832BB30D5BA66F456BF` |
| `docs/goal_progress_nodes_and_skipped_assets_20260719.md` | `14254FD9D68F8B79BB69BAB1F48FD2DB3A916AB7AE4FEB6126BD39E9D077D8FD` |
| `reports/decision_checkpoint/phase5_decision.json` | `B6EC74EA3AF9984EA37F17EC38C35DDD67758F94D88E1FE9A1764143768FBE90` |
| `reports/gates/gate_f_oracle_gap.json` | `D70A77F86A34A526834A6DFB35AB88220E6C6D667E1900A275966864705FDEA2` |
| `reports/final_project_report.json` | `4F7408860F959C3F7E01E296726306D50BDBB53E450CFFF2C7BC0920989E9CB0` |
| `reports/gate_f_failure_diagnostics/gate_f_failure_diagnostics.json` | `F046ED0A3F3F730473A5D4579EC635A0FE038D7C38872ED62E65FCFCED438513` |
| `reports/gate_f_failure_diagnostics/gate_f_failure_diagnostics.md` | `8F48E876D89F1074AA70F4FCDBD0536AAC9FE1010CD22F6E5274329318571F9B` |
| `reports/representation_analysis/representation_summary.json` | `0F4F941602F08653211A036563F61C3118E690304E06A17A15B2C48D122D4E1C` |
| `reports/representation_analysis/representation_report.md` | `DE219468D82B203FEF720EEB2739BE42A17EE9CAAC3EA147D30DEE9C71508310` |
| `reports/representation_analysis/episode_representation.csv` | `40F61334A498FBE052A7A0A07E2DD0B8CDE2C5A0384BA062E0D82486D33BD8EC` |
| `reports/new_domain_pilot_readiness.json` | `B4B3F98D2802A933FA41D2CECFB63333140247F2A98AC3D8F0B0A3178133EB40` |

## Frozen MoNuSAC Facts

| Item | Value |
| --- | --- |
| Phase status | `PIVOT_ORACLE_ANALYSIS` |
| Gate F status | `FAIL_ORACLE_GAP_SMALL` |
| MoNuSAC v2 episodes | `10` |
| MoNuSAC v2 utility records | `120` |
| Best fixed action | `B_top5_pos3neg` |
| Mean best fixed Dice | `0.37123847211635835` |
| Mean oracle Dice | `0.376249802278007` |
| Dice oracle gap | `0.0050113301616486194` |
| Gate F minimum Dice gap | `0.010` |
| Mean best fixed AJI | `0.22834095079987268` |
| Mean oracle AJI | `0.23216115704701398` |
| AJI oracle gap | `0.0038202062471413` |
| Current MoNuSAC claim | `oracle_analysis_only` |

## Phase 2 Local Environment

| Item | Value |
| --- | --- |
| Python | `3.11.15` |
| PyTorch | `2.11.0+cu128` |
| CUDA available | `true` |
| CUDA runtime | `12.8` |
| GPU | `NVIDIA GeForce RTX 5090 D v2` |
| Matcher | `DINOv2-L`, frozen |
| Segmenter | `SAM-B`, frozen |
| DINOv2-L checkpoint observed | `D:\SAM\assets\checkpoints\dinov2_vitl14_pretrain.pth` |
| SAM-B checkpoint observed | `D:\SAM\assets\checkpoints\sam_vit_b.pth` |

## GlaS Data Facts at Phase 2 Start

| Item | Value |
| --- | --- |
| Local root | `D:\SAM\external\research_data\extracted\GlaS\Warwick QU Dataset (Released 2016_07_08)` |
| Official train image-mask pairs | `85` |
| Official testA image-mask pairs | `60` |
| Official testB image-mask pairs | `20` |
| Grade metadata | `Grade.csv` present |
| GlaS grade counts | `benign=74`, `malignant=91` across all 165 rows |
| Official train patients | `15` |
| Official test patients | `12` |
| Train/test patient-ID overlap | `1,2,3,4,5,7,9,10,11,13,15` |

## Phase 2 Allowed Changes

- Create Phase 2 protocol, audit, manifest, report, paper_phase2, and cache artifacts.
- Build a new GlaS official-train-only 55/30 diagnostic split.
- Use only official train images for policy/action utility training and strict validation before any formal test.
- Preserve official testA/testB from action, threshold, architecture, or paper-structure selection.
- Add prompt-statistics and reference-sensitivity exporters if they are diagnostic-only.
- Fix clean LaTeX build issues without altering scientific claims.

## Phase 2 Prohibited Changes

- Do not modify `prompt_set_bank_v2` for MoNuSAC.
- Do not resample observed MoNuSAC episodes.
- Do not lower the MoNuSAC Gate F threshold.
- Do not train a MoNuSAC adaptive policy.
- Do not use GlaS testA/testB for action-bank design, threshold tuning, model selection, or paper claim selection.
- Do not train any GlaS policy before Gate F returns `CONTINUE_GLAS_POLICY`.
- Do not claim universal, model-agnostic, or broad transfer behavior.

## Phase 2 Manifest Built

The Phase 2 GlaS split builder generated the following artifacts after protocol setup:

| Artifact | SHA256 |
| --- | --- |
| `manifests/phase2/glas/case_assignments.csv` | `F3853E2FCA693C7D202BD003E3F3C01A33A3027C0B7428B718A288EEFE5F7F48` |
| `manifests/phase2/glas/episodes_train.csv` | `07DF11179BFB944D9BAF2FA472224183E6BE30954286C38661627ED2612D9CE1` |
| `manifests/phase2/glas/episodes_val.csv` | `B687AF2180D5468A3485F2860A1DE20D4F971AF643C4F84352B086CF8CDDD2DB` |
| `manifests/phase2/glas/locked_test_cases.csv` | `71577640385F53224D03AFBB758BC72B39BA04E7D992BCBE87A10C29EE208CAE` |
| `manifests/phase2/glas/protocol_metadata.json` | `28AC00279B7F76F26B43EC74262B5652DA438BCC8C8C7C6B8698623B280477BF` |
| `reports/phase2_glas/split/split_report.json` | `7E3FA79B01F085715A586186432BB3C61C32BAA164371BC91A76967EDBD9EC3E` |

Manifest checks:

- phase2 roles: `utility_train=55`, `strict_val=30`, `locked_testA=60`, `locked_testB=20`;
- episode counts: `train=165`, `val=90`;
- query counts: `train=55`, `val=30`;
- independent leakage check found `BadCount=0` for support-not-train, self-reference, and test-query leakage;
- no testA/testB episode file was generated.

## Current Decision

The current Phase 2 state is `PHASE2_GATE_C_READY`.

Candidate diagnostics may start next. Segmenter smoke, action-bank execution, utility generation, policy training, and formal test remain gated.
