# AdaPromptSet Implementation Checklist

> **使用方式**：Codex 每完成一项，将 `[ ]` 改为 `[x]`，并在该项下方记录日期、run ID、commit hash、结果路径和异常。不得一次性批量勾选尚未验收的任务。
>
> **节点标记**：
>
> - `GATE`：必须通过后才能进入下一阶段；
> - `STOP`：失败时暂停扩大实验，先诊断；
> - `PIVOT`：需要重新收紧研究问题或改变方法；
> - `EXT`：条件性扩展，只有主线成功后才执行；
> - `LOCK`：一旦锁定，不得根据 test 结果修改；
> - `REPORT`：必须写入论文或实验日志，即使结果不符合预期。

> **Gate 失败处理总原则**：
>
> 1. gate 失败时不得直接扩大正式实验，也不得修改已观察 test 的动作库、阈值、模型或叙事；
> 2. 可启动一次或多次**预声明救援实验**，但救援实验只能使用 train/validation 或新的未观察外部数据；
> 3. 每个救援实验必须预先写清楚：要验证的失败原因、允许改变的组件、不能触碰的数据、通过/失败标准；
> 4. 救援后若仍失败，必须降级主张或 pivot，而不是把失败结果藏起来；
> 5. 使用扩展数据集时，先通过 data manifest gate，再进入 method gate。

---

## Goal-mode 决策锁定（2026-07-17）

本 checklist 现在按 `AdaPromptSet_Codex_Response.md` 执行。`open_questions_and_gate_extensions.md` 中的确认项已被该文件裁决，不再阻塞 Phase 0/1；后续只在新 evidence 与裁决冲突时重新开问题。

- [x] 最高优先级裁决文件已落盘为 `AdaPromptSet_Codex_Response.md`，并以 SHA256 记录复制一致性。
  - 2026-07-17：附件 `pasted-text.txt` 原样复制到 `D:\SAM\adapromptset\AdaPromptSet_Codex_Response.md`；SHA256=`1C14B3A06A91C6ED91A71C6AE05E0CF7D31249499B6A94EC6B4E9A3DAB13FC0F`。
- [x] MVP 范围已锁定为 MoNuSAC 主开发、MoNuSeg sanity/historical、GlaS 第一扩展；DINOv2-L + SAM-B；DDR/IDRID/CPM-17/SAM-H/SAM2/SAM3 均不进入第一轮调参主线。
  - 2026-07-17：该范围来自 `AdaPromptSet_Codex_Response.md`，优先级高于本 checklist 的长期蓝图项。

## 标准 Gate 报告（2026-07-17）

- [x] 按 `AdaPromptSet_Codex_Response.md` 第 10 节导出 `reports/gates/gate_<letter>_<name>.{json,md}`。
  - 2026-07-17：新增 `src/adapromptset/reporting/gate_reports.py`、`src/adapromptset/scripts/export_gate_reports.py` 与 `scripts/export_gate_reports.py`；命令为 `D:\SAM\conda_envs\sam_gpu\python.exe scripts\export_gate_reports.py --project-root D:\SAM\adapromptset`。
  - 2026-07-17：已生成 Gates A-I 共 18 个标准报告文件；每个 JSON 包含 gate/name/status/pass_fail、input_hashes、commands、diagnostics、rule_applied、allowed/prohibited next actions 与 deviations。
  - 2026-07-17：`reports/gates/gate_f_oracle_gap.json` 记录 `status=FAIL_ORACLE_GAP_SMALL`、`pass_fail=FAIL`、`dice_oracle_gap=0.0050113301616486194`、`phase5_status=PIVOT_ORACLE_ANALYSIS`；`reports/gates/gate_h_policy_learning.json` 记录 `PROHIBITED_BY_GATE_F_PIVOT`，继续禁止 MoNuSAC policy training。
  - 2026-07-17：`tests/test_gate_report_export.py` 覆盖 Gate F pivot 报告、Gate H 禁止状态、输入 hash、命令 provenance 与 markdown 禁止动作段落。

## Goal continuation — GlaS strict Phase 2（2026-07-19）

- [x] 创建 Phase 2 GlaS strict protocol/read-receipt/freeze/exposure/gate/ledger 文档。
  - 2026-07-19：新增 `docs/phase2_goal_read_receipt.md`、`docs/glas_protocol_v2.md`、`docs/glas_exposure_audit.md`、`docs/monusac_freeze_notice.md`、`reports/phase2_gate_status.md` 与 `reports/phase2_compute_ledger.csv`；MoNuSAC 冻结为 `PIVOT_ORACLE_ANALYSIS`。
- [x] 建立 GlaS official-train-only 55/30 strict diagnostic split。
  - 2026-07-19：`manifests/phase2/glas/*` 记录 `utility_train=55`、`strict_val=30`、`locked_testA=60`、`locked_testB=20`；validation 为 30 query / 90 reference episodes；testA/testB 未生成 episode 文件。
- [x] 运行 Phase 2 GlaS Gate C frozen-candidate diagnostic。
  - 2026-07-19：`reports/phase2_glas/candidates/gate_c_summary.json` 记录 `status=FAIL`、`decision=REDESIGN_GLAS_CANDIDATES_ONCE_ALLOWED`；precision delta=`0.422222`、precision ratio=`1.769231`、negative blank fraction=`0.135556`、coordinate validation=`PASS`，但 component coverage=`0.418820 < 0.600000`。
  - 2026-07-19：已生成 `query_level_candidate_metrics.csv`、`candidate_pool_curve.csv` 与 `paper_phase2/figures/candidate_diagnostics/{precision_vs_pool,coverage_vs_pool,negative_blank_vs_pool}.png`。
- [x] 预注册一次 GlaS candidate redesign，并在不使用 testA/testB 的前提下重新运行 Gate C。
  - 2026-07-19：`docs/glas_candidate_redesign_v1.md` 预注册唯一 redesign：`local_peaks`、pool sizes=`5,10,20`、spacing=`2.0`；`reports/phase2_glas/candidates_redesign_v1/gate_c_summary.json` 记录 `status=PASS`、`decision=CONTINUE_TO_GATE_D`，component coverage=`0.630221 >= 0.600000`。
- [x] 运行 GlaS Gate D segmenter smoke。
  - 2026-07-19：SAM-B 在 `reports/phase2_glas/gate_d_segmenter_smoke/gate_d_summary.json` 中记录 `status=FAIL`；mean privileged best Dice=`0.733908`，mean control best Dice=`0.638274`，但至少一个 case 未满足 privileged-control margin。
- [x] 按 Gate D 规则仅测试一个替代 segmenter。
  - 2026-07-19：`docs/glas_gate_d_segmenter_alternative_v1.md` 预注册 SAM-L 为唯一替代；`reports/phase2_glas/gate_d_segmenter_smoke_saml/gate_d_summary.json` 记录 `status=FAIL`，mean privileged best Dice=`0.762905`，mean control best Dice=`0.412671`，仍有 case 未满足 margin。
- [x] Gate D 失败后停止 GlaS policy 路径。
  - 2026-07-19：当前总体状态为 `STOP_GLAS_SEGMENTER_FAILURE`；Gate E/F/G/H/I 仍不得继续，testA/testB 保持 locked/unused。
- [x] 归档上一轮 MoNuSAC oracle-analysis 论文。
  - 2026-07-19：旧 8 页 oracle-analysis PDF 与 source/tables/figures/build log/claim audits 已复制到 `paper/archive/monusac_oracle_analysis/`；不删除原始 `paper/` 内容。

## Phase 2D - GlaS Gate D confirmation preregistration (2026-07-19)

- [x] Preserve previous 6-case Gate D failure as historical exploratory evidence.
  - 2026-07-19: Previous SAM-B/SAM-L Gate D reports remain in `reports/phase2_glas/gate_d_segmenter_smoke/` and `reports/phase2_glas/gate_d_segmenter_smoke_saml/`, recorded as `EXPLORATORY_GATE_D_FAIL_UNDER_ALL_CASE_MARGIN_RULE`.
- [x] Create Gate D confirmation amendment rationale before new SAM inference.
  - 2026-07-19: Added `docs/glas_gate_d_amendment_rationale.md`; it explains why the 6-case all-case rule is too strong for an execution/viability gate and states that no second Gate D criterion revision is allowed after this confirmation.
- [x] Create Gate D confirmation protocol before new SAM inference.
  - 2026-07-19: Added `docs/glas_gate_d_confirmation_protocol_v1.md`; primary family is `P3_component_grouped_centroids`, primary control is `mean(C1_random_coordinate, C2_mismatched_reference)`, statistical unit is query image, and all 30 strict-validation queries must be included.
- [x] Freeze Phase 2D config, protocol snapshot, input hashes, and compute plan.
  - 2026-07-19: Added `configs/segmenter/glas_gate_d_confirmation_v1.yaml` and `reports/phase2_glas/gate_d_confirmation/{protocol_snapshot,input_hashes,compute_plan}.json`. GlaS testA/testB remain locked and not used.
- [x] Implement CPU-verifiable confirmation runner scaffolding and focused tests.
  - 2026-07-19: Added `src/adapromptset/scripts/run_glas_gate_d_confirmation.py`, `scripts/run_glas_gate_d_confirmation.py`, and `tests/test_run_glas_gate_d_confirmation.py`. SAM confirmation inference has not run yet.
- [x] Create preregistration git commit before SAM inference.
  - Required commit message: `phase2d: preregister glas gate d confirmation`.
  - 2026-07-19: Commit `f5c2de3` created before SAM confirmation inference.
- [x] Run SAM-B 30-query confirmation.
  - 2026-07-19: `reports/phase2_glas/gate_d_confirmation/gate_d_confirmation_summary.json` records SAM-B status `FAIL`, P3 margin `-0.157937`, win rate `0.433333`, catastrophic failure rate `0.666667`.
- [x] Run SAM-L 30-query confirmation.
  - 2026-07-19: The same summary records SAM-L status `FAIL`, P3 margin `-0.096909`, win rate `0.466667`, catastrophic failure rate `0.400000`.
- [x] Generate required Phase 2D metrics, audits, tables, figures, and paper section.
  - 2026-07-19: Generated confirmation reports under `reports/phase2_glas/gate_d_confirmation/`, 10 figures under `paper_phase2/figures/gate_d_confirmation/`, tables under `paper_phase2/tables/`, and `paper_phase2/sections/gate_d_confirmation.tex`.
- [x] Write the unique final Gate D state and update `reports/phase2_final_report.{md,json}`, `reports/phase2_gate_status.md`, `docs/decision_log.md`, `docs/protocol_history.md`, and this checklist.
  - 2026-07-19: Final Phase 2D state is `STOP_GLAS_SEGMENTER_FAILURE_CONFIRMED`; Gate E/F/G/H/I remain closed/prohibited and testA/testB remain locked/unused.

Current Phase 2D status: `STOP_GLAS_SEGMENTER_FAILURE_CONFIRMED`.

## Phase 3 - GlaS Automatic Region Prompt Feasibility (2026-07-19)

- [x] Create Phase 3 branch from Phase 2D artifact commit.
  - 2026-07-19: Created `phase3-glas-region-prompts` from `93b319e`.
- [x] Complete CPU-only Phase 2D interpretation audit before new SAM inference.
  - 2026-07-19: Generated `reports/phase3_interpretation_audit/*`; status `PHASE3_INTERPRETATION_AUDIT_COMPLETE`; interpretation state `PIVOT_GLAS_REGION_PROMPT_REQUIRED`.
- [x] Add GlaS Gate D interpretation addendum.
  - 2026-07-19: Added `docs/glas_gate_d_interpretation_addendum.md`; original operational state remains `STOP_GLAS_SEGMENTER_FAILURE_CONFIRMED`.
- [x] Create Phase 3 scientific pivot and region protocol documents.
  - 2026-07-19: Added `docs/phase3_scientific_pivot.md`, `docs/glas_region_prompt_protocol_v1.md`, `docs/glas_region_prompt_exposure_audit.md`, and `docs/glas_region_prompt_claim_registry.md`.
- [x] Create 40/15 internal region split inside the 55-image utility pool.
  - 2026-07-19: Added `manifests/phase3/glas_region_calibration_40.csv`, `manifests/phase3/glas_region_selection_15.csv`, `manifests/phase3/glas_region_references.csv`, and `reports/phase3_glas/split/region_split_report.{json,md}`.
- [x] Freeze region action bank v1 and parameter grid.
  - 2026-07-19: Added `configs/actions/glas_region_bank_v1.{yaml,json}` with R00-R09 automatic actions and U01-U03 privileged upper bounds.
- [x] Create Phase 3 protocol snapshot, input hashes, gate status, and compute ledger before new SAM inference.
  - 2026-07-19: Added `reports/phase3_protocol_snapshot.json`, `reports/phase3_input_hashes.json`, `reports/phase3_gate_status.md`, and `reports/phase3_compute_ledger.csv`.
- [x] Create preregistration git commit before new SAM inference.
  - Required commit message: `phase3: preregister glas automatic region prompt study`.
  - 2026-07-19: Commit `471f529` created before the internal candidate calibration and Gate R1 evaluation.
- [x] Run internal parameter calibration.
  - 2026-07-19: Generated `reports/phase3_glas/region_candidate_calibration/*` from 165 internal reference episodes with frozen DINOv2-L/local-peaks candidate evidence. The reused Gate-C script returned a Gate-C-style FAIL, but artifacts were complete and used only for Phase 3 R1.
- [x] Run Gate R1 proposal-quality evaluation.
  - 2026-07-19: Generated `reports/phase3_glas/gate_r1_region_proposal/*`; final R1 state is `STOP_GLAS_REGION_PROPOSAL_FAILURE`.
- [x] Run SAM-B Gate R2 only if R1 passes.
  - 2026-07-19: R1 failed, so SAM-B R2 was correctly not run.
- [x] Run R3 oracle only if R2 passes.
  - 2026-07-19: R2 was blocked by R1 failure, so R3 was correctly not run.
- [x] Train router only if R3 reaches `CONTINUE_PROMPT_TYPE_ROUTER`.
  - 2026-07-19: R3 did not run, so router training was correctly prohibited.
- [x] Access testA/testB only if a final method is frozen.
  - 2026-07-19: No final method was frozen; testA/testB remained locked and unused.
- [x] Generate `reports/phase3_final_report.{md,json}` and claim-safe `paper_phase3` PDF/source.
  - 2026-07-19: `reports/phase3_final_report.{md,json}` exists; `paper_phase3/main.tex` and `paper_phase3/build/main.pdf` were built as a claim-safe 8-page negative/gated-study paper. PDF SHA256=`B75339E91F5F9BCE1F77DA8FA84E3E1F9095DB2623954D78E376B9FB14B0DD2A`; log scan passed for overfull/undefined/missing/rerun warnings; 8 rendered PNG pages were verified.

Current Phase 3 status: `STOP_GLAS_REGION_PROPOSAL_FAILURE`.

## Phase -1 — 当前工作区可行性核对（2026-07-17 已审）

此阶段用于约束 `D:\SAM` 当前工作区的真实起点。它不是方法贡献本身，但必须先读完，避免把长期蓝图误当作已经具备的工程状态。

- [x] 核对 `adapromptset/` 当前只包含 `execution.md` 与 `checklist.md`，尚未包含可运行包、配置、脚本、测试或 artifact 目录。
  - 2026-07-17：已确认 `D:\SAM\adapromptset` 当前只有两个 md 文件。
- [x] 核对可复用的相邻实现存在于 `D:\SAM\FSPL`，包括 candidate pools、DINOv2/SAM wrapper、fixed rules、utility policy、statistics、stage scripts 与 tests。
  - 2026-07-17：已确认 `D:\SAM\FSPL\fspl` 与 `D:\SAM\FSPL\scripts` 可作为 AdaPromptSet-MVP 的迁移起点。
- [x] 核对本地数据边界：`MonuSeg\official`、`MoNuSAC\processed`、`IDRID`、DDR、GlaS、CPM-17 原始目录与 nnU-Net 派生数据存在，但 readiness 不同。
  - 2026-07-17：GlaS 已在 `external\research_data\extracted\GlaS` 中确认 85/60/20 image-mask pairs；DDR 已确认 train/valid/test 和 4 类 lesion labels；CPM-17 只确认到 `cpm15` 15 image + 15 `.mat` labels，不满足当前 32/32 脚本假设。
- [x] 核对本地权重边界：DINOv2-B/L、SAM-B/L/H 与 SAM2.1 权重存在；第一版主线使用 DINOv2-L + SAM-B，SAM-L/H 用于复现，SAM2/SAM3 暂不进入 MVP。
  - 2026-07-17：SAM3 为 Windows 适配式 bring-up，不作为主 segmenter。
- [x] 核对历史实验边界：FSPL 已支持固定 spatial layout/radius-3 在 MoNuSAC 上的正效应，但 adaptive selector 外部转移失败。
  - 2026-07-17：AdaPromptSet 不得预设 policy 正结果；必须先过 oracle-gap 与 validation gate。

### [GATE--1] MVP 范围判定

- [x] 第一轮只创建 `AdaPromptSet-MVP`，复用/迁移 FSPL 稳定模块，不重写完整论文流水线。
  - 2026-07-17：已创建独立 `src/adapromptset` 包、CPU-only audit CLI、基础配置与日志；尚未进入 GPU/论文矩阵。
- [x] 第一轮只以 MoNuSAC 为主开发数据、MoNuSeg 为 sanity/cross-check；GlaS 作为第一扩展；DDR/IDRID 作为眼底边界测试；CPM-17 补齐前推迟。
  - 2026-07-17：`docs/data_exposure_log.md` 已记录 MoNuSAC tuning_allowed=True，MoNuSeg/GlaS tuning_allowed=False；DDR/IDRID/CPM-17 未进入第一轮 audit CLI。
- [x] 第一轮只跑 10-20 query action-bank/oracle-gap pilot，再决定是否训练完整 policy。
  - 2026-07-17：已在 10 个 MoNuSAC validation episodes 上运行 MVP subset oracle-gap pilot；`reports/oracle_gap_pilot/oracle_gap_pilot_summary.json` 记录 Gate F=`FAIL_ORACLE_GAP_SMALL`。
- [x] 若 oracle gap 小、oracle action 单一、或 policy 不优于 validation best fixed rule，立即转入 fixed-rule/action-oracle 分析，不继续扩大 policy。
  - 2026-07-17：MVP subset Dice oracle gap=0.0047 < 0.010，不启动 policy；按 `[STOP-F1]` / `[RESCUE-F1]` 进入预声明动作库扩展诊断。
  - 2026-07-17：Rescue-F1 v2 Dice oracle gap=0.0050 仍 < 0.010；`reports/gate_f_status.md` 记录 `PIVOT_F_REQUIRED`，policy training 继续禁止。
  - 2026-07-17：已执行 fixed-rule/action-oracle pivot analysis；当前 recommended claim=`oracle_analysis_only`。

**通过标准**：MVP 任务边界与 `execution.md` 的“当前工作区可行性附录”一致。

**若不通过**：

- [ ] `[STOP--1]` 先重写执行范围，不得直接启动完整 Phase 0-16。

---

## Phase 0 — 项目初始化与研究边界

- [x] 创建 `execution.md` 指定的仓库目录结构。
  - 2026-07-17：新增 `scripts/audit_phase0_scope_structure.py` / `src/adapromptset/scripts/audit_phase0_scope_structure.py`；`reports/phase0_scope_structure/phase0_scope_structure.json` 记录 structure status=`PASS`，required_path_count=26，missing_paths=[]。
- [x] 配置 `pyproject.toml`、`environment.yml`、代码格式化、lint 和 pytest。
  - 2026-07-17：`pyproject.toml` 已声明 pytest/ruff dev extra、package discovery、script entry points 与 `[tool.ruff]` / `[tool.ruff.lint]` / `[tool.ruff.format]`；`environment.yml` 已包含 pytest 与 ruff。`reports/phase0_scope_structure/phase0_scope_structure.json` 记录 toolchain status=`PASS`、`lint_config_present=true`。
- [x] 记录 Python、PyTorch、CUDA、GPU、matcher 和 segmenter 版本。
  - 2026-07-17：`reports/matcher_runtime.json` 记录 Python、platform、torch、CUDA availability/version、GPU name，以及 DINOv2-L/SAM-B architecture 与 checkpoint path。
- [x] 下载并校验冻结 matcher checkpoint，保存 SHA256。
  - 2026-07-17：本地冻结 matcher `D:\SAM\assets\checkpoints\dinov2_vitl14_pretrain.pth` 已校验；SHA256=`D5383EA8F4877B2472EB973E0FD72D557C7DA5D3611BD527CEEB1D7162CBF428`。
- [x] 下载并校验冻结 segmenter checkpoint，保存 SHA256。
  - 2026-07-17：本地冻结 segmenter `D:\SAM\assets\checkpoints\sam_vit_b.pth` 已校验；SHA256=`EC2DF62732614E57411CDCF32A23FFDF28910380D03139EE0F4FCBE91EB8C912`。
- [x] 在 README 写明 matcher 与 segmenter 全程冻结。
  - 2026-07-17：`README.md` 当前执行边界写明 frozen MVP matcher/segmenter scope 为 DINOv2-L + SAM-B。
- [x] 在代码中加入 assertion，训练 optimizer 不得包含 matcher/segmenter 参数。
  - 2026-07-17：`src/adapromptset/training/optimizer_guard.py` 实现 `assert_optimizer_excludes_frozen_modules`，即使 frozen 参数 `requires_grad=False` 但仍在 optimizer param groups 中也会抛出 `FrozenParameterViolation`。
  - 2026-07-17：`scripts/audit_optimizer_guard.py --out-dir reports/optimizer_guard` 仅执行 toy safety audit，不启动 AdaPromptSet policy training；`reports/optimizer_guard/optimizer_guard.json` 记录 status=`PASS`、policy_training=`not_started`、checked_roles=`matcher,segmenter`，SHA256=`409D0A9BB08C54D0A3952646B05C10EEA1C540C7E3D624081775BDB2BB652D57`。
  - 2026-07-17：`tests/test_optimizer_guard.py` 与 `tests/test_optimizer_guard_cli.py` 覆盖通过路径、matcher violation、segmenter violation 和 CLI PASS/FAIL 报告。
- [x] 实现 run metadata 和 config hash。
  - 2026-07-17：`src/adapromptset/reproducibility/metadata.py` 与 `scripts/write_run_metadata.py` 已实现 canonical config hash、input/config file SHA256、compact git metadata、Python/Torch/CUDA/GPU metadata；`tests/test_run_metadata.py` 覆盖 hash 稳定性、GT 字段排除和 JSON 写出。
  - 2026-07-17：当前 Phase 0 record 写入 `reports/run_metadata/phase0_reproducibility/run_metadata.json`；config hash=`1DD5A11E2C6A897F0EDA671F0E10A77B0F5C3FB9F706341F0EFDB98383954F60`，metadata file SHA256=`1BAEB43D97C91E697B2E7F132DB777AED6EB4B4D79A167BF0CDAF75A302D0A27`。
- [x] 实现模拟结果水印和 submission build guard。
  - 2026-07-17：`src/adapromptset/scripts/build_paper.py` 新增 mode-aware paper build：draft mode 写入 `\simulatedtrue` 与 `DRAFT WATERMARK`，submission mode 写入 `\simulatedfalse` 且 `guard_submission_placeholders` 拒绝 `SIMULATED PLACEHOLDER` / `simulatedtrue`。`tests/test_build_paper.py` 覆盖 draft watermark、submission 清除水印与 guard 行为。
- [ ] 将论文主张锁定为：budget-constrained structured prompt-set selection。
  - 2026-07-17：`paper/main.tex` 与 `paper/sections/*.tex` 当前写成 oracle-analysis-only skeleton，并记录 `PIVOT_ORACLE_ANALYSIS`；完整论文主张锁定仍待 Phase 15/16 正式验收。
- [x] 将 sequential RL、连续坐标生成和在线 SAM response refinement 标为非主线。
  - 2026-07-17：`reports/phase0_scope_structure/phase0_scope_structure.json` 记录 `sequential_rl`、`continuous_coordinate_generation`、`online_sam_response_refinement` 均 present=true 且 mainline=false；policy training 仍为 false。

### [GATE-0] 范围验收

- [x] 能用一句话解释研究问题，且不使用“first to learn prompts”。
  - 2026-07-17：`reports/phase0_scope_structure/phase0_scope_structure.json` 写入一句话研究问题，并记录 `does_not_claim_first_to_learn_prompts=true`。
- [x] 能明确说明与 GF-SAM、SynPo、AlignSAM、PPD 的区别。
  - 2026-07-17：同一审计报告记录 `similar_work_boundaries` 中 GF-SAM、SynPo、AlignSAM、Point Prompt Defender 均 present=true。
- [x] `execution.md` 中的三条 RQ 和五条假设未被擅自扩大。
  - 2026-07-17：同一审计报告记录 `rq_count=3`、`hypothesis_count=5`、status=`PASS`。

**通过标准**：仓库可安装，`pytest` 至少能运行空测试；研究范围无歧义。

**若不通过**：

- [ ] `[STOP-0]` 暂停写模型代码，先修订任务定义和 action semantics。

---

## Phase 1 — 数据获取、许可与泄漏审计

当前 `D:\SAM` MVP 的数据优先级：先用已存在的 `MoNuSAC\processed` 做主开发/held-out protocol，用 `MonuSeg\official` 做小规模 sanity 与交叉检查。GlaS 已有完整 image-mask pairs，可作为第一扩展；DDR/IDRID 用于眼底病灶边界测试；CPM-17 需先补齐/修复 manifest，不阻塞 Phase 0-lite 与 pilot oracle-gap。

### MoNuSeg

- [ ] 获取官方 MoNuSeg 数据和许可信息。
- [x] 记录所有图像、mask、case/slide/patient ID 和文件 hash。
  - 2026-07-17：`manifests/monuseg_manifest.csv` 已写入 37 train / 14 test case-level rows、image/mask path 与 SHA256；MoNuSeg 当前没有可靠 patient ID，暂以 case_id 作为 patient_id。
- [ ] 验证 train 30 / official test 14 的当前数据版本。
- [ ] 检查 mask 是 semantic 还是 instance，并实现统一读取。
- [ ] 按 patient/case 建立 train/validation，不按随机 patch 跨病例切分。
- [ ] 保留 official test，不用其结果选择任何超参数。

### GlaS

- [x] `[EXT-DATA]` 确认本地 GlaS 解压目录存在，并包含 85 train / 60 testA / 20 testB 图像与 annotation pairs。
  - 2026-07-17：路径为 `D:\SAM\external\research_data\extracted\GlaS\Warwick QU Dataset (Released 2016_07_08)`；已有 `build_glass_strict_policy_splits.py` 可迁移/复用。
- [ ] 获取 GlaS 数据和许可信息。
- [x] 验证 85 train / 60 test A / 20 test B 的当前版本。
  - 2026-07-17：`manifests/glas_manifest.summary.json` 记录 source split counts: train=85, testA=60, testB=20。
- [x] 记录 slide/patient/benign-malignant 元数据。
  - 2026-07-17：`manifests/glas_manifest.csv` 从 `Grade.csv` 记录 patient_id 与 grade。
- [x] 建立 patient/slide-isolated train/validation。
  - 2026-07-17：Phase 1 先建立 role-level patient isolation：official test patients 对应的 source-train cases 标为 `excluded_overlap`；GlaS 严格后仅剩 role=train 8 cases，后续只作为扩展确认，不作为早期调参主数据。

### CPM-17

- [x] `[EXT-DATA]` 确认本地 CPM-17 原始目录存在，但当前不满足正式 split 构建要求。
  - 2026-07-17：`D:\SAM\external\research_data\raw\CPM17\cpm15` 只有 15 image + 15 `.mat` labels；`train/test` 目录未形成当前脚本要求的 32/32 image-label pairs。
- [ ] 获取 CPM-17 当前公开版本和许可。
- [ ] 验证 32 train / 32 test 或实际版本数量，并记录偏差。
- [ ] 统一标注与 instance mask 格式。

### DDR / IDRID / nnU-Net 派生对照

- [x] 核对 DDR lesion segmentation 数据存在。
  - 2026-07-17：`D:\SAM\code\ddr_datasets\DDR-dataset-01\lesion_segmentation` 包含 train 383、valid 149、test 225 images；每张图对应 EX/HE/MA/SE 四类 `.tif` labels。
- [x] 核对 IDRID train/test image-mask 数量。
  - 2026-07-17：training 55 images/54 masks，testing 27 images/27 masks；已有 test 观察历史，不能用于新的调参式外部正结果。
- [x] 核对 nnU-Net 派生数据存在。
  - 2026-07-17：`Dataset901_OCSAMPhase3` 与 `Dataset902_OCSAMPhase3Stratified` 存在；Dataset902 描述为 IDRID + MonuSeg stratified，适合强基线对照。
- [x] 定义 DDR 多 lesion mask 合并/分任务规则。
  - 2026-07-17：新增 `src/adapromptset/scripts/audit_ddr_protocol.py`、`scripts/audit_ddr_protocol.py` 与 entry point `adapromptset-audit-ddr-protocol`；命令为 `D:\SAM\conda_envs\sam_gpu\python.exe scripts\audit_ddr_protocol.py --project-root D:\SAM\adapromptset --data-root D:\SAM\code\ddr_datasets\DDR-dataset-01\lesion_segmentation`。
  - 2026-07-17：`reports/ddr_protocol_audit/ddr_protocol_audit.json` 记录 status=`PASS_PROTOCOL_DEFINED`；Stage 1 规则为 `binary_union_of_EX_HE_MA_SE` 且 allowed_use=`infrastructure_only`；Stage 2 为 EX/HE/MA/SE 四个 reference-defined intents，先采用 positive-query protocol。该项只关闭 DDR protocol/配对审计，不允许正式 DDR policy training。
- [x] 明确 IDRID 是否仅作为记录性负结果/边界测试，不作为新 policy selection。
  - 2026-07-17：由 `AdaPromptSet_Codex_Response.md` 锁定为 read-only/non-tunable。
- [x] 明确 Dataset902 只用于监督强基线，不回流 AdaPromptSet policy。
  - 2026-07-17：由 `AdaPromptSet_Codex_Response.md` 与 `docs/data_exposure_log.md` 共同约束；Dataset902 未进入第一轮 tuning/audit。

### 通用审计

- [x] 实现 `audit_data.py`。
  - 2026-07-17：`scripts/audit_data.py` 调用 `src/adapromptset/data/audit.py`，生成 MoNuSAC/MoNuSeg/GlaS manifests、audit JSON、exposure log、gate status 与 compute ledger。
- [x] 输出 `artifacts/manifests/data_audit.json`。
  - 2026-07-17：MVP 实际输出路径为 `reports/dataset_audit/{monusac,monuseg,glas}.json` 与 `manifests/*.summary.json`；未使用旧蓝图中的 `artifacts/manifests/data_audit.json` 单文件路径。
- [x] 输出 split overlap report。
  - 2026-07-17：`reports/gate_status.md` 与各 dataset audit JSON 记录 role-level Gate A；MoNuSAC/GlaS source train-test patient overlap 通过 `excluded_overlap` 显式隔离。
- [x] 输出图像尺寸、mask 面积、实例数分布。
  - 2026-07-17：新增 `src/adapromptset/scripts/audit_data_profile.py`、`scripts/audit_data_profile.py` 与 entry point `adapromptset-audit-data-profile`；命令为 `D:\SAM\conda_envs\sam_gpu\python.exe scripts\audit_data_profile.py --project-root D:\SAM\adapromptset`。
  - 2026-07-17：`reports/data_profile_audit.json` 与 `.md` 记录 status=`PASS`、dataset_count=3、row_count=526、issues=[]。摘要：GlaS 165 cases，mean size 758.74×515.51，mean mask fraction 0.507109，mean instance/label count 9.27；MoNuSAC 310 cases，mean size 602.38×558.70，mean mask fraction 0.220378，mean instance count 159.07；MoNuSeg 51 cases，1000×1000，mean mask fraction 0.253054，semantic-label count 1.00。
- [x] 检查同一 WSI/case 是否跨 split。
  - 2026-07-17：Gate A 检查 duplicate case_id 与 role-level train/test patient overlap；当前三个 MVP 数据集均 PASS。
- [x] 检查 reference pool 只来自 train。
  - 2026-07-17：新增 `src/adapromptset/scripts/audit_reference_exposure.py`、`scripts/audit_reference_exposure.py` 与 entry point `adapromptset-audit-reference-exposure`；命令为 `D:\SAM\conda_envs\sam_gpu\python.exe scripts\audit_reference_exposure.py --project-root D:\SAM\adapromptset`。
  - 2026-07-17：`reports/reference_exposure_audit.json` 记录 status=`PASS`、`reference_pool_train_only=true`、total_issues=0；覆盖 MoNuSAC 1505 episodes、MoNuSeg 51 episodes、GlaS 88 episodes。
- [x] 实现 leakage unit tests。
  - 2026-07-17：`tests/test_audit_manifest.py` 覆盖配对缺失、SHA256、MoNuSAC patient manifest、GlaS Grade.csv、Gate A patient overlap rejection、manifest/exposure 输出。

### [GATE-A] 数据完整性

- [x] patient/case overlap = 0。
  - 2026-07-17：role-level Gate A 对 MoNuSAC/MoNuSeg/GlaS 均 PASS；source-level overlap 未隐藏，MoNuSAC 9 cases、GlaS 77 cases 被标记为 `excluded_overlap`。
- [x] reference-test leakage = 0。
  - 2026-07-17：`reports/reference_exposure_audit.json` 记录 `reference_test_leakage_zero=true`；所有 support/reference rows 均为 `support_role=train`，且未指向 test case、test patient、test image path 或 test mask path。
- [x] 每个文件有 hash 和来源记录。
  - 2026-07-17：`manifests/*_manifest.csv` 包含 image_sha256/mask_sha256，`manifests/*.csv.sha256` 记录 manifest 文件哈希。
- [x] 所有 dataset 数量差异已在日志解释。
  - 2026-07-17：`manifests/*_manifest.summary.json` 与 `reports/dataset_audit/*.json` 记录 source split 与 role_counts；GlaS 严格隔离后 role=train=8 已作为风险记录。

**若不通过**：

- [ ] `[STOP-A]` 禁止运行正式实验。
- [ ] 修复 ID 解析或重新下载数据。
- [ ] 若无法恢复 patient IDs，在论文中降低统计单位并明确限制；不得假装患者隔离。

---

## Phase 2 — Episode 与冻结 feature pipeline

- [x] 实现 reference-query episode schema。
  - 2026-07-17：`src/adapromptset/data/protocol.py` 输出 support/query case、patient、path、role 和 source split 字段。
- [x] 确保 reference 与 query 不来自同一 case。
  - 2026-07-17：`validate_gate_b` 与 `tests/test_protocol_episodes.py` 检查 support_case_id != query_case_id。
- [x] 生成 train episodes。
  - 2026-07-17：`manifests/episodes/*/episodes_train.csv` 已生成；MoNuSAC=810，MoNuSeg=29，GlaS=7。
- [x] 生成固定 validation episodes。
  - 2026-07-17：`manifests/episodes/*/episodes_val.csv` 已生成；MoNuSAC=190，MoNuSeg=8，GlaS=1。
- [x] 生成固定 test episodes。
  - 2026-07-17：`manifests/episodes/*/episodes_test.csv` 已生成；MoNuSAC=505，MoNuSeg=14，GlaS=80。
- [x] 记录每个 manifest 的 seed 和 hash。
  - 2026-07-17：episode protocol metadata 记录 seed、references_per_query、val_patient_count；source manifest hash sidecars 已在 `manifests/*.csv.sha256`。
- [x] 实现多 reference 的 query-level 聚合。
  - 2026-07-17：`src/adapromptset/statistics/units.py` 将 dataset/role/query/action 下的多 reference rows 聚合为 query-level 均值，并记录 `reference_episode_count` 与 support case IDs。
- [x] 实现 query/patient-level统计单位。
  - 2026-07-17：`scripts/audit_statistical_units.py` 写入 `reports/statistical_units/statistical_units.{json,md}`；MoNuSAC train/val/test 推荐 patient-cluster 统计单位，MoNuSeg/GlaS smoke splits 为 query-level。
- [ ] 实现 matcher feature extraction。
- [x] 实现坐标 resize/pad 和逆映射测试。
  - 2026-07-17：新增 `src/adapromptset/geometry/resize_pad.py` 与 `tests/test_resize_pad_transform.py`，实现 `(x,y)` 点在 original image 与 resized+letterbox padded image 之间的正向/逆向映射、批量映射和 padded-to-original pixel clamp/rounding。
  - 2026-07-17：单元测试覆盖 wide/tall image padding、边界点 inverse round-trip、批量点顺序保持和非法尺寸拒绝；该项只完成坐标几何工具与 round-trip 测试，不代表 feature cache 已完成。
- [ ] 缓存 reference/query features。
- [ ] 验证重复运行 feature hash 相同。

### [GATE-B] Episode 可复现

- [x] 同一 seed 重建 manifest 完全一致。
  - 2026-07-17：`tests/test_protocol_episodes.py` 覆盖同 seed deterministic equality；真实 `scripts/build_episodes.py` 运行通过。
- [x] 不同方法读取同一 manifest。
  - 2026-07-17：episode manifests 固化在 `manifests/episodes/{monusac,monuseg,glas}`，后续 fixed/adaptive/oracle 必须读取这些 CSV。
- [ ] feature cache 不包含 query GT。
- [x] 多 reference 不被当作独立测试样本。
  - 2026-07-17：statistical-unit audit 明确 `multi_reference_episodes_are_independent=false`；MoNuSAC train/val/test 每个 query 最多 5 个 reference episodes，只能作为重复测量聚合。

**若不通过**：

- [ ] `[STOP-B]` 修复数据管线，不得继续 baseline。

---

## Phase 3 — Matching evidence 与 candidate pools

- [x] 实现 foreground prototype。
  - 2026-07-17：`src/adapromptset/matching/evidence.py` 实现 support mask foreground patch weights 与 weighted normalized prototype；`scripts/run_real_matcher_smoke.py` 已在 MoNuSAC validation episode 上完成 DINOv2-L forward smoke。
- [x] 实现 background prototype。
  - 2026-07-17：`src/adapromptset/matching/evidence.py` 实现 support background patch weights 与 weighted normalized prototype。
- [x] 实现 robust normalization。
  - 2026-07-17：`src/adapromptset/candidates/points.py` 实现 z-score foreground/background normalization 与 contrast score-map kernel；已接入 DINOv2-L validation smoke/pilot。
- [x] 实现 `s_fg`、`s_bg`、`s_ctr`。
  - 2026-07-17：`matching_scores_from_features` 输出 `s_fg`、`s_bg`、`s_ctr=s_fg-lambda*s_bg`；真实 smoke artifact 为 `reports/real_matcher_smoke/score_maps.pt`。
- [x] 实现 matching entropy、margin、high-score area 等全局特征。
  - 2026-07-17：`src/adapromptset/matching/evidence.py` 新增 `score_map_global_features`，输入 2D score map 和可选 valid mask，输出 score_count、score_mean/std、top score/index、second score、top margin、归一化 matching entropy、high-score threshold 与 high-score area fraction。
  - 2026-07-17：`tests/test_matching_evidence.py` 覆盖 entropy/margin/high-score-area、valid mask 排除无效 cell、top index 与非法形状/参数的基础行为；`scripts/run_real_matcher_smoke.py` 已将 `s_fg`、`s_bg`、`s_ctr` 的全局特征写入 `reports/real_matcher_smoke/real_matcher_smoke.json` 的 `score_map_features`。
- [ ] 实现 positive candidate sources：global top、local maxima、coverage、fg-only。
  - 2026-07-17：已实现 `topk` 与 `spread` 两类 score-map positive selection kernel；local maxima/coverage/fg-only 语义尚未接入真实 matcher evidence，因此本项保持未完成。
- [ ] 实现 negative candidate sources：high-bg、negative contrast、ambiguous band、positive ring。
  - 2026-07-17：已实现 contrast-low negative selection kernel，并与 positive 去重；high-bg/ambiguous/ring 的专门来源尚未完成，因此本项保持未完成。
- [x] 实现图像有效区域 mask，排除 padding/空白背景。
  - 2026-07-17：`build_candidate_pools(..., valid_mask=...)` 支持 score-grid valid mask，并由 `tests/test_candidates.py` 验证 invalid cell 不会被选中。
- [x] 实现 candidate dedup 和坐标边界检查。
  - 2026-07-17：`validate_candidate_pools` 检查 duplicate grid index、未知 kind 与 image coordinate out-of-bounds。
- [ ] 输出至少 50 个 episode 的候选可视化。
- [ ] 人工抽查候选是否落在合理图像区域。
- [x] 实现 point precision 与 component coverage。
  - 2026-07-17：`point_precision` 与 `component_coverage` 已有 synthetic mask 单元测试；仅用于 Gate C 诊断，不参与候选生成。

### [GATE-C] Candidate 质量

建议 pilot 标准，不是最终论文阈值：

- [x] Top candidate point precision 明显高于随机点。
  - 2026-07-17：`reports/candidate_pilot/candidate_pilot_summary.json` 记录 5 个 MoNuSAC validation episodes，positive_limit=5 时 mean precision=0.32，feature-grid random baseline=0.20，delta=0.12，高于预声明 margin=0.05。
- [x] candidate pool component coverage 随 pool size 增加而合理上升。
  - 2026-07-17：同一 pilot 覆盖 `positive_limit` 1/3/5，并由 `summarize_candidate_pilot` 检查 coverage 不随 pool size 下降；mean component coverage at K=5 为 0.0069，绝对值低，后续 oracle/action 阶段必须谨慎解释。
- [x] 负点不大量集中在 padding、纯白或纯黑区域。
  - 2026-07-17：pilot 增加 `negative_blank_fraction` 诊断；K=5 mean negative blank fraction=0.0。
- [x] 无坐标错位、镜像、转置或缩放错误。
  - 2026-07-17：`validate_candidate_pools` 在 pilot 中 `validation_issue_count=0`；真实 DINOv2 smoke 和 multi-episode pilot 均通过 coordinate/duplicate checks。

**若不通过**：

- [ ] `[STOP-C1]` 检查坐标映射与 feature stride。
- [ ] `[STOP-C2]` 检查 reference mask resize。
- [ ] `[PIVOT-C]` 若单 prototype 在病理背景上明显失败，启动 multi-prototype pilot，但先不扩大 policy。

---

## Phase 4 — Frozen segmenter 与共享 fusion

- [ ] 实现 image embedding cache。
- [x] 实现正点/负点标签输入。
  - 2026-07-17：`src/adapromptset/actions/bank.py` 与 `scripts/run_sam_smoke.py` 已验证 positive/negative prompt label arrays 可被 SAM-B 接收。
- [ ] 实现 multimask selection 的 validation-frozen 规则。
- [x] 实现一个 prompt group 对应一次 mask decode。
  - 2026-07-17：`PromptGroup` 到 SAM predictor 的 smoke path 已按 group 调用 mask decode；详见 `reports/sam_smoke/sam_smoke_summary.json`。
- [x] 实现每个 action 的真实 calls 统计。
  - 2026-07-17：`sam_call_count(action)` 对 valid action 以 prompt group 数计费，invalid action cost=0；`reports/action_bank/action_bank_summary.json` 已记录每个 action 的 `sam_calls`。
- [x] 实现 mask area filter。
  - 2026-07-17：`src/adapromptset/fusion/masks.py` 实现 `filter_masks_by_area`，`tests/test_fusion_masks.py` 覆盖 known case。
- [x] 实现 mask IoU NMS。
  - 2026-07-17：`nms_masks` 按 score 降序保留非重复 mask，`mask_iou`/NMS known case 已测试。
- [x] 实现 semantic union。
  - 2026-07-17：`semantic_union` 与 `fuse_masks` 已实现，作为共享 fusion 输出。
- [ ] 实现 instance overlap arbitration 或统一 watershed。
- [x] 实现 Dice、IoU、AJI known-case 单元测试。
  - 2026-07-17：`tests/test_segmentation_metrics.py` 覆盖 Dice、IoU 与 AJI known cases。
- [x] 验证所有方法使用同一 fusion config。
  - 2026-07-17：MVP 共享配置落盘为 `configs/segmenter/fusion_mvp.json`；content hash=`5F15975FBE2BBF0E16B2A06DAF0A6F67F0BF7E50824F97CE6B3B66C0A743A8F4`，由 `load_fusion_config` 单元测试锁定。
- [ ] 测量 image encoding 与 mask decoding 的独立时间。

### [GATE-D] Segmenter 执行正确性

- [x] GT-center privileged point 能得到合理 mask，证明 API 和坐标正确。
  - 2026-07-17：SAM-B API/坐标链路通过 component-aware 检查；GT-center 所在 connected component Dice=0.9382、IoU=0.8836。整张 semantic-union Dice=0.0084 仅作为 one-point SAM prompt 的诊断限制，不作为 wrapper correctness gate。
- [x] 增加负点时标签方向正确。
  - 2026-07-17：`reports/sam_smoke/sam_smoke_summary.json` 记录 positive+negative prompt group 被 SAM-B 接收，且 `tests/test_action_bank.py` 覆盖 label array 构造。
- [x] calls 与 groups 一一对应。
  - 2026-07-17：SAM smoke 记录 call_count=4、prompt_group_count=4。
- [x] 同一 prompts 重复运行输出一致或在允许数值误差内一致。
  - 2026-07-17：SAM smoke 对 candidate prompt 重复 decode，mask exact equality 为 true。

**若不通过**：

- [ ] `[STOP-D]` 不得生成 action utility 数据；先修复 segmenter wrapper。

---

## Phase 5 — Fixed baselines 与完整 action bank

- [x] 实现 Top-1/3/5/10 one-group。
  - 2026-07-17：`build_mvp_action_bank` 生成 `A_top1/3/5/10_one_group`；当前 candidate pilot 只含 5 个正点，因此 `A_top10_one_group` 在 manifest 中正确标为 invalid。
- [x] 实现 fixed hard positive/negative。
  - 2026-07-17：MVP v1 生成 `B_top3_pos1neg`、`B_top3_pos3neg`、`B_top5_pos1neg`、`B_top5_pos3neg`，负点来自统一 candidate pool 的 negative section。
- [x] 实现 independent positive groups。
  - 2026-07-17：MVP v1 生成 `C_top3_independent` 与 `C_top5_independent`，每个正点一个 prompt group，calls 分别为 3/5。
- [ ] 实现 KMeans groups。
- [ ] 实现 DBSCAN/MeanShift 或明确删去不稳定动作。
- [ ] 实现 connected local-peak groups。
- [ ] 实现 SynPo-style baseline。
- [ ] 实现 GF-SAM-style graph baseline。
- [ ] 实现 safe anchors。
- [x] 所有动作只读取统一 candidate pools。
  - 2026-07-17：候选索引顺序锁定为 positives-then-negatives；`candidate_coordinates` 与 `build_action_bank.py` 均读取 `reports/candidate_pilot/candidate_pilot_candidates.csv`。
- [x] 所有 action 有稳定 ID、family、config 和 hash。
  - 2026-07-17：`action_manifest_record` 写出 `action_id`、`family`、`config` 与 `content_hash`；bank hash=`586342F7077C4CAD769436F2BD191762C8ABCD3805F97D6ED371BFB21010DDF7`。
  - 2026-07-17：Rescue-F1 v2 追加 `rescue_split_groups` 与 `rescue_group_specific_negatives` actions；`prompt_set_bank_v2` hash=`35E44B3B937A9D6AA9E969CDAFC9AC6F45A77E5BA1EF72BC6770E4B5F16D75B8`。
- [x] 实现 action validity 和 invalid reason。
  - 2026-07-17：当前 `prompt_set_bank_v1` 记录 10 actions、9 valid、1 invalid；`A_top10_one_group` 的 invalid_reason=`requires_10_positive_candidates`。
  - 2026-07-17：`prompt_set_bank_v2` 记录 13 actions、12 valid、1 invalid；invalid 仍来自当前 5-positive pilot 无法支撑 `A_top10_one_group`。
- [x] 实现重复 prompt group 的 content-hash mask cache。
  - 2026-07-17：`group_cache_key`/`action_cache_keys` 对 prompt coordinates 与 labels 生成稳定 SHA256，可用于后续跨 action 复用相同 group decode。
- [x] 验证 action cost 和 runtime。
  - 2026-07-17：`scripts/run_action_utility_smoke.py --max-actions 9` 在单个 MoNuSAC validation episode 上记录每个 action 的 `sam_calls`、`decoder_calls`、`cache_hits` 与 `runtime_ms`；完整多 query runtime 仍待 Phase 6/7 扩展。
- [ ] 在 validation 上锁定动作参数。

### [LOCK-1] 动作库冻结

- [ ] 将最终动作库版本写为 `prompt_set_bank_v1`。
  - 2026-07-17：MVP subset bank 已写为 `configs/actions/prompt_set_bank_v1.{json,yaml}`，但 KMeans/local-peak/SynPo/GF-SAM/safe anchors 尚未实现，因此仍不是完整最终动作库。
  - 2026-07-17：Rescue-F1 subset bank 已写为 `configs/actions/prompt_set_bank_v2.{json,yaml}`；它是 Gate F rescue 冻结版本，不是完整论文动作库。
- [x] 保存 YAML、hash、冻结日期。
  - 2026-07-17：`configs/actions/prompt_set_bank_v1.yaml` 与 JSON manifest 已保存；bank hash=`586342F7077C4CAD769436F2BD191762C8ABCD3805F97D6ED371BFB21010DDF7`。
  - 2026-07-17：`configs/actions/prompt_set_bank_v2.yaml` 与 JSON manifest 已保存；bank hash=`35E44B3B937A9D6AA9E969CDAFC9AC6F45A77E5BA1EF72BC6770E4B5F16D75B8`。
- [ ] 从此不得根据 test 结果新增/删除动作。

### [GATE-E] Baseline 公平性

- [ ] 所有自动 baseline 使用相同 episodes。
- [ ] 所有自动 baseline 使用相同 candidate pools。
- [ ] 所有方法使用相同 fusion/postprocess。
- [ ] privileged baselines 单独分组。
- [ ] GF-SAM-style 的预调用全部计入成本。
- [ ] SynPo-style 的额外 encoder/matching 成本计入 runtime。

**若不通过**：

- [ ] `[STOP-E]` 修复协议；不能用不公平主表进入 policy 阶段。

---

## Phase 6 — Pilot Oracle Gap：最关键立项节点

- [x] 在 10–20 个 train/validation query pilot 上运行全部 actions。
  - 2026-07-17：已对 10 个 MoNuSAC validation episodes 运行当前 `prompt_set_bank_v1` 的 9 个 valid MVP subset actions，共 90 条 utility records；完整扩展动作库尚未运行。
  - 2026-07-17：已对同一 10 个 MoNuSAC validation episodes 运行 `prompt_set_bank_v2` 的 12 个 valid rescue subset actions，共 120 条 utility records。
- [x] 估算完整 action-utility 数据的 GPU-hours。
  - 2026-07-17：10-query MVP subset pilot 记录 `gpu_hours_estimate=0.005537`；该估算仅覆盖当前 subset action bank。
  - 2026-07-17：10-query Rescue-F1 v2 pilot 记录 `gpu_hours_estimate=0.008595`。
- [x] 计算每个 query 的 oracle action。
  - 2026-07-17：`oracle_action_counts` 为 `B_top5_pos3neg=5`、`B_top5_pos1neg=3`、`B_top3_pos3neg=1`、`A_top5_one_group=1`。
  - 2026-07-17：v2 `oracle_action_counts` 为 `B_top5_pos3neg=5`、`B_top5_pos1neg=2`、`B_top3_pos3neg=1`、`A_top5_one_group=1`、`R_split5_2groups_localneg=1`。
- [x] 计算 best fixed action。
  - 2026-07-17：validation subset best fixed action 为 `B_top5_pos3neg`，mean Dice=0.3712。
  - 2026-07-17：fixed-oracle pivot analysis 再次确认 v2 best fixed action 为 `B_top5_pos3neg`。
- [x] 计算 Dice oracle gap。
  - 2026-07-17：mean oracle Dice=0.3759，mean best fixed Dice=0.3712，Dice oracle gap=0.0047，低于 Gate F 阈值 0.010。
  - 2026-07-17：Rescue-F1 v2 mean oracle Dice=0.3762，mean best fixed Dice=0.3712，Dice oracle gap=0.0050，仍低于 Gate F 阈值 0.010。
- [x] 计算 AJI oracle gap。
  - 2026-07-18：`execute_action` 现在在 action utility CSV 中记录 `aji`，并已重跑 MoNuSAC Rescue-F1 v2 10-episode pilot；`reports/oracle_gap_pilot_v2/oracle_gap_pilot_summary.json` 记录 `aji_available=true`、mean oracle AJI=0.232161、mean best fixed AJI=0.228341、AJI oracle gap=0.003820。该 AJI gap 同样低于 checklist 中 0.015 的辅助阈值，不改变 `PIVOT_ORACLE_ANALYSIS` 与 MoNuSAC policy-training 禁止状态。
- [x] 统计每个 action/family 成为 oracle 的频率。
  - 2026-07-17：oracle family counts 为 `single_group_positive_negative=9`、`single_group_positive=1`；independent grouping 未成为 oracle。
  - 2026-07-17：v2 oracle family counts 为 `single_group_positive_negative=8`、`single_group_positive=1`、`rescue_group_specific_negatives=1`；rescue family 只赢 1/10。
- [x] 计算最常见单一 action 的 oracle 占比。
  - 2026-07-17：top oracle action fraction=0.5，未触发 single-action dominance，但 gap 太小。
- [x] 画 pilot action heatmap。
  - 2026-07-17：`reports/decision_checkpoint/oracle_action_heatmap.csv` 已保存 v2 的 10 episodes × 12 valid actions，共 120 行，并标记每个 episode 的 oracle action。
- [x] 检查 oracle 是否只是偏好更多 calls。
  - 2026-07-17：mean oracle SAM calls=1.0，best fixed SAM calls=1；当前 oracle 优势不是由更多 calls 驱动。
- [x] 在严格相同 budget 下重新计算 oracle gap。
  - 2026-07-17：budget 1/3/5 下 Dice oracle gap 均为 0.0047，因为所有 oracle actions 均为 1-call actions。
  - 2026-07-17：v2 budget 1 gap=0.0047，budget 2/3/5 gap=0.0050；提升来自一个 2-call rescue action，但幅度不足。
  - 2026-07-17：v2 budget Pareto analysis 在 budget 1/2/3/5 下 best fixed 均为 `B_top5_pos3neg`；gap 范围 0.004667-0.005011，仍低于 0.010。

### [GATE-F] 是否值得学习 policy

建议通过标准：

- [ ] 主预算下 oracle − best fixed ≥ 0.010 Dice，或 AJI ≥ 0.015。
- [ ] 最常见单一 action 的 oracle 占比 < 70%。
- [ ] 至少三个 action families 在不同 query 上成为 oracle。
- [ ] oracle 优势在 matched budget 下仍存在。

#### 若全部或大部分通过

- [ ] 记录 `[GO-F]`，进入完整 utility 数据生成。

#### 若 oracle gap 小

- [x] `[STOP-F1]` 暂停 policy 训练。
  - 2026-07-17：Gate F=`FAIL_ORACLE_GAP_SMALL`，禁止进入 policy 训练。
- [x] 检查 candidate pool 是否缺乏覆盖。
  - 2026-07-17：新增 `src/adapromptset/scripts/audit_gate_f_failure_diagnostics.py`、`scripts/audit_gate_f_failure_diagnostics.py` 与 entry point `adapromptset-audit-gate-f-failure-diagnostics`；命令为 `D:\SAM\conda_envs\sam_gpu\python.exe scripts\audit_gate_f_failure_diagnostics.py --project-root D:\SAM\adapromptset`。
  - 2026-07-17：`reports/gate_f_failure_diagnostics/gate_f_failure_diagnostics.json` 记录 coverage status=`LOW_COMPONENT_COVERAGE`，max-pool mean component coverage=0.052004904966278354；该诊断解释 Gate F 失败风险，但不允许 MoNuSAC post-hoc action redesign。
- [x] 检查 proposal bank 是否都是近似动作。
  - 2026-07-17：同一诊断报告记录 proposal status=`ACTIONS_SEPARATED`，mean top-2 Dice margin=0.02044071753259889，near-tie episode fraction=0.4；因此不能简单宣称所有 actions 都近似，当前 pivot 仍限定为 action-oracle/representation analysis。
- [x] 检查 SAM 对点数/grouping 是否不敏感。
  - 2026-07-17：同一诊断报告记录 SAM sensitivity status=`NO_HIGHER_CALL_GAIN`，best high-call minus best one-call Dice=-0.020569709494256128；当前 v2 pilot 未显示更多 calls/grouping 带来足够收益。
- [x] 启动一次预声明救援实验 `[RESCUE-F1]`：加入更有差异的 local negatives、group-specific negatives、spread/radius variants、graph groups 或 GF-SAM-style grouping。
  - 2026-07-17：已执行 v2 subset rescue：`R_split5_2groups`、`R_split5_2groups_localneg`、`R_top5_independent_localneg`。
- [x] 重新冻结为 `prompt_set_bank_v2`，只允许在 train/validation 上完成。
  - 2026-07-17：`configs/actions/prompt_set_bank_v2.{json,yaml}` 已冻结；action_count=13，valid_action_count=12。
- [x] 重新执行 pilot oracle gap。
  - 2026-07-17：`reports/oracle_gap_pilot_v2/oracle_gap_pilot_summary.json` 记录 Gate F=`FAIL_ORACLE_GAP_SMALL`；`reports/gate_f_status.md` 记录 policy decision=`PIVOT_F_REQUIRED`。
  - 2026-07-17：`reports/decision_checkpoint/phase5_decision.json` 与 `.md` 记录唯一 Phase 5 状态为 `PIVOT_ORACLE_ANALYSIS`；允许 `fixed_rule_action_oracle_analysis` 或预声明新域 pilot，禁止 MoNuSAC policy training 与 post-hoc action redesign。
- [ ] 若 MoNuSAC oracle gap 小但 GlaS/DDR 任务形态明显不同，可启动 `[RESCUE-F2]`：在 GlaS 或 DDR 上重建小 pilot，检验 action heterogeneity 是否是数据域问题；不得把已观察的 MoNuSAC/IDRID test 用作选择依据。
  - 2026-07-17：已预注册并执行严格 GlaS val smoke：`reports/new_domain_pilot_glas_protocol.md`、`reports/candidate_pilot_glas/`、`reports/oracle_gap_pilot_glas/`。candidate pilot 在 1 个 strict val episode 上 PASS；oracle pilot 为 `PENDING_INSUFFICIENT_EPISODES`，episode_count=1、gate_min_episodes=10，不能作为 GlaS Gate F 结论。
  - 2026-07-18：新增 `src/adapromptset/scripts/audit_new_domain_pilot_readiness.py`、`scripts/audit_new_domain_pilot_readiness.py` 与 entry point `adapromptset-audit-new-domain-pilot-readiness`；命令为 `D:\SAM\conda_envs\sam_gpu\python.exe scripts\audit_new_domain_pilot_readiness.py --project-root D:\SAM\adapromptset`。该审计只读已有 artifact，不启动 policy 训练或新 GPU 实验。
  - 2026-07-18：`reports/new_domain_pilot_readiness.json` 记录 status=`READY_WITH_BLOCKED_FORMAL_DATASETS`；GlaS=`BLOCKED_INSUFFICIENT_EPISODES`（episode_count=1、gate_min_episodes=10），DDR=`READY_FOR_INFRASTRUCTURE_ONLY`，IDRID=`READ_ONLY_NON_TUNABLE`，CPM-17=`DEFERRED_MISSING_OR_INCOMPLETE`。允许的补充实验为预声明 GlaS train/val diagnostic、DDR Stage-1 infrastructure smoke、CPM-17 missing-asset report；禁止 MoNuSAC policy training、post-hoc MoNuSAC action redesign、test-time action search，以及把 insufficient-episode smoke 升格为 gate pass。
- [ ] 若所有 dense nuclei 数据都 near-oracle，记录 `[REPORT-F1]`：固定 spatial diversification 可能已足够，policy 主张需降级。

#### 二次 redesign 后 oracle gap 仍小

- [x] `[PIVOT-F]` 放弃“学习 policy 是必要的”主张。
  - 2026-07-17：v2 rescue 后 oracle gap 仍太小；在 MoNuSAC 当前证据上不得继续训练或包装 adaptive policy 正结果。
- [x] 从以下方向选择一个并记录理由：
  - [ ] 转为系统性分析论文：simple fixed prompting is near-oracle；
  - [ ] 转为 deterministic cost-aware prompt grouping；
  - [x] 转为 action-oracle/representation analysis；
    - 2026-07-17：`reports/fixed_oracle_analysis/fixed_oracle_summary.json` 记录 mean oracle gap=0.005011、near-oracle episode fraction=0.6、top oracle action fraction=0.5，未满足 deterministic promotion。
    - 2026-07-17：`reports/representation_analysis/representation_summary.json` 记录 `PIVOT_REPRESENTATION_ANALYSIS`，mean top-2 Dice margin=0.020441、near-tie episode fraction=0.4、mean candidate point precision=0.52、mean component coverage=0.0520；claim 限定为 `oracle_analysis_only_insufficient_separation`，仍禁止 MoNuSAC policy training。
  - [ ] 更换到真正具有 action heterogeneity 的任务/数据集；
  - [ ] 结束项目，不继续堆模型。

#### 若 oracle gap 大但几乎总由同一动作产生

- [ ] `[PIVOT-F2]` 不训练复杂 policy；将该动作作为新 deterministic method，重新评估创新性。
- [ ] 可补充 `[RESCUE-F3]`：在 GlaS 或 DDR 上验证该单一动作是否仍占据 oracle；若跨域仍单一，则论文主线转为 deterministic rule；若跨域出现 action heterogeneity，改为 domain-conditioned action-selection 问题。

---

## Phase 7 — 完整 Action-Utility 数据生成

- [ ] 估算并批准计算预算。
- [ ] 对 train episodes 运行全部合法 actions。
- [ ] 对 validation episodes 运行全部合法 actions。
- [ ] 缓存重复 segmenter calls。
- [ ] 支持断点续跑。
- [ ] 每个 shard 写 checksum。
- [ ] 验证每个 episode 的 action 数量与配置一致。
- [ ] 检查 invalid actions 分布。
- [ ] 检查 utility NaN/Inf。
- [ ] 检查 Dice/IoU 逻辑关系。
- [ ] 检查 calls 与 runtime 异常值。
- [ ] 输出 `action_utility.parquet`。
- [ ] 生成 action-level EDA 报告。

### [GATE-G] Utility 数据质量

- [ ] 不存在 test GT 进入 train files。
- [ ] 每个 action 的结果可由 prompt 坐标和 mask hash 追溯。
- [ ] utility 分布不是近乎常数。
- [ ] oracle gap 与 pilot 同方向。
- [ ] 没有某个动作因实现 bug 获得异常优势。

**若不通过**：

- [ ] `[STOP-G]` 修复数据后重跑受影响 shards；不得带脏标签训练 policy。

---

## Phase 8 — Policy 实现与训练

- [ ] 实现 episode global encoder。
- [ ] 实现 positive candidate DeepSets encoder。
- [ ] 实现 negative candidate DeepSets encoder。
- [ ] 实现 prompt group encoder。
- [ ] 实现 action encoder。
- [ ] 实现 budget embedding。
- [ ] 实现 pairwise ranking loss。
- [ ] 实现 Huber auxiliary regression。
- [ ] 实现 top-action auxiliary classification。
- [ ] 实现 absolute utility labels。
- [ ] 实现 anchor-relative utility labels。
- [ ] 实现 grouped-by-episode dataloader。
- [ ] 实现 utility difference threshold。
- [ ] 实现 early stopping。
- [ ] 实现 5-seed training。
- [ ] 记录每个 seed 的 config、checkpoint 和 validation metrics。
- [ ] 验证 policy 参数量 < 2M 或解释偏差。
- [ ] 验证 optimizer 不含 matcher/segmenter 参数。

### [GATE-H] Policy 是否真正学到 action utility

建议 validation 通过标准：

- [ ] policy 比 validation best fixed ≥ 0.008 Dice。
- [ ] oracle recovery ≥ 25%。
- [ ] 95% bootstrap CI 不明显偏负。
- [ ] predicted utility 与 true utility 有正 rank correlation。
- [ ] policy 改善不是来自超出预算或更多未计 calls。

#### 若通过

- [ ] `[GO-H]` 锁定 architecture、loss 和 checkpoint selection rule。

#### 若 oracle gap 大但 policy 不提升

- [ ] `[STOP-H1]` 检查 label leakage 与 normalization。
- [ ] 比较 regression、pairwise、listwise。
- [ ] 检查 action features 是否能区分 actions。
- [ ] 检查 episode features 是否能解释 oracle family。
  - 2026-07-17：已用现有 v2 oracle-gap utility matrix 与 candidate diagnostics 生成 pilot-level representation pivot 诊断：`reports/representation_analysis/episode_representation.csv` 与 `representation_summary.json`。该诊断只覆盖 candidate point precision/component coverage 与 regret/top-2 margin 的弱关联，尚未训练 oracle-family classifier，故本项不勾选。
- [ ] 训练 oracle-family classifier 作为可学习性诊断。
- [ ] 检查 train/val reference 分布差异。
- [ ] 检查 policy 是否只学习 action frequency prior。
- [ ] 启动 `[RESCUE-H1]`：训练更简单的 binary/ordinal gate，例如是否采用 spread、是否加入负点、是否提高 budget；若简单 gate 有效，主张降级为 constrained action gate，而不是完整 set utility predictor。
- [ ] 启动 `[RESCUE-H2]`：在 train/validation 内加入 domain-conditioned heads 或 dataset token 消融，验证失败是否来自跨域混合；dataset token 只能作为消融，不能在主模型中偷学 test domain 捷径。
- [ ] 启动 `[RESCUE-H3]`：若 MoNuSAC 失败但 GlaS pilot 通过，可把 GlaS 作为新的 policy 学习主场；必须重新锁定数据、动作库和 test 协议。

#### 允许的一轮方法扩展

- [ ] `[EXT-H1]` 用 Set Transformer 替代 DeepSets。
- [ ] `[EXT-H2]` 加入 candidate-action cross-attention。
- [ ] `[EXT-H3]` 改为 listwise softmax utility distribution。
- [ ] `[EXT-H4]` 加入 calibrated uncertainty/abstention。
- [ ] `[EXT-H5]` 若 DDR/IDRID 被纳入，增加 domain-specific candidate generators，但必须在眼底域 train/validation 上预声明，不能根据 observed IDRID test 调整。

#### 扩展后仍失败

- [ ] `[PIVOT-H]` 将论文转为 action oracle/representation analysis，或停止项目。
- [ ] 不得通过 test 调参继续挽救。

---

## Phase 9 — Validation 锁定与正式测试

- [ ] 锁定 data manifests。
- [ ] 锁定 action bank hash。
- [ ] 锁定 matcher/segmenter checkpoints。
- [ ] 锁定 candidate config。
- [ ] 锁定 fusion/postprocess。
- [ ] 锁定 policy architecture。
- [ ] 锁定 checkpoint selection rule。
- [ ] 锁定主预算 B=3 或经 validation 确定的唯一主预算。
- [ ] 锁定次预算集合 `{1,3,5,8}`。
- [ ] 写 `frozen_test_protocol.yaml`。
- [ ] 由脚本检查无 test-specific threshold。

### [LOCK-2] 正式测试锁定

- [ ] 记录 lock commit hash。
- [ ] 正式 test 运行前备份 validation 结果。
- [ ] 不再修改主方法和 baseline 配置。

### 正式测试

- [ ] 运行 best fixed baselines。
- [ ] 运行 SynPo-style。
- [ ] 运行 GF-SAM-style。
- [ ] 运行 5 个 policy seeds。
- [ ] 运行 privileged upper bounds，仅分析。
- [ ] 离线计算 oracle，仅分析，不反馈方法。
- [ ] 保存所有 predictions 和 prompt coordinates。
- [ ] 计算 query-level metrics。
  - 2026-07-17：已实现 reference-row 到 query-level 的 metric 聚合工具；正式主测试尚未在完整结果表上调用该聚合。
- [ ] 计算 patient-cluster bootstrap。
  - 2026-07-17：`src/adapromptset/statistics/bootstrap.py` 与 `scripts/run_bootstrap.py` 已实现 query-level paired patient-cluster bootstrap；当前 v2 pilot 报告为 `reports/bootstrap/bootstrap_summary.{json,md}`，状态 `PENDING_INSUFFICIENT_CLUSTERS`（2 queries / 1 patient cluster），因此不能作为正式 CI/显著性证据，正式主测试仍未完成。
- [ ] 计算 paired significance。
  - 2026-07-17：已实现 query-level exact sign-flip paired permutation test；当前 v2 pilot p-value=0.5，但因为统计单位不足，仅作为诊断记录，不作为正式显著性证据。
- [ ] 计算 win/tie/loss。
  - 2026-07-17：`reports/bootstrap/bootstrap_summary.json` 已输出 pilot-level win/tie/loss=1/1/0；正式主测试 win/tie/loss 尚未完成。
- [ ] 计算 oracle recovery。

### [GATE-I] 主结论是否成立

建议成功标准：

- [ ] Policy − best fixed ≥ 0.010 Dice。
- [ ] Dice difference 95% CI 不跨 0。
- [ ] AJI 不显著恶化；最好同步提升。
- [ ] matched budget 下仍提升。
- [ ] 结果不是由单个异常 patient/query 驱动。

#### 若 Dice 提升但 AJI 降低

- [ ] `[STOP-I1]` 分析 merge rate 和 grouping。
- [ ] 检查 semantic union 是否牺牲 instance separation。
- [ ] 论文不得声称 dense instance quality 全面提升。
- [ ] 可将主任务限定为 semantic region segmentation，或改进共享 fusion 后只在 validation 重选；不得针对 test 调。

#### 若平均提升但 CI 跨 0

- [ ] `[REPORT-I2]` 如实报告不显著。
- [ ] 增加更多独立 test cases 只能通过预先确定的外部数据集，不得重复采样同一 query 冒充样本量。
- [ ] 将结论降为趋势或 exploratory。
- [ ] 可启动 `[RESCUE-I2]`：在此前未参与模型/动作选择的 GlaS、DDR 或补齐后的 CPM-17 上执行预注册外部验证；该验证只能支持“新数据上的独立结果”，不能修改原 test 结论。

#### 若主测试无提升

- [ ] `[PIVOT-I]` 不得隐藏结果或只挑预算。
- [ ] 分析 failure subsets。
- [ ] 检查 validation overfitting。
- [ ] 决定是否转为负结果/分析论文或结束。
- [ ] 可启动 `[RESCUE-I3]`：若 failure subset 与数据形态强相关，补做 domain-specific candidate/policy 诊断，但只在 train/validation 或新的未观察外部数据上完成；已观察 test 只用于报告失败模式。
- [ ] 若 GlaS/DDR 外部验证正向但原主 test 失败，主张必须改为 domain-conditional，不得写 universal policy。

---

## Phase 10 — Factorial ablation 与机制分析

- [ ] 完成 2×2×2 adaptive positive/negative/grouping ablation。
- [ ] 保持同一 budget 和 candidate pool。
- [ ] 计算主效应。
- [ ] 计算两两交互。
- [ ] 计算三阶交互。
- [ ] 完成 global-only / candidate encoder / group encoder 消融。
- [ ] 完成 ranking vs regression。
- [ ] 完成 absolute vs relative utility。
- [ ] 完成 single vs multi-reference。
- [ ] 完成 point precision/component coverage。
  - 2026-07-17：`reports/representation_analysis/representation_summary.json` 已记录 pilot-level candidate point precision/component coverage 与 regret 的相关诊断（pearson precision vs regret=-0.2141，coverage vs regret=-0.1441）；完整 Phase 10 机制分析仍未完成，因此本项不勾选。
- [ ] 完成 oracle family 分布。
  - 2026-07-17：pilot-level v2 oracle/action summary 已在 `reports/fixed_oracle_analysis/action_summary.csv`；完整 Phase 10 机制分析仍未完成。
- [ ] 完成 utility calibration。

### [GATE-J] 贡献组件是否一致

- [ ] 至少一个核心组件有稳定独立增益。
- [ ] 完整模型不是仅因更多 calls 获益。
- [ ] 若负点贡献为零，删除“joint negative selection”作为主要贡献。
- [ ] 若 grouping 贡献为零，删除“multi-instance grouping”作为主要贡献。
- [ ] 若仅 global features 有效，简化模型并避免过度宣称 set reasoning。

**分支**：

- [ ] `[PIVOT-J1]` 负点无效 → 收紧为 cost-aware positive/group allocation。
- [ ] `[PIVOT-J2]` grouping 无效 → 收紧为 adaptive positive/negative budget selection。
- [ ] `[PIVOT-J3]` policy 只学 action prior → 改为更简单的 calibrated action classifier，并如实表述。

---

## Phase 11 — 跨数据集与迁移

执行顺序按本地 readiness 排序：GlaS 优先，其次 DDR/IDRID 边界测试，CPM-17 只有在 32/32 或实际版本 manifest 通过后进入。任何跨数据集实验都必须先写 frozen transfer / in-domain retraining 的区别，不能把重训结果写成零样本迁移。

### Data readiness gates

- [ ] `[DATA-GlaS]` 运行或迁移 `build_glass_strict_policy_splits.py`，生成 patient-disjoint case assignments 与 train/val/test episodes。
  - 2026-07-17：已有 `manifests/episodes/glas/`，但 role-level strict validation 只有 1 个 episode；后续若要正式 GlaS Gate F，需预声明 train/val 诊断或重新冻结不泄漏的 GlaS protocol。
  - 2026-07-18：`reports/new_domain_pilot_readiness.json` 明确当前 GlaS 只能作为 smoke/readiness 证据；除非新的预声明 protocol 提供足够 query-level units，否则不得训练 GlaS policy、不得评估 official testA/testB、不得把该 smoke 写成 Gate F 通过。
- [x] `[DATA-DDR]` 决定 EX/HE/MA/SE 是合并成 lesion-all，还是作为四个 intent 分任务；写入 mask 合并规则和评价指标。
  - 2026-07-17：Stage 1 lesion-all union 仅用于 loader、mask merge、coordinate、candidate、segmenter smoke 与 runtime infrastructure；Stage 2 formal protocol 为 EX/HE/MA/SE 四 intent per-class evaluation 和 macro average。真实配对审计：train 383/383 complete、valid 149/149 complete、test 225/225 complete，total_missing_label_count=0。
- [ ] `[DATA-IDRID]` 确认 IDRID 只作为记录性边界测试或新协议中未观察 split；不得再次用 observed test 选方法。
- [ ] `[DATA-CPM17]` 补齐或修复 CPM-17 32/32 image-label pairs；解析 `.mat` instance labels；若无法补齐，则从正式 cross-dataset 表中删除。

### In-domain retraining

- [ ] 在 GlaS 建立 action utility。
- [ ] 在 GlaS 训练 in-domain policy。
- [ ] 在 GlaS official test 评估。
- [ ] 在 CPM-17 建立 action utility。
- [ ] 在 CPM-17 训练 in-domain policy。
- [ ] 在 CPM-17 official test 评估。

### Frozen transfer

- [ ] 冻结 MoNuSeg policy 到 CPM-17，不看 target labels 重训。
- [ ] 冻结 MoNuSeg policy 到 GlaS，不看 target labels 重训。
- [ ] 冻结 GlaS policy 到 MoNuSeg，作为补充。
- [ ] 每个 target 与 target best fixed 比较。
- [ ] 统计 CI 和 non-inferiority/positive improvement。

### [GATE-K] 泛化表述

- [ ] 若只有 target-retrained policy 有效：只写 `plug-compatible with retraining` 或 `in-domain learned policy`。
- [ ] 若 frozen policy 在同类 nuclei target 稳定正向：可写 limited same-task domain transfer。
- [ ] 若 frozen policy 在 GlaS 失败：明确不能跨目标形态 transfer。
- [ ] 不得因为代码可运行就写 model-agnostic。

---

## Phase 12 — Reference sensitivity 与失败案例

- [ ] 每个 test query 使用固定多个 reference。
- [ ] 计算跨 reference Dice std。
- [ ] 计算 10th percentile 和 worst-reference Dice。
- [ ] 计算 adaptive action consistency。
- [ ] 分析 reference mask area、appearance gap 对 action 的影响。
- [ ] 选择至少一个 reference failure case。
- [ ] 检查 policy 是否降低 reference 偶然性。

### [GATE-L] Reference 鲁棒性

- [ ] 若均值提升但 reference std 大幅增加，论文必须报告 trade-off。
- [ ] 若 policy 只对高质量 reference 有效，加入 reference quality limitation。
- [ ] 不得只汇报多 reference 聚合结果隐藏单 reference 不稳定。

---

## Phase 13 — 效率与成本

- [ ] 在同一 GPU 进行 warm-up。
- [ ] 固定 batch size = 1 或明确部署 batch。
- [ ] 测量 matching time。
- [ ] 测量 candidate/proposal time。
- [ ] 测量 policy time。
- [ ] 测量 image encoder time。
- [ ] 测量每次 mask decoder time。
- [ ] 测量 fusion time。
- [ ] 测量 total wall-clock。
- [ ] 测量 peak VRAM。
- [ ] 报告实际 calls 和 points。
- [ ] 报告 train utility generation GPU-hours。
- [ ] 生成 Dice–calls Pareto curve。
- [ ] 生成 Dice–runtime Pareto curve。

### [GATE-M] 效率主张

- [ ] Calls 更少但 runtime 不更少 → 不能简单写 efficient，需解释 matcher/overhead。
- [ ] Policy runtime 可忽略但 action generation 成本高 → 报告完整成本。
- [ ] GF-SAM 需要额外 masks 建图 → 全部计入 calls。
- [ ] 只有真实 wall-clock 和质量同时改善时，才写 efficiency gain。

---

## Phase 14 — 条件性扩展实验

只有 `[GATE-I]` 主结论通过后，才可勾选以下扩展。

### [EXT-N1] One-step response-conditioned refinement

- [ ] 先运行一次 anchor action。
- [ ] 从 SAM response 构造 uncertainty/overshoot evidence。
- [ ] 选择一次额外 correction action。
- [ ] 所有额外 calls 计入预算。
- [ ] 与 AlignSAM-style iterative prompting 清楚区分。

### [EXT-N2] Multi-prototype matching

- [ ] foreground multi-prototype。
- [ ] background multi-prototype。
- [ ] 比较 candidate oracle gap 是否增大。
- [ ] 检查提升来自 matcher 还是 policy。

### [EXT-N3] 第二 matcher/segmenter

- [ ] DINOv2 ↔ pathology encoder。
- [ ] SAM ↔ SAM2。
- [ ] 区分重新训练 policy 与 frozen transfer。
- [ ] 不建立过大的 8×模型矩阵，除非资源充足且主线已稳定。

### [EXT-N4] Sequential RL

- [ ] 仅当 one-shot action bank 明显限制 oracle 时考虑。
- [ ] 先证明新增 sequential action 的 oracle gain。
- [ ] 定义 state/action/reward/calls。
- [ ] 与 AlignSAM、PPD 做算法边界对比。
- [ ] 不得仅因“RL 更高级”而启动。

---

## Phase 15 — 图表与论文生成

### 表格

- [x] 自动生成 related-work table。
  - 2026-07-17：新增 `src/adapromptset/reporting/related_work.py`、`src/adapromptset/scripts/export_related_work.py` 与 `scripts/export_related_work.py`；生成 `reports/related_work_boundaries/related_work_boundaries.{json,md}` 和 `paper/tables/table_related_work.tex`。JSON status=`PASS_RELATED_WORK_BOUNDARIES`，覆盖 GF-SAM、SynPo、AlignSAM、Point Prompt Defender 四个边界，并记录不声明 first-to-learn prompts；`paper/sections/02_related.tex` 已自动引用该表。本项只关闭 related-work table 资产，不代表完整论文相关工作章节最终验收。
- [ ] 自动生成 main comparison。
- [ ] 自动生成 budget table。
- [ ] 自动生成 oracle heterogeneity。
  - 2026-07-17：已生成 pilot-level oracle-analysis 资产：`paper/tables/table_oracle_pivot.tex`、`paper/tables/table_action_summary.tex`、`paper/generated/oracle_analysis_macros.tex` 与 `reports/oracle_analysis_paper_assets.md`。这些资产来自真实 `reports/fixed_oracle_analysis/` 与 GlaS smoke artifact，结论限定为 `oracle_analysis_only`；完整 Phase 15 oracle heterogeneity/论文主表仍未完成，因此本项不勾选。
  - 2026-07-17：已将 representation-pivot 诊断接入 claim-safe paper assets：`paper/tables/table_representation_pivot.tex`、`paper/generated/oracle_analysis_macros.tex` 中的 `RepresentationAnalysis*` 宏，以及 `reports/oracle_analysis_paper_assets.md` 中的 representation claim / prohibited actions。该资产来自真实 `reports/representation_analysis/representation_summary.json`，结论仍限定为 `oracle_analysis_only_insufficient_separation`，不支持 MoNuSAC policy training。
  - 2026-07-18：新增 `src/adapromptset/scripts/export_oracle_analysis_figures.py`、`scripts/export_oracle_analysis_figures.py` 与 entry point `adapromptset-export-oracle-analysis-figures`；命令为 `D:\SAM\conda_envs\sam_gpu\python.exe scripts\export_oracle_analysis_figures.py --project-root D:\SAM\adapromptset`。该脚本从真实 `reports/fixed_oracle_analysis/budget_pareto.csv` 与 `action_summary.csv` 生成 `paper/figures/oracle_pareto.png` 和 `paper/figures/oracle_action_wins.png`，并写入 `reports/oracle_analysis_figures.{json,md}`。当前只是 oracle-analysis pivot 图，不代表完整 adaptive-policy oracle heterogeneity 图表已完成。
- [ ] 自动生成 factorial ablation。
- [ ] 自动生成 architecture ablation。
- [ ] 自动生成 cross-dataset table。
- [ ] 自动生成 frozen transfer table。
- [ ] 自动生成 prompt statistics。
- [ ] 自动生成 reference sensitivity。
- [ ] 自动生成 efficiency table。
- [ ] 所有表格从 parquet/CSV 自动导出，不手抄数字。
  - 2026-07-17：当前新增的 oracle-analysis partial tables 由 `scripts/export_oracle_analysis_tables.py` 从 JSON/CSV 自动导出；完整论文所有表格仍未覆盖，因此本项不勾选。
  - 2026-07-17：representation pivot partial table 同样由 `scripts/export_oracle_analysis_tables.py` 从 `reports/representation_analysis/representation_summary.json` 自动导出；完整论文所有表格仍未覆盖，因此本项不勾选。

### 图

- [x] TikZ method overview。
  - 2026-07-18：新增 `src/adapromptset/scripts/export_method_overview_figure.py`、`scripts/export_method_overview_figure.py` 与 entry point `adapromptset-export-method-overview-figure`；命令为 `D:\SAM\conda_envs\sam_gpu\python.exe scripts\export_method_overview_figure.py --project-root D:\SAM\adapromptset`。当前生成 `paper/figures/method_overview.tex`，`reports/method_overview_figure.json` 记录 status=`PASS_METHOD_OVERVIEW_FIGURE`、claim_boundary=`oracle_analysis_only`、figure SHA256=`E2D4CEB312CD4FC67FCD54B051D2FEC4397590D7E6F1991E6C124AA0C280E6D6`；该图只服务当前 oracle-analysis skeleton，不代表完整 adaptive-policy 方法图已验收。
- [x] TikZ action bank illustration。
  - 2026-07-18：新增 `src/adapromptset/scripts/export_action_bank_figure.py`、`scripts/export_action_bank_figure.py` 与 entry point `adapromptset-export-action-bank-figure`；命令为 `D:\SAM\conda_envs\sam_gpu\python.exe scripts\export_action_bank_figure.py --project-root D:\SAM\adapromptset`。当前从真实 `configs/actions/prompt_set_bank_v2.json` 生成 `paper/figures/action_bank.tex`，`reports/action_bank_figure.json` 记录 status=`PASS_ACTION_BANK_FIGURE`、claim_boundary=`oracle_analysis_only_current_mvp_subset`、source bank hash=`35E44B3B937A9D6AA9E969CDAFC9AC6F45A77E5BA1EF72BC6770E4B5F16D75B8`、figure SHA256=`290AB62437347460BCB8F2D5BB85C21A4539B245180E67955C0A18FE3155F862`。该图明确把 KMeans、density、GF-SAM-style graph families 标为当前 MVP subset 中 deferred/missing，不代表完整 action bank 已完成。
- [ ] Dice–calls Pareto plot。
  - 2026-07-18：当前 oracle-analysis skeleton 已接入 `paper/figures/oracle_pareto.png`，展示 matched-budget best-fixed 与 oracle Dice 以及 Dice oracle gap；`reports/oracle_analysis_figures.json` 记录 status=`PASS_ORACLE_ANALYSIS_FIGURES`、source_rows: budget_pareto=4、action_summary=12。完整论文级 Dice–calls Pareto（含 policy/CI/正式测试）仍未完成，因此本项不勾选。
- [ ] Dice–runtime Pareto plot。
- [ ] qualitative success cases。
- [ ] qualitative failure cases。
- [ ] behavior plots。
- [ ] 所有图字体在双栏 PDF 中可读。

### 文本

- [x] Abstract 不含未验证主张。
  - 2026-07-17：claim-free skeleton abstract 已生成，明确当前不是 adaptive-policy result；完整论文仍未验收。
  - 2026-07-18：新增 `src/adapromptset/scripts/audit_paper_sections.py` 与 `scripts/audit_paper_sections.py`；`reports/paper_section_audit.json` 记录 `abstract_claim_free=true`、status=`PASS`，仅关闭当前 oracle-analysis skeleton 的 section-boundary 审计，不代表完整 8 页论文验收。
- [x] Introduction 准确区分四篇相似工作。
  - 2026-07-17：skeleton introduction 已限制为当前 Gate F pivot 口径；四篇相似工作的完整差异表述仍未完成。
  - 2026-07-18：`reports/paper_section_audit.json` 记录 `related_work_boundaries_present=true`；`paper/sections/02_related.tex` 明确 GF-SAM、SynPo、AlignSAM、Point Prompt Defender 边界，并声明不主张 first-to-learn prompts。
- [x] Method 不声称 end-to-end differentiable。
  - 2026-07-17：skeleton method 描述 frozen DINOv2-L + SAM-B 与 predeclared actions，未声称 end-to-end differentiable。
  - 2026-07-18：`reports/paper_section_audit.json` 记录 `method_no_end_to_end_differentiable=true`；当前 method section 只描述 frozen DINOv2-L/SAM-B 与 predeclared actions。
- [x] Experiments 明确 patient/query 统计单位。
  - 2026-07-17：skeleton experiments 只列当前数据角色；patient/query 统计单位的完整论文表述仍未完成。
  - 2026-07-18：`paper/sections/04_experiments.tex` 已写明 query/case 与可用 patient cluster 作为统计单位，multi-reference episodes 是 repeated measurements、not independent samples；`reports/paper_section_audit.json` 记录 `experiments_statistical_units_present=true`。
- [x] Results 同时报告成本和准确率。
  - 2026-07-17：skeleton results 输入 oracle pivot/action summary 表；完整成本-准确率结果表仍未完成。
  - 2026-07-18：`paper/sections/05_results.tex` 已写明 Dice accuracy 与 cost descriptors 同报，包含 SAM calls、prompt counts、runtime 与 GPU-hour ledger status；`reports/paper_section_audit.json` 记录 `results_cost_and_accuracy_present=true`。完整正式结果表仍未完成。
- [x] Limitations 包含 action bank 上界、reference quality、冻结 matcher 错误不可恢复、utility generation cost。
  - 2026-07-17：skeleton limitations 已写 MVP subset、GlaS episode 不足、无 policy/transfer/efficiency/SOTA claim；完整四项 limitation 仍未覆盖。
  - 2026-07-18：`paper/sections/06_limitations.tex` 已覆盖 action bank upper bound、reference quality sensitivity、frozen matcher errors cannot be recovered、utility generation cost；`reports/paper_section_audit.json` 对四项 limitation 均记录 true。
- [x] Conclusion 与真实证据一致。
  - 2026-07-17：skeleton conclusion 限定为 oracle-analysis-only report；完整论文结论仍未验收。
  - 2026-07-18：`reports/paper_section_audit.json` 记录 `conclusion_oracle_analysis_only=true`；当前 conclusion 与 `PIVOT_ORACLE_ANALYSIS` / oracle-analysis-only 证据一致，不主张 adaptive policy success。

### [GATE-O] 模拟数据清除

- [ ] 所有 `SIMULATED PLACEHOLDER` 已移除。
  - 2026-07-17：生成的 `paper/**/*.tex` 与 `.bib` 未包含 simulated placeholder；仓库级清理仍未最终验收。
- [x] `\simulatedfalse`。
  - 2026-07-17：重新运行 `D:\SAM\conda_envs\sam_gpu\python.exe scripts\build_paper.py --project-root D:\SAM\adapromptset --mode submission --compile` 后，`paper/main.tex` 第 13 行写入 `\simulatedfalse`，且 paper 目录无 `simulatedtrue` / `DRAFT WATERMARK` 命中。
- [x] submission build guard 通过。
  - 2026-07-17：`D:\SAM\conda_envs\sam_gpu\python.exe scripts\build_paper.py --project-root D:\SAM\adapromptset --mode submission --compile` 通过；guard 未发现 `SIMULATED PLACEHOLDER`/`simulatedtrue`，并生成 `paper/main.pdf`。
  - 2026-07-17：`D:\SAM\conda_envs\sam_gpu\python.exe scripts\build_paper.py --project-root D:\SAM\adapromptset --mode submission` 通过。
- [ ] 搜索仓库不存在模拟数值残留。

---

## Phase 16 — LaTeX/PDF 验收

- [ ] `latexmk` 可从干净环境编译。
  - 2026-07-17：`latexmk` 尚未作为独立验收工具执行；当前通过的是 `scripts/build_paper.py --mode submission --compile` 内部的 `pdflatex` 路径。第一次 `pdflatex` 因非必要 `hyperref` 依赖加载卡住；移除该依赖并压缩自动导出的双栏表后，`pdflatex` 成功生成 `paper/main.pdf`。
- [x] 无 undefined citation。
  - 2026-07-17：`rg -n "LaTeX Warning|undefined citation|Citation .* undefined" paper\main.log` 未发现 citation warning；当前 skeleton 无正式 bibliography 引用。
- [x] 无 undefined reference。
  - 2026-07-17：第二轮/脚本编译后的 `paper/main.log` 未发现 undefined reference 或 rerun warning。
- [x] 无 missing figure。
  - 2026-07-17：当前 claim-free skeleton 未引用外部 figure，`paper/main.log` 未发现 missing file/figure 错误。
- [x] 无 overfull hbox > 5 pt。
  - 2026-07-17：`paper\tables\table_oracle_pivot.tex` 与 `table_action_summary.tex` 已改为 `table*` 并使用 `\small`；`rg -n "Overfull" paper\main.log` 无命中。
  - 2026-07-17：新增 `paper\tables\table_representation_pivot.tex` 后再次运行 `scripts\build_paper.py --mode submission --compile`；`rg -n "Overfull|undefined|Citation .* undefined|LaTeX Warning: Label\(s\) may have changed|Rerun" paper\main.log` 无命中。
- [ ] 主文约 8 页双栏。
- [ ] 参考文献完整。
- [ ] 表格字号可读。
- [ ] 公式编号和引用正确。
- [x] Figure 1 清楚区分 frozen/learned/train-only paths。
  - 2026-07-18：`paper/figures/method_overview.tex` 与 `paper/sections/03_method.tex` 明确区分 frozen matcher DINOv2-L、frozen segmenter SAM-B、train/validation-only utility labels、Gate-F-passing 前才允许的 learned utility predictor，以及 test-time safe path；`reports/paper_section_audit.json` 新增并通过 `method_figure_frozen_learned_train_only=true`。
- [x] Oracle 和 privileged rows 的 caption 无歧义。
  - 2026-07-18：`scripts\export_oracle_analysis_tables.py` 生成的 oracle pivot/action summary/representation 表 caption 已明确区分 oracle analysis upper bounds、post-hoc oracle labels、not deployable method outputs/selections，并声明当前 skeleton 不包含 privileged-input row；`reports/paper_section_audit.json` 记录 `oracle_privileged_caption_boundaries=true`、status=`PASS`、violations=[]。
- [x] PDF metadata、title、authors 占位符正确。
  - 2026-07-18：新增 `src/adapromptset/reporting/pdf_metadata.py`、`src/adapromptset/scripts/audit_pdf_metadata.py` 与 `scripts/audit_pdf_metadata.py`；`paper/main.tex` 通过 pdfTeX 原生 `\pdfinfo` 写入 Title、Author、Subject、Keywords。`reports/pdf_metadata_audit.json` 记录 status=`PASS`，`pdf_title_matches=true`、`pdf_author_matches=true`、`pdf_subject_matches=true`、`pdf_keywords_match=true`，当前 author 占位符为 `Internal MVP Draft`。
- [x] `paper/main.pdf` 保存到最终位置。
  - 2026-07-17：`paper/main.pdf` 已生成，当前为 2 页 claim-free oracle-analysis skeleton，文件大小 86345 bytes，SHA256=`D422B8FC545EE85E18D5BBB07AAF1CD7DF6ACAEB5BD7C03810DE915297FE8378`；这不代表最终 8 页完整论文已完成。
  - 2026-07-17：接入 representation-pivot 表后重新生成 `paper/main.pdf`，当前仍为 2 页 claim-free oracle-analysis skeleton，文件大小 95591 bytes，SHA256=`D1809296B349DC56983E841CC0587CFEF15CE7E39460277C261DE625BFE427B6`；这不代表最终 8 页完整论文已完成。
  - 2026-07-17：接入 Gate-P claim audit 后重新编译 `paper/main.pdf`，当前仍为 2 页 claim-free oracle-analysis skeleton，文件大小 95612 bytes，SHA256=`4C6A283FD8BF00C94FF7AD9A899067CCE29C3C82BD20926C14FDFA8478207AFD`；这不代表最终 8 页完整论文已完成。
  - 2026-07-18：接入 AJI oracle-gap 记录与报告链路后重新编译 `paper/main.pdf`，当前仍为 2 页 claim-free oracle-analysis skeleton，文件大小 128866 bytes，SHA256=`F59371410A05C310B563A4A75610A7E16730E482BA6FEFF689CA051DE68F8BB3`；这仍不代表最终 8 页完整论文已完成。
  - 2026-07-18：接入 oracle-analysis figures 后重新编译 `paper/main.pdf`，当前仍为 2 页 claim-free oracle-analysis skeleton，文件大小 224396 bytes，SHA256=`1B34DE77B0B7314D27A782D7C6C1BBFA66E88FD208DC6819451AF38EF59582DB`；`paper/main.log` 经 `rg -n "Overfull|undefined|Citation .* undefined|LaTeX Warning: Label\(s\) may have changed|Rerun|File .* not found|Missing"` 扫描无命中。这仍不代表最终 8 页完整论文已完成。
  - 2026-07-18：接入 TikZ method overview Figure 1 后重新编译 `paper/main.pdf`，当前为 3 页 claim-free oracle-analysis skeleton，文件大小 247474 bytes，SHA256=`661FA24807E7C60E3DAD5B9B4502591B50EBF79D2C9618662CB5814F46B86EA9`；`paper/main.log` 经同一 warning/undefined/overfull/rerun 扫描无命中。这仍不代表最终 8 页完整论文已完成。
  - 2026-07-18：接入 TikZ action-bank Figure 2 后重新编译 `paper/main.pdf`，当前为 4 页 claim-free oracle-analysis skeleton，文件大小 291717 bytes，SHA256=`26BFB8F8F5FB30113DC40EDFAC7462AFCCC107D8D4E57C891D9D4075CF8DDBDC`；`paper/main.log` 经同一 warning/undefined/overfull/rerun 扫描无命中。这仍不代表最终 8 页完整论文已完成。
  - 2026-07-18：接入 oracle/privileged caption boundary audit 后重新编译 `paper/main.pdf`，当前仍为 4 页 claim-free oracle-analysis skeleton，文件大小 292020 bytes，SHA256=`A332ED507B81D886D27A3B4E5D9674440EF505AB9526425BEE5FD775EF06C34B`；`reports/final_project_report.md` 记录 Oracle/privileged captions=`PASS`。这仍不代表最终 8 页完整论文已完成。
  - 2026-07-18：接入 PDF metadata audit 后重新编译 `paper/main.pdf`，当前仍为 4 页 claim-free oracle-analysis skeleton，文件大小 292317 bytes，SHA256=`B6E8E628BA04CA64E8A1E6E45BE3A2C749E88C9030A832C8A77EE0EC365CB9C1`；`reports/final_project_report.md` 记录 PDF metadata=`PASS`。这仍不代表最终 8 页完整论文已完成。

### [GATE-P] 最终科学审查

- [x] 生成机器可复现的科学主张边界审计。
  - 2026-07-17：新增 `src/adapromptset/scripts/audit_scientific_claims.py`、`scripts/audit_scientific_claims.py` 与 entry point `adapromptset-audit-scientific-claims`；命令为 `D:\SAM\conda_envs\sam_gpu\python.exe scripts\audit_scientific_claims.py --project-root D:\SAM\adapromptset`。
  - 2026-07-17：已生成 `reports/scientific_claim_audit.json` 与 `reports/scientific_claim_audit.md`，当前 status=`PASS`、claim boundary=`oracle_analysis_only`、Gate F status=`FAIL_ORACLE_GAP_SMALL`、violations=[]。
  - 2026-07-17：审计检查 README、final report、oracle-analysis paper assets、`paper/main.tex` 与 `paper/sections/*.tex`，并要求 Gate F pivot、MoNuSAC policy-training 禁止、post-hoc MoNuSAC action redesign 禁止、test-time action search 禁止、GlaS insufficient episodes、bootstrap insufficient clusters 均有 JSON artifact 支撑。该项是 Gate-P 的机器审计，不代表 Final Sign-off。
- [x] 每个主张都能指向表格/图或统计结果。
  - 2026-07-17：`reports/scientific_claim_audit.json` 当前 status=`PASS`、violations=[]；required evidence 覆盖 Gate F pivot、oracle-analysis-only claim boundary、representation pivot、GlaS insufficient episodes、bootstrap insufficient clusters 与 PDF hash。该结论只适用于当前 claim-free oracle-analysis skeleton，不代表完整 8 页论文。
- [x] 没有将 in-domain retraining 写成 transfer。
  - 2026-07-17：`audit_scientific_claims.py` 新增正向边界检查 `no_in_domain_transfer_claim`；`reports/final_project_report.md` 明确 `No in-domain retraining or transfer claim is made.`，当前 PASS。
- [x] 没有将 background reference 写成 explicit SAM negative point。
  - 2026-07-17：`audit_scientific_claims.py` 新增正向边界检查 `background_reference_not_explicit_negative`；`reports/final_project_report.md` 明确 `Background reference evidence is not explicit SAM negative point prompting.`，当前 PASS。
- [x] 没有将 oracle 当成可部署方法。
  - 2026-07-17：`audit_scientific_claims.py` 新增正向边界检查 `oracle_not_deployable`；`reports/final_project_report.md` 明确 `Oracle rows are analysis upper bounds, not deployable method outputs.`，当前 PASS。
- [x] 没有把多 reference episodes 当独立样本。
  - 2026-07-17：`audit_scientific_claims.py` 新增正向边界检查 `multi_reference_not_independent`；`reports/final_project_report.md` 明确 `Multi-reference episodes are repeated measurements and not independent samples.`，当前 PASS。
- [x] 没有 test-time action search 或 test threshold tuning。
  - 2026-07-17：`audit_scientific_claims.py` 继续要求 `test_time_search_prohibited=true`，并扫描 `test-time action search` / `test time action search` 等禁止短语；当前 JSON status=`PASS`、violations=[]。
- [x] 失败结果和限制未被隐藏。
  - 2026-07-17：`audit_scientific_claims.py` 新增正向边界检查 `failures_and_limitations_disclosed`；`reports/final_project_report.md` 明确 `Failure results and limitations are disclosed.`，当前 PASS。

---

## Final Handoff / Reproducibility Report

- [x] 生成当前 pivot 状态下的项目报告和复现命令。
  - 2026-07-17：新增 `src/adapromptset/scripts/export_final_report.py`、`scripts/export_final_report.py` 与 entry point `adapromptset-export-final-report`；命令为 `D:\SAM\conda_envs\sam_gpu\python.exe scripts\export_final_report.py --project-root D:\SAM\adapromptset`。
  - 2026-07-17：已生成 `reports/final_project_report.json` 与 `reports/final_project_report.md`。报告从真实 Phase 5、Gate F、fixed-oracle、representation、GlaS smoke、bootstrap、run metadata、compute ledger 与 PDF artifact 读取；当前 status=`PIVOT_ORACLE_ANALYSIS`、claim boundary=`oracle_analysis_only`、GlaS=`PENDING_INSUFFICIENT_EPISODES`、bootstrap=`PENDING_INSUFFICIENT_CLUSTERS`。
  - 2026-07-17：报告明确禁止 MoNuSAC policy training、post-hoc MoNuSAC action redesign 与 test-time action search；该报告是 handoff/reproducibility 资产，不代表 Final Sign-off 或 8 页完整论文完成。
  - 2026-07-17：复现命令列表已追加 `scripts\audit_scientific_claims.py --project-root D:\SAM\adapromptset`，确保最终交付前重新执行 Gate-P claim-boundary audit。
  - 2026-07-17：复现命令列表已追加 `scripts\audit_reference_exposure.py --project-root D:\SAM\adapromptset`，且 final report JSON 记录 reference exposure status=`PASS` 与 `reference_test_leakage_zero=true`。
  - 2026-07-17：复现命令列表已追加 `scripts\audit_data_profile.py --project-root D:\SAM\adapromptset`，且 final report JSON 记录 data profile status=`PASS`、row_count=526。
  - 2026-07-17：复现命令列表已追加 `scripts\audit_gate_f_failure_diagnostics.py --project-root D:\SAM\adapromptset`，且 final report JSON 记录 Gate F failure diagnostics：coverage=`LOW_COMPONENT_COVERAGE`、proposal=`ACTIONS_SEPARATED`、SAM sensitivity=`NO_HIGHER_CALL_GAIN`、policy_training_allowed=false。
  - 2026-07-17：复现命令列表已追加 `scripts\audit_ddr_protocol.py --project-root D:\SAM\adapromptset`，且 final report JSON 记录 DDR protocol status=`PASS_PROTOCOL_DEFINED`、stage1 mask rule=`binary_union_of_EX_HE_MA_SE`、stage2 intents=`EX,HE,MA,SE`、formal_policy_training_allowed=false。
  - 2026-07-18：复现命令列表已追加 `scripts\audit_new_domain_pilot_readiness.py --project-root D:\SAM\adapromptset`，且 final report JSON 记录 new-domain readiness：status=`READY_WITH_BLOCKED_FORMAL_DATASETS`、GlaS=`BLOCKED_INSUFFICIENT_EPISODES`、DDR=`READY_FOR_INFRASTRUCTURE_ONLY`、IDRID=`READ_ONLY_NON_TUNABLE`、CPM-17=`DEFERRED_MISSING_OR_INCOMPLETE`。
  - 2026-07-18：复现命令列表已追加 `scripts\export_oracle_analysis_figures.py --project-root D:\SAM\adapromptset`，且 final report JSON 记录 oracle-analysis figures：status=`PASS_ORACLE_ANALYSIS_FIGURES`、claim_boundary=`oracle_analysis_only`、figures=2、PDF SHA256=`1B34DE77B0B7314D27A782D7C6C1BBFA66E88FD208DC6819451AF38EF59582DB`。
  - 2026-07-18：复现命令列表已追加 `scripts\export_method_overview_figure.py --project-root D:\SAM\adapromptset`，且 final report JSON 记录 method overview figure：status=`PASS_METHOD_OVERVIEW_FIGURE`、claim_boundary=`oracle_analysis_only`、figure SHA256=`E2D4CEB312CD4FC67FCD54B051D2FEC4397590D7E6F1991E6C124AA0C280E6D6`、PDF SHA256=`661FA24807E7C60E3DAD5B9B4502591B50EBF79D2C9618662CB5814F46B86EA9`。
  - 2026-07-18：复现命令列表已追加 `scripts\export_action_bank_figure.py --project-root D:\SAM\adapromptset`，且 final report JSON 记录 action-bank figure：status=`PASS_ACTION_BANK_FIGURE`、source bank=`prompt_set_bank_v2`、figure SHA256=`290AB62437347460BCB8F2D5BB85C21A4539B245180E67955C0A18FE3155F862`、deferred families=`kmeans_groups,density_groups,gf_sam_style_graph_groups`、PDF SHA256=`26BFB8F8F5FB30113DC40EDFAC7462AFCCC107D8D4E57C891D9D4075CF8DDBDC`。
  - 2026-07-18：复现命令列表已追加 `scripts\audit_pdf_metadata.py --project-root D:\SAM\adapromptset`，且 final report JSON 记录 `pdf_metadata_audit`：status=`PASS`、expected title=`AdaPromptSet: Cost-Aware Prompt Set Selection for Reference-Guided Dense Medical Segmentation`、author=`Internal MVP Draft`、PDF SHA256=`B6E8E628BA04CA64E8A1E6E45BE3A2C749E88C9030A832C8A77EE0EC365CB9C1`。

# Final Sign-off

- [ ] Code lead sign-off：代码、测试、config、hash 完整。
- [ ] Data lead sign-off：许可、split、leakage 审计完整。
- [ ] Experiment lead sign-off：protocol lock 后无 test 调参。
- [ ] Statistics sign-off：bootstrap、paired tests、multiple comparison 正确。
- [ ] Paper lead sign-off：8 页双栏 PDF、图表、引用和结论一致。
- [ ] Principal investigator sign-off：同意最终主张和失败/限制披露。

## 最终状态

- [ ] **READY FOR INTERNAL REVIEW**
- [ ] **READY FOR SUBMISSION PREPARATION**
- [ ] **PIVOT REQUIRED**
- [ ] **PROJECT STOPPED — INSUFFICIENT ORACLE/POLICY EVIDENCE**

在最终状态下方填写：

```text
Date:
Git commit:
Action bank hash:
Data manifest hash:
Frozen protocol hash:
Main result run IDs:
Paper PDF SHA256:
Decision summary:
```


## Phase X — Action-Space Adequacy / Search-Oracle Audit (2026-07-20)

- [x] Read all historical action-bank and oracle artifacts before preregistration.
  - 2026-07-20: Read `configs/actions/prompt_set_bank_v1.json`, `configs/actions/prompt_set_bank_v2.json`, `reports/oracle_gap_pilot_v2/*`, `reports/fixed_oracle_analysis/*`, `reports/decision_checkpoint/*`, `reports/gates/gate_f_oracle_gap.*`, `reports/candidate_pilot/candidate_pilot_candidates.csv`.
- [x] Create historical Oracle correction doc.
  - 2026-07-20: `docs/historical_oracle_correction.md` and `reports/phaseX_historical_bank_audit/historical_claim_correction.md` rename the old claim to `FINITE_BANK_ORACLE_GAP_SMALL`.
- [x] Export old 12-action exact definitions.
  - 2026-07-20: `reports/phaseX_historical_bank_audit/action_bank_exact_definitions.{csv,md}`.
- [x] Export each historical action's absolute Dice / IoU / AJI / cost.
  - 2026-07-20: `reports/phaseX_historical_bank_audit/fixed_action_absolute_metrics.csv` and `fixed_action_rankings.csv`.
- [x] Pinpoint historical strongest fixed exact config.
  - 2026-07-20: `B_top5_pos3neg` (5 positives + 3 negatives in one group, 1 SAM call, MVP fusion). Mean Dice = 0.371238.
- [x] Freeze candidate scaling / dense prompt scaling / search oracle / saturation protocols.
  - 2026-07-20: `docs/dense_prompt_scaling_protocol.md`, `docs/search_oracle_protocol.md`, `docs/action_space_saturation_protocol.md`, `docs/formal_test_lock_notice.md`, `docs/action_space_problem_statement.md`, `docs/phaseX_goal_read_receipt.md`.
- [x] Freeze K / negative / grouping / call / point / search grids.
  - 2026-07-20: Recorded in `reports/phaseX_protocol_snapshot.json` and `reports/phaseX_input_hashes.json`.
- [x] Generate config/input hashes.
  - 2026-07-20: `reports/phaseX_input_hashes.json` records SHA256 of directive, v1/v2 banks, oracle summaries, decision checkpoint, gate F report, action utility CSV, fixed action summary CSV, budget Pareto CSV, oracle heatmap CSV, MoNuSAC validation episodes CSV, candidate pilot CSV.
- [ ] Run focused tests on the preregistration builder.
- [ ] Create preregistration git commit `phaseX: preregister action-space adequacy and search-oracle audit` before any new SAM inference.
- [ ] Run candidate coverage scaling (Stage S1 / Gate X2).
- [ ] Run single-variable K+ scaling.
- [ ] Run grouping scaling.
- [ ] Run negative scaling.
- [ ] Run coarse joint grid + successive halving.
- [ ] Build nested action banks A_small ⊂ A_medium ⊂ A_large ⊂ A_search_augmented.
- [ ] Compute fixed baseline saturation.
- [ ] Compute finite-bank oracle saturation.
- [ ] Run GT-guided greedy/beam search oracle.
- [ ] Compare bank oracle vs search oracle under matched budgets.
- [ ] Generate Phase X tables, figures, and `reports/phaseX_final_report.{md,json}`.
- [ ] Output the unique final Phase X state (one of the section 24 conclusions).

Current Phase X status: `PREREGISTRATION_PENDING_COMMIT`.

