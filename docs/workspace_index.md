# SAM Workspace Index

This file is the current map for `D:\SAM`. For the Chinese reading order inside `docs/`, start with `docs/README.md`.

## Current Working Documents

| File | Purpose |
| --- | --- |
| `docs/README.md` | Chinese reading map: what to read first, how docs are categorized, and task-specific reading order. |
| `docs/research/teacher_direction_closed_loop_summary.md` | Current Chinese synthesis: teacher questions, experiments, conclusions, and next research priorities. |
| `docs/research/research_direction_artifact_index.md` | Reproducible artifact index for scripts, commands, and output directories. |
| `docs/research/research_direction_priority_plan.md` | Prioritized research plan for SAM3 + Matcher + medical promptable segmentation. |
| `docs/reports/sam3/sam3_teacher_minimal_validation_report.md` | Minimal closed-loop validation of SAM3 text/point teacher hypotheses. |
| `docs/reports/matcher/` | Matcher Stage 1/2/3 reports and the multi-instance horse validation. |
| `docs/ops/code_entrypoints.md` | Practical code entrypoints and canonical scripts. |
| `docs/ops/cleanup_todo.md` | Cleanup policy and remaining refactor plan. |
| `docs/ui/sam_workbench_ui_research_and_design.md` | UI/workbench design notes. |

## Docs Layout

| Path | Meaning |
| --- | --- |
| `docs/research/` | Current research synthesis, artifact index, teacher-direction report, and priority plan. |
| `docs/reports/sam3/` | SAM3 text/point validation reports. |
| `docs/reports/matcher/` | Matcher Stage 1/2/3 and multi-instance validation reports. |
| `docs/ops/` | Engineering entrypoints, cleanup plans, and workspace operation notes. |
| `docs/ui/` | UI/workbench design notes and diagrams. |
| `docs/workspace/` | Workspace layout and environment notes. |
| `docs/papers/`, `docs/papers2/`, `docs/papers3/` | Paper PDFs and extracted text. |
| `docs/archive/` | Historical phase reports and old prompt-selector notes. |
| `docs/superpowers/plans/` | Implementation and organization plans created during this work. |
| `docs/setup_legacy/` | Legacy dependency lists. |

## Top-Level Workspace Layout

| Path | Status | Notes |
| --- | --- | --- |
| `code/` | Main local implementation | Web demo, benchmarks, adapters, Matcher integration, workbench code. Prefer editing here. |
| `sam3/` | External/local SAM3 source | Treat as vendor-style dependency. Keep local compatibility patches small and documented. |
| `sam2/` | External/local SAM2 source | Treat as dependency. |
| `Matcher/` | Upstream Matcher reference clone | Keep as reference. Local OCSAM wrappers live under `code/`. |
| `MedSAM/`, `SAM-Med2D/`, `Medical-SAM-Adapter/`, `nnUNet/` | External baselines | Treat as imported baselines unless a task explicitly targets them. |
| `assets/` | Local weights/images | Large, not for commit. Includes checkpoints. |
| `runs/` | Generated experiments | Keep results, logs, and masks here. Do not commit by default. |
| `conda_envs/` | Local environments | Machine-local only. |
| `docs/` | Current docs and archived notes | Start with `docs/README.md`. |
| `scripts/` | Workspace helpers | Use for orchestration, setup, and one-off utilities that are not model code. |

## Edit Policy

- New demo behavior should go through `code/demos/medical_sam_click_app.py` until the web app is split into smaller modules.
- New teacher-direction experiments should go under `code/benchmarks/` and write outputs under `runs/`.
- New summary documents should be linked from `docs/README.md` and this file.
- Do not edit `assets/`, `runs/`, or `conda_envs/` for cleanup unless the user explicitly asks to delete or package artifacts.
- Treat external baseline directories as read-mostly; wrap them from `code/` when possible.
