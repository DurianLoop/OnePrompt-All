# Docs Classification and Reading Order Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `D:\SAM\docs` 根目录下零散文档按用途归类，并提供清晰的中文阅读顺序。

**Architecture:** 保留 `docs/workspace_index.md` 作为根入口；新增 `docs/README.md` 作为中文阅读地图。将研究主线放入 `docs/research/`，阶段实验报告放入 `docs/reports/`，工程入口和清理说明放入 `docs/ops/`，UI/图示放入 `docs/ui/`。已有 `archive/`、`papers*/`、`workspace/`、`superpowers/`、`setup_legacy/` 目录保持原结构。

**Tech Stack:** Markdown 文档、PowerShell 文件移动、`rg`/`Test-Path` 验证。

---

### Task 1: Move Root Documents Into Categories

**Files:**
- Move research docs to `docs/research/`
- Move matcher/SAM3 reports to `docs/reports/`
- Move operational docs to `docs/ops/`
- Move UI docs to `docs/ui/`

- [x] **Step 1: Create directories**

Run:

```powershell
New-Item -ItemType Directory -Force -Path docs\research,docs\reports\sam3,docs\reports\matcher,docs\ops,docs\ui
```

- [x] **Step 2: Move files**

Move only root-level loose files; do not move existing directory trees.

### Task 2: Add Reading Order

**Files:**
- Create: `docs/README.md`
- Modify: `docs/workspace_index.md`

- [x] **Step 1: Create Chinese reading map**

`docs/README.md` should answer:

```text
先看什么、后看什么、每个目录是什么、想复现实验去哪看、想看论文去哪看。
```

- [x] **Step 2: Update workspace index**

Point current working documents to their new paths.

### Task 3: Update Moved Index Links

**Files:**
- Modify: `docs/research/teacher_direction_closed_loop_summary.md`
- Modify: `docs/research/research_direction_artifact_index.md`

- [x] **Step 1: Replace moved report paths**

Update references from old root paths to:

```text
docs/reports/sam3/
docs/reports/matcher/
docs/research/
docs/ops/
docs/ui/
```

### Task 4: Verify

**Files:**
- Read/check only.

- [x] **Step 1: Check key files**

Run:

```powershell
Test-Path docs\README.md
Test-Path docs\research\teacher_direction_closed_loop_summary.md
Test-Path docs\reports\matcher\matcher_stage3_feature_matching_report.md
Test-Path docs\ops\code_entrypoints.md
```

- [x] **Step 2: Check old loose root files are gone except root indexes**

Run:

```powershell
Get-ChildItem docs -File | Select-Object Name
```

Expected root-level files:

```text
README.md
workspace_index.md
```
