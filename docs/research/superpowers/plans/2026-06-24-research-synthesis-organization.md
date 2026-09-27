# Research Synthesis Organization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把当前 SAM3/Matcher 教师方向的零散实验报告、脚本入口和运行产物整理成可交付的中文总文档。

**Architecture:** 不改动模型逻辑和实验输出，只新增总览文档与产物索引，并从 `docs/workspace_index.md` 挂入口。总报告按“老师问题 -> 已完成验证 -> 证据路径 -> 当前结论 -> 下一步优先级”组织，避免读者在多个阶段报告之间来回跳。

**Tech Stack:** Markdown 文档、现有 `code/benchmarks/` 脚本、现有 `runs/research_direction/` 实验产物。

---

### Task 1: 生成中文总报告

**Files:**
- Create: `docs/teacher_direction_closed_loop_summary.md`

- [x] **Step 1: 汇总已有阶段报告**

读取并归纳以下文件的关键信息：

```text
docs/sam3_teacher_minimal_validation_report.md
docs/matcher_stage1_export_report.md
docs/matcher_stage1_feedback_report.md
docs/matcher_stage2_sampling_report.md
docs/matcher_stage2_sam_proposals_report.md
docs/matcher_stage3_filtering_report.md
docs/matcher_stage3_feature_matching_report.md
docs/matcher_multi_instance_horse_validation_report.md
docs/research_direction_priority_plan.md
```

- [x] **Step 2: 写入总报告**

总报告必须包含：

```text
1. 当前一句话主线
2. 老师问题逐条正面回应
3. 已完成最小闭环
4. 可复现实验命令与产物路径
5. 当前不能过度宣称的点
6. 下一步优先级研究方案
```

### Task 2: 生成实验产物索引

**Files:**
- Create: `docs/research_direction_artifact_index.md`

- [x] **Step 1: 写入脚本入口表**

列出当前主线脚本：

```text
code/benchmarks/sam3_teacher_minimal_validation.py
code/benchmarks/export_matcher_stage1_points.py
code/benchmarks/apply_stage1_feedback.py
code/benchmarks/sample_stage2_prompts.py
code/benchmarks/generate_stage2_sam_proposals.py
code/benchmarks/filter_stage3_proposals.py
code/benchmarks/feature_match_stage3_proposals.py
```

- [x] **Step 2: 写入运行产物表**

列出 `runs/teacher_minimal_validation/` 和 `runs/research_direction/` 下的关键结果目录。

### Task 3: 更新工作区索引

**Files:**
- Modify: `docs/workspace_index.md`

- [x] **Step 1: 将总报告和产物索引加入 Current Working Documents**

新增两行：

```markdown
| `docs/teacher_direction_closed_loop_summary.md` | Current Chinese synthesis: teacher questions, experiments, conclusions, and next research priorities. |
| `docs/research_direction_artifact_index.md` | Reproducible artifact index for scripts, commands, and output directories. |
```

### Task 4: 验证

**Files:**
- Read/check only.

- [x] **Step 1: 检查新增文档存在**

Run:

```powershell
Test-Path docs\teacher_direction_closed_loop_summary.md
Test-Path docs\research_direction_artifact_index.md
```

Expected:

```text
True
True
```

- [x] **Step 2: 检查关键脚本和关键产物存在**

Run:

```powershell
Test-Path code\benchmarks\feature_match_stage3_proposals.py
Test-Path runs\research_direction\multi_instance_horse_head\stage3_feature_matching\feature_scores.csv
```

Expected:

```text
True
True
```
