# P0-P3 Research Closures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `docs/research/teacher_direction_closed_loop_summary.md` 中 P0-P3 的下一步研究方案推进成最小可复现闭环，并把结论追加到该文档底部。

**Architecture:** 复用现有 Stage 1/2/3 benchmark 脚本，新增少量面向缺口的脚本：feature/prototype feedback、统一 case runner、text failure taxonomy、sliding-window matching、medical data gate、adapter gate。所有新实验输出写入 `runs/research_direction/p0_p3_closure/`，避免覆盖已完成的 elephant/horse 主线结果。

**Tech Stack:** Python, DINOv2 local checkpoint, SAM ViT-B local checkpoint, Markdown reports, static check scripts.

---

### Task 1: P0 Feature Feedback and Runner

**Files:**
- Create: `code/benchmarks/apply_stage1_feature_feedback.py`
- Create: `code/benchmarks/check_stage1_feature_feedback.py`
- Create: `code/benchmarks/run_research_direction_case.py`
- Create: `code/benchmarks/check_research_direction_runner.py`

- [x] **Step 1: Write static checks first**

Checks must verify:

```text
apply_stage1_feature_feedback.py contains:
--positive-points
--negative-points
compute_feedback_prototype
feedback_points.json

run_research_direction_case.py contains:
stage1
feature_feedback
stage2_sampling
stage2_proposals
stage3_feature
```

- [x] **Step 2: Implement feature/prototype feedback**

Use DINO target patch features near positive/negative clicks to build feedback prototypes:

```text
updated_proto = normalize(reference_proto + positive_weight * positive_proto - negative_weight * negative_proto)
similarity_after = updated_proto @ target_features.T
```

- [x] **Step 3: Implement case runner**

The runner should execute Stage 1, optional feature feedback, Stage 2, proposals, and Stage 3 with one command.

### Task 2: P1 Text Failure Taxonomy

**Files:**
- Create: `code/benchmarks/build_sam3_text_failure_taxonomy.py`
- Create: `code/benchmarks/check_text_failure_taxonomy.py`

- [x] **Step 1: Write static check**

Check that the script writes `taxonomy.csv`, `taxonomy.json`, and has fallback trigger logic.

- [x] **Step 2: Implement taxonomy builder**

Use existing `runs/teacher_minimal_validation/results.json` when present, and add rule-based fallback labels:

```text
instances == 0 -> fallback_required
prompt type in action/attribute/referring/medical -> risk tag
area abnormal or many instances -> fallback_candidate
```

### Task 3: P1.5 Sliding-Window Matching

**Files:**
- Create: `code/benchmarks/export_sliding_window_stage1_points.py`
- Create: `code/benchmarks/check_sliding_window_stage1.py`

- [x] **Step 1: Write static check**

Check for tile generation, coordinate restoration, and merged point output.

- [x] **Step 2: Implement no-downsample Stage 1 tiling**

Tile large target image, resize each tile to DINO input, compute per-tile similarity, restore coordinates to original image, and merge top-k points.

### Task 4: P2 Medical Small-Data Gate

**Files:**
- Create: `code/benchmarks/medical_small_data_gate.py`
- Create: `code/benchmarks/check_medical_small_data_gate.py`

- [x] **Step 1: Write static check**

Check for data discovery, modality summary, and `medical_gate_report.json` output.

- [x] **Step 2: Implement medical data discovery**

Search local likely medical data locations for image/mask files and emit whether P2 can run real Dice/IoU now.

### Task 5: P3 Adapter Gate

**Files:**
- Create: `code/benchmarks/adapter_lora_gate.py`
- Create: `code/benchmarks/check_adapter_lora_gate.py`

- [x] **Step 1: Write static check**

Check for gate criteria and output `adapter_gate_report.json`.

- [x] **Step 2: Implement adapter readiness gate**

Read P0/P1/P2 summaries and decide:

```text
ready_for_training = true only if medical data with masks exists and P0/P1 evidence is present
otherwise recommend no training yet
```

### Task 6: Append Final Conclusions

**Files:**
- Modify: `docs/research/teacher_direction_closed_loop_summary.md`

- [x] **Step 1: Append P0-P3 closure section**

Add a dated section summarizing:

```text
P0 done / evidence
P1 done / evidence
P1.5 done / evidence
P2 data gate result
P3 adapter gate result
what can be claimed
what cannot be claimed
next exact action
```

### Task 7: Verification

**Files:**
- Read/check only.

- [x] **Step 1: Run static checks**

Run all new check scripts with `python -B`.

- [x] **Step 2: Compile new scripts**

Compile new scripts directly with UTF-8 reads.

- [x] **Step 3: Check artifacts and appended doc section**

Use `Test-Path` and `rg` to verify outputs and appended conclusions.
