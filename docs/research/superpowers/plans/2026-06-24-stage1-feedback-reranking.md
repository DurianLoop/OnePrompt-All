# Stage 1 Feedback Reranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal feedback reranking loop for Matcher Stage 1 points using positive and negative points.

**Architecture:** Extend the Stage 1 exporter to save raw similarity as `similarity.npy`. Add a standalone reranking script that loads `points.json` and `similarity.npy`, applies Gaussian positive/negative feedback in the target coordinate space, exports reranked points, and saves before/after overlays and heatmaps.

**Tech Stack:** Python, NumPy, PIL, existing Stage 1 exporter outputs.

---

### Task 1: Extend Stage 1 Exporter

**Files:**
- Modify: `code/benchmarks/export_matcher_stage1_points.py`
- Modify: `code/benchmarks/check_stage1_exporter.py`

- [ ] **Step 1: Save raw similarity grid**

Save `similarity.npy` beside `points.csv`.

- [ ] **Step 2: Add static check token**

Require `similarity.npy` in `check_stage1_exporter.py`.

### Task 2: Add Feedback Reranker

**Files:**
- Create: `code/benchmarks/apply_stage1_feedback.py`
- Create: `code/benchmarks/check_stage1_feedback.py`

- [ ] **Step 1: Write static check first**

The check should fail until `apply_stage1_feedback.py` exists and includes required tokens.

- [ ] **Step 2: Implement reranking CLI**

Arguments:

```text
--stage1-dir
--positive-points
--negative-points
--sigma
--positive-weight
--negative-weight
--top-k
--threshold
--out-dir
```

- [ ] **Step 3: Save feedback artifacts**

Output:

```text
feedback_points.csv
feedback_points.json
feedback_overlay_before_after.png
feedback_heatmap_before_after.png
summary.md
```

### Task 3: Run Elephant Feedback Smoke Case

**Files:**
- Read: `runs/research_direction/stage1_matching/elephant_pair/`
- Output: `runs/research_direction/stage1_feedback/elephant_pair/`

- [ ] **Step 1: Rerun Stage 1 export with raw similarity**

```powershell
cd D:\SAM
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\export_matcher_stage1_points.py --reference-image code\gradio_demo\images\elephant1.png --target-image code\gradio_demo\images\elephant2.png --reference-box 0,60,295,425 --top-k 100 --threshold 0.55 --out-dir runs\research_direction\stage1_matching\elephant_pair
```

- [ ] **Step 2: Run feedback**

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\apply_stage1_feedback.py --stage1-dir runs\research_direction\stage1_matching\elephant_pair --positive-points 268,331 --negative-points 30,430;455,420 --sigma 42 --positive-weight 0.18 --negative-weight 0.22 --top-k 100 --threshold 0.55 --out-dir runs\research_direction\stage1_feedback\elephant_pair
```

Expected: exports feedback files and reports before/after selected counts.

### Task 4: Docs and Verification

**Files:**
- Create: `docs/matcher_stage1_feedback_report.md`
- Modify: `docs/research_direction_priority_plan.md`

- [ ] **Step 1: Write report**

Summarize selected feedback points, before/after top points, artifacts, and interpretation.

- [ ] **Step 2: Run checks**

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\check_stage1_exporter.py
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\check_stage1_feedback.py
```
