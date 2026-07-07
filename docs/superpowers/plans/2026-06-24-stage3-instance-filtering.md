# Stage 3 Instance Filtering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Filter Stage 2 SAM mask proposals into retained instance masks and estimate `K_pred`.

**Architecture:** A standalone benchmark script reads `proposal_scores.json`, `proposal_masks/`, and Stage 2 cluster points. It computes proposal-level metrics, applies interpretable filters for SAM score, mask area, bbox area, cluster point coverage, and prompt point coverage, then performs mask NMS per cluster. The output is retained masks, a filtered overlay/grid, a score table, and `K_pred`.

**Tech Stack:** Python standard library, `numpy`, `Pillow`, local Stage 2 proposal artifacts.

---

### Task 1: Static Check First

**Files:**
- Create: `code/benchmarks/check_stage3_filtering.py`
- Expected future target: `code/benchmarks/filter_stage3_proposals.py`

- [ ] **Step 1: Write failing static check**

The check must require:

```text
--proposals-dir
--min-score
--min-area-pct
--max-area-pct
--max-bbox-area-pct
--min-cluster-coverage
--nms-iou
filtered_scores.csv
filtered_scores.json
retained_overlay.png
retained_grid.png
load_proposals
load_cluster_points
compute_proposal_metrics
filter_candidates
mask_iou
run_nms
save_filter_outputs
```

- [ ] **Step 2: Run check and verify failure**

Run:

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\check_stage3_filtering.py
```

Expected: FAIL because `filter_stage3_proposals.py` does not exist.

### Task 2: Implement Minimal Filtering

**Files:**
- Create: `code/benchmarks/filter_stage3_proposals.py`

- [ ] **Step 1: Load inputs**

Read:

```text
runs/research_direction/stage2_proposals/elephant_pair/proposal_scores.json
runs/research_direction/stage2_sampling/elephant_pair/prompt_groups.json
runs/research_direction/stage2_proposals/elephant_pair/proposal_masks/*.png
```

- [ ] **Step 2: Compute metrics**

For each proposal:

```text
sam_score
mask_area_pct
bbox_area_pct
cluster_coverage = cluster points inside mask / total cluster points
prompt_coverage = prompt points inside mask / prompt points
filter_score = sam_score + 0.2 * cluster_coverage + 0.1 * prompt_coverage - area_penalty
```

- [ ] **Step 3: Filter candidates**

Default thresholds:

```text
min_score = 0.80
min_area_pct = 5.0
max_area_pct = 45.0
max_bbox_area_pct = 45.0
min_cluster_coverage = 0.30
nms_iou = 0.75
```

- [ ] **Step 4: Run per-cluster NMS**

Sort by `filter_score` descending and keep non-overlapping masks. Estimate:

```text
K_pred = retained proposal count
```

For this single elephant case, success means `K_pred=1` or a small set where NMS leaves one best proposal.

### Task 3: Run Smoke and Write Report

**Files:**
- Create: `docs/matcher_stage3_filtering_report.md`
- Modify: `docs/research_direction_priority_plan.md`

- [ ] **Step 1: Run smoke**

Run:

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\filter_stage3_proposals.py --proposals-dir runs\research_direction\stage2_proposals\elephant_pair --min-score 0.80 --min-area-pct 5 --max-area-pct 45 --max-bbox-area-pct 45 --min-cluster-coverage 0.30 --nms-iou 0.75 --out-dir runs\research_direction\stage3_filtering\elephant_pair
```

- [ ] **Step 2: Write report**

Report sections:

1. 实验目的；
2. 输入和可复现命令；
3. 过滤指标；
4. before/after proposal count；
5. retained masks and `K_pred`；
6. 对老师 Stage 3 threshold / K 估计问题的回答；
7. 下一步 feature-level matching。

- [ ] **Step 3: Final verification**

Run:

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\check_stage3_filtering.py
Test-Path docs\matcher_stage3_filtering_report.md
Test-Path runs\research_direction\stage3_filtering\elephant_pair\filtered_scores.csv
Test-Path runs\research_direction\stage3_filtering\elephant_pair\retained_overlay.png
```
