# Stage 3 Feature-level Instance Matching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add DINO/Matcher-style feature-level instance similarity to Stage 3 proposal filtering.

**Architecture:** A standalone script reads Stage 1 metadata, Stage 2 proposal masks, and Stage 2 cluster points. It loads the same DINOv2 model used by `export_matcher_stage1_points.py`, computes a reference object prototype from the reference box/mask, computes a proposal prototype from each target mask, scores cosine similarity, then filters/NMS proposals with this feature score plus the existing mask-level constraints.

**Tech Stack:** Python, `torch`, `numpy`, `Pillow`, local DINOv2 under `Matcher`, local `assets/checkpoints/dinov2_vitl14_pretrain.pth`.

---

### Task 1: Static Check First

**Files:**
- Create: `code/benchmarks/check_stage3_feature_matching.py`
- Expected future target: `code/benchmarks/feature_match_stage3_proposals.py`

- [ ] **Step 1: Write failing static check**

Require:

```text
--proposals-dir
--stage1-dir
--dinov2-weights
--min-instance-similarity
--nms-iou
feature_scores.csv
feature_scores.json
feature_retained_overlay.png
feature_retained_grid.png
load_stage1_metadata
compute_reference_prototype
compute_proposal_prototype
compute_instance_similarity
filter_feature_candidates
run_feature_nms
save_feature_outputs
```

- [ ] **Step 2: Run check and verify failure**

Run:

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\check_stage3_feature_matching.py
```

Expected: FAIL because target script is missing.

### Task 2: Implement Feature-level Matching

**Files:**
- Create: `code/benchmarks/feature_match_stage3_proposals.py`

- [ ] **Step 1: Load Stage 1 and proposal inputs**

Read:

```text
runs/research_direction/stage1_matching/elephant_pair/points.json
runs/research_direction/stage2_proposals/elephant_pair/proposal_scores.json
runs/research_direction/stage2_sampling/elephant_pair/prompt_groups.json
```

- [ ] **Step 2: Compute reference prototype**

Use Stage 1 `reference_image`, `reference_box` or `reference_mask`, DINOv2 features, and patch-pooled mask weights.

- [ ] **Step 3: Compute proposal prototype**

For each SAM proposal mask:

```text
resize mask to 518
pool to DINO patch grid
weighted average target DINO features inside mask
normalize proposal prototype
instance_similarity = cosine(reference_proto, proposal_proto)
```

- [ ] **Step 4: Filter and NMS**

Default thresholds:

```text
min_instance_similarity = 0.45
min_score = 0.80
min_area_pct = 5
max_area_pct = 45
max_bbox_area_pct = 45
min_cluster_coverage = 0.30
nms_iou = 0.75
```

Rank by feature-aware score:

```text
feature_filter_score = instance_similarity
                     + 0.1 * sam_score
                     + 0.1 * cluster_coverage
                     + 0.05 * prompt_coverage
                     - penalties
```

### Task 3: Run Smoke and Write Report

**Files:**
- Create: `docs/matcher_stage3_feature_matching_report.md`
- Modify: `docs/research_direction_priority_plan.md`

- [ ] **Step 1: Run smoke**

Run:

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\feature_match_stage3_proposals.py --proposals-dir runs\research_direction\stage2_proposals\elephant_pair --stage1-dir runs\research_direction\stage1_matching\elephant_pair --dinov2-weights assets\checkpoints\dinov2_vitl14_pretrain.pth --min-instance-similarity 0.45 --min-score 0.80 --min-area-pct 5 --max-area-pct 45 --max-bbox-area-pct 45 --min-cluster-coverage 0.30 --nms-iou 0.75 --out-dir runs\research_direction\stage3_feature_matching\elephant_pair
```

- [ ] **Step 2: Write Chinese report**

Report how feature-level similarity changes Stage 3 interpretation and whether retained `K_pred` is stable.

- [ ] **Step 3: Final verification**

Run:

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\check_stage3_feature_matching.py
Test-Path docs\matcher_stage3_feature_matching_report.md
Test-Path runs\research_direction\stage3_feature_matching\elephant_pair\feature_scores.csv
Test-Path runs\research_direction\stage3_feature_matching\elephant_pair\feature_retained_overlay.png
```
