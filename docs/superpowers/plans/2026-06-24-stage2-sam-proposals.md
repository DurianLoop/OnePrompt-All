# Stage 2 SAM Proposal Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert Stage 2 sampled point prompt groups into real SAM mask proposals and save reproducible proposal artifacts.

**Architecture:** A standalone benchmark script reads `prompt_groups.json`, loads the local SAM ViT-B checkpoint through `segment_anything.SamPredictor`, runs each prompt group as foreground point prompts, chooses the highest-scoring mask per group, and writes masks, overlays, scores, JSON, grid visualization, and a Chinese report. This uses SAM as the stable local visual-prompt backend; SAM3-native proposal generation can be swapped in later once its Windows path is equally stable.

**Tech Stack:** Python, `torch`, `numpy`, `Pillow`, local `code/segment_anything`, `assets/checkpoints/sam_vit_b.pth`.

---

### Task 1: Static Check First

**Files:**
- Create: `code/benchmarks/check_stage2_proposals.py`
- Expected future target: `code/benchmarks/generate_stage2_sam_proposals.py`

- [ ] **Step 1: Write failing static check**

The check must require:

```text
--stage2-dir
--checkpoint
--model-type
--device
--max-groups
proposal_scores.csv
proposal_scores.json
proposal_grid.png
proposal_masks
proposal_overlays
load_prompt_groups
load_sam_predictor
predict_prompt_group
save_proposal_outputs
```

- [ ] **Step 2: Run check and verify failure**

Run:

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\check_stage2_proposals.py
```

Expected: FAIL because `generate_stage2_sam_proposals.py` does not exist.

### Task 2: Implement SAM Proposal Generator

**Files:**
- Create: `code/benchmarks/generate_stage2_sam_proposals.py`

- [ ] **Step 1: Implement loaders**

Read `runs/research_direction/stage2_sampling/elephant_pair/prompt_groups.json`, load `target_image`, and parse each prompt group into Nx2 foreground point coordinates.

- [ ] **Step 2: Implement SAM predictor**

Load:

```text
assets/checkpoints/sam_vit_b.pth
```

with:

```python
from segment_anything import SamPredictor, sam_model_registry
sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
sam.to(device=device)
sam.eval()
predictor = SamPredictor(sam)
```

- [ ] **Step 3: Predict masks**

For each prompt group:

```python
masks, scores, _ = predictor.predict(
    point_coords=np.asarray(points, dtype=np.float32),
    point_labels=np.ones(len(points), dtype=np.int32),
    multimask_output=True,
)
best = int(np.argmax(scores))
```

Save the selected mask and metrics.

- [ ] **Step 4: Save artifacts**

Write:

```text
proposal_masks/group_###.png
proposal_overlays/group_###.png
proposal_scores.csv
proposal_scores.json
proposal_grid.png
summary.md
```

Metrics must include group id, cluster id, sample size, point count, SAM score, mask area percent, and bbox.

### Task 3: Run Smoke and Write Report

**Files:**
- Create: `docs/matcher_stage2_sam_proposals_report.md`
- Modify: `docs/research_direction_priority_plan.md`

- [ ] **Step 1: Run smoke**

Run:

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\generate_stage2_sam_proposals.py --stage2-dir runs\research_direction\stage2_sampling\elephant_pair --checkpoint assets\checkpoints\sam_vit_b.pth --model-type vit_b --device auto --max-groups 12 --out-dir runs\research_direction\stage2_proposals\elephant_pair
```

Expected: 12 proposal masks and overlays.

- [ ] **Step 2: Write report**

Report sections:

1. 实验目的；
2. 输入和命令；
3. 真实 SAM proposal 生成方式；
4. proposal score/area 结果；
5. 对老师“重复 m 次得到 m 个分割”的回答；
6. 下一步 Stage 3 filtering。

- [ ] **Step 3: Final verification**

Run:

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\check_stage2_proposals.py
Test-Path docs\matcher_stage2_sam_proposals_report.md
Test-Path runs\research_direction\stage2_proposals\elephant_pair\proposal_scores.csv
Test-Path runs\research_direction\stage2_proposals\elephant_pair\proposal_grid.png
```
