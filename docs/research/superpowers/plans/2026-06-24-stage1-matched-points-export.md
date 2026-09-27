# Stage 1 Matched Points Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small reproducible Stage 1 exporter that shows Matcher/DINO top matched points before SAM mask generation.

**Architecture:** Add one standalone benchmark script that loads DINOv2, accepts reference/target images and a reference mask or box, computes prototype-to-target patch similarity, and writes CSV/JSON/overlay/heatmap artifacts. Keep upstream `Matcher/` and local `code/matcher/` behavior unchanged.

**Tech Stack:** Python, PyTorch, DINOv2 code already vendored under `Matcher/`, PIL, NumPy, matplotlib.

---

### Task 1: Add Stage 1 Exporter Script

**Files:**
- Create: `code/benchmarks/export_matcher_stage1_points.py`

- [ ] **Step 1: Create CLI and DINO loading path**

Implement arguments for `--reference-image`, `--target-image`, `--reference-box`, `--reference-mask`, `--dinov2-weights`, `--top-k`, `--threshold`, `--out-dir`, and `--device`.

- [ ] **Step 2: Implement reference mask creation**

If `--reference-mask` is provided, load it as grayscale and resize to `518x518`. Otherwise parse `--reference-box x0,y0,x1,y1` and create a binary mask.

- [ ] **Step 3: Compute Stage 1 similarity**

Resize reference and target to `518x518`, normalize with the same DINO transform as Matcher, extract patch features, average reference foreground patch features as prototype, compute cosine similarity against all target patches, and export top-k patches.

- [ ] **Step 4: Save artifacts**

Write:

```text
points.csv
points.json
topk_overlay.png
similarity_heatmap.png
summary.md
```

Each point row should include rank, patch index, x, y, similarity, and selected-by-threshold flag.

### Task 2: Run Minimal Smoke Case

**Files:**
- Read: `code/gradio_demo/images/elephant1.png`
- Read: `code/gradio_demo/images/elephant2.png`
- Output: `runs/research_direction/stage1_matching/elephant_pair/`

- [ ] **Step 1: Run exporter**

```powershell
cd D:\SAM
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\export_matcher_stage1_points.py --reference-image code\gradio_demo\images\elephant1.png --target-image code\gradio_demo\images\elephant2.png --reference-box 0,60,295,425 --top-k 100 --threshold 0.65 --out-dir runs\research_direction\stage1_matching\elephant_pair
```

- [ ] **Step 2: Verify artifacts exist**

```powershell
Test-Path runs\research_direction\stage1_matching\elephant_pair\points.csv
Test-Path runs\research_direction\stage1_matching\elephant_pair\topk_overlay.png
Test-Path runs\research_direction\stage1_matching\elephant_pair\similarity_heatmap.png
```

Expected output: all three paths are `True`.

### Task 3: Update Research Docs

**Files:**
- Modify: `docs/research_direction_priority_plan.md`
- Create: `docs/matcher_stage1_export_report.md`

- [ ] **Step 1: Write a short report**

Summarize the input images, reference box, top-k/threshold, artifact paths, and how this satisfies P0-1.

- [ ] **Step 2: Add the report to the research plan**

Add a short pointer under P0-1 with the report path and output directory.

### Task 4: Verification

**Files:**
- Read: `code/benchmarks/export_matcher_stage1_points.py`

- [ ] **Step 1: Compile check**

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B -c "from pathlib import Path; p=Path('code/benchmarks/export_matcher_stage1_points.py'); compile(p.read_text(encoding='utf-8'), str(p), 'exec'); print('stage1 exporter compiles')"
```

Expected output:

```text
stage1 exporter compiles
```
