# SAM3 Demo Cleanup Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the SAM workspace easier to navigate, fix SAM3 point prompt accumulation in the web demo, and capture a minimal experiment path for the teacher's hypothesis.

**Architecture:** Keep the current monolithic demo operational while adding only local UI state controls and a static regression check. Treat code cleanup as documentation-first to avoid deleting useful research artifacts.

**Tech Stack:** FastAPI, inline HTML/JavaScript in `code/demos/medical_sam_click_app.py`, Python static checks, Markdown reports.

---

### Task 1: Add a Regression Check for Prompt Controls

**Files:**
- Create: `code/benchmarks/check_demo_prompt_controls.py`

- [x] **Step 1: Write the failing check**

Create a script that reads `medical_sam_click_app.py` and asserts that point run mode, point scope, manual SAM3 defaults, and latest-only request trimming exist.

- [x] **Step 2: Run check to verify it fails**

Run:

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\check_demo_prompt_controls.py
```

Expected before implementation: missing controls and wiring are reported.

### Task 2: Add Prompt Accumulation Settings to the Demo

**Files:**
- Modify: `code/demos/medical_sam_click_app.py`

- [x] **Step 1: Add controls**

Add `Point Run Mode` and `Point Scope` under click prompt controls.

- [x] **Step 2: Wire controls**

Add `pointRunModeInput` and `pointScopeInput`, update disabled states, prompt summaries, drawer help, `runClick()`, `currentImagePromptPayload()`, and click handling.

- [x] **Step 3: Set SAM3 defaults**

When backend is SAM3, set manual cumulative prompts. For other backends, use auto cumulative prompts.

### Task 3: Verify and Document

**Files:**
- Create: `docs/sam_workspace_cleanup_experiment_plan.md`

- [x] **Step 1: Run regression check**

Run:

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\check_demo_prompt_controls.py
```

Expected: passed.

- [x] **Step 2: Run syntax check without pycache writes**

Run:

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B -c "from pathlib import Path; compile(Path('code/demos/medical_sam_click_app.py').read_text(encoding='utf-8'), 'code/demos/medical_sam_click_app.py', 'exec'); compile(Path('code/benchmarks/check_demo_prompt_controls.py').read_text(encoding='utf-8'), 'code/benchmarks/check_demo_prompt_controls.py', 'exec'); print('compile checks passed')"
```

Expected: `compile checks passed`.

- [x] **Step 3: Run minimal SAM3 smoke**

Run a small natural-image smoke on `code/gradio_demo/images/elephant1.png` using SAM3 point and text prompts. Record the output in the cleanup/experiment document.

