# Feedback Learning Matcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the minimum Feedback- and Learning-Guided Matcher extension requested by the teacher.

**Architecture:** Add a pure core module for feedback prototype update and lightweight threshold learning, then expose it through a CLI and the existing Stage1-3 runner. Keep the implementation interpretable and independent from heavy DINO/SAM inference so it can be tested quickly.

**Tech Stack:** Python, NumPy, existing `code/benchmarks` runner scripts, JSON artifacts.

---

### Task 1: Core Feedback and Learning Module

**Files:**
- Create: `code/benchmarks/matcher_feedback_learning.py`
- Test: `code/benchmarks/check_matcher_feedback_learning.py`

- [x] **Step 1: Write the failing test**

Create checks for:

```text
positive/negative prototype feedback improves positive-vs-background margin
adaptive threshold keeps selected points inside budget
labeled proposal threshold separates positives from negatives
proposal score penalizes overly large background masks
```

- [x] **Step 2: Run test to verify it fails**

Run:

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\check_matcher_feedback_learning.py
```

Expected initial result:

```text
ModuleNotFoundError: No module named 'matcher_feedback_learning'
```

- [x] **Step 3: Implement minimal module**

Implement:

```text
FeedbackPoint
ThresholdLearningResult
apply_feature_feedback
select_threshold_by_budget
learn_scalar_threshold
score_stage3_proposal
```

- [x] **Step 4: Run test to verify it passes**

Expected result:

```text
Matcher feedback learning checks passed.
```

### Task 2: Learning Policy CLI

**Files:**
- Create: `code/benchmarks/run_matcher_feedback_learning_policy.py`
- Test: `code/benchmarks/check_matcher_feedback_learning_policy_cli.py`

- [x] **Step 1: Write the failing CLI test**

Use a synthetic `feature_scores.json` with four proposals and a `labels.json` with two positives and two negatives.

- [x] **Step 2: Run test to verify it fails**

Run:

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\check_matcher_feedback_learning_policy_cli.py
```

Expected initial result:

```text
can't open file '...run_matcher_feedback_learning_policy.py'
```

- [x] **Step 3: Implement CLI**

The CLI reads:

```text
--stage3-feature-json
--labels-json
--out-dir
```

and writes:

```text
matcher_learning_policy.json
summary.md
```

- [x] **Step 4: Run test to verify it passes**

Expected result:

```text
Matcher feedback learning policy CLI checks passed.
```

### Task 3: Runner Integration

**Files:**
- Modify: `code/benchmarks/run_research_direction_case.py`
- Modify: `code/benchmarks/check_research_direction_runner.py`

- [x] **Step 1: Extend runner static check first**

Require these tokens:

```text
learning_policy
--labels-json
run_matcher_feedback_learning_policy.py
```

- [x] **Step 2: Run test to verify it fails**

Expected initial result:

```text
Missing required tokens
```

- [x] **Step 3: Add optional runner step**

When `--labels-json` is provided, append:

```text
stage3_feature -> learning_policy
```

- [x] **Step 4: Run test to verify it passes**

Expected result:

```text
Research direction runner static checks passed.
```

### Task 4: Documentation

**Files:**
- Create: `docs/superpowers/specs/2026-07-01-feedback-learning-matcher-design.md`
- Create: `docs/superpowers/plans/2026-07-01-feedback-learning-matcher.md`
- Create: `docs/research/feedback_learning_matcher_extension.md`

- [x] **Step 1: Write Chinese design explanation**

Explain how feedback enters Matcher and how lightweight learning enters threshold/filtering.

- [x] **Step 2: Record reproducible commands**

Include all check commands and example CLI commands.

- [x] **Step 3: State limitations**

Clarify this is a minimum interpretable learning loop, not large-scale training.
