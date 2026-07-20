# Spatial Layout Prompt Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic spatially spread prompt action and validate its train-only multi-reference utility policy.

**Architecture:** Extend the candidate-pool builder with a layout mode while preserving current Top-K behavior. Add layout metadata to fixed actions and policy features, then run a bounded MoNuSAC training/validation feasibility experiment without reading test masks.

**Tech Stack:** Python, NumPy, PyTorch frozen DINOv2/SAM, scikit-learn ExtraTrees, pytest.

## Global Constraints

- Work only under `D:\SAM\FSPL`; preserve prior artifacts.
- Use deterministic ranking and integer patch-grid coordinates.
- Do not use MoNuSAC test masks or metrics in this feasibility pass.
- Run pytest with `-p no:cacheprovider`.

---

### Task 1: Add deterministic spread candidate selection

**Files:**
- Modify: `fspl/candidates.py`
- Modify: `tests/test_candidates.py`

**Interfaces:**
- Produces: `build_candidate_pools(..., positive_layout="topk" | "spread", min_positive_distance_patches=3)`.

- [ ] Write a failing test where adjacent top-scoring patches are excluded by `positive_layout="spread"` but retained by `"topk"`.
- [ ] Implement deterministic grid-distance suppression with ranked fallback when there are too few separated candidates.
- [ ] Run `python -B -m pytest tests/test_candidates.py -q -p no:cacheprovider`.

### Task 2: Add action metadata and layout-aware policy features

**Files:**
- Modify: `fspl/protocol.py`, `fspl/policy.py`, `tests/test_policy.py`

**Interfaces:**
- Produces: `FixedRuleConfig(..., positive_layout, min_positive_distance_patches)` and policy features `positive_layout_is_spread`, `positive_nearest_distance_patches`.

- [ ] Write failing tests for action metadata and finite layout features.
- [ ] Implement default-compatible action fields and the two new features.
- [ ] Run the focused protocol/policy tests.

### Task 3: Evaluate layout actions in bounded multi-reference feasibility

**Files:**
- Modify: `fspl/stage4.py`, `scripts/run_stage4_multireference_feasibility.py`
- Create: `docs/2026-07-16-stage5-spatial-layout-feasibility-plan.md`

**Interfaces:**
- Consumes layout-aware fixed rules and writes `case_action_utilities.csv` plus `summary.json` under a new Stage 5 artifact directory.

- [ ] Write a test verifying layout metadata is serialized in a stage action row.
- [ ] Propagate layout fields into candidate construction and CSV feature records.
- [ ] Run the full test suite, then execute `3 references × (12 train + 6 validation queries) × 2 actions` on CPU.
- [ ] Verify row counts, role disjointness, finite utilities, and paired policy-vs-`top5_n3` validation evidence.
