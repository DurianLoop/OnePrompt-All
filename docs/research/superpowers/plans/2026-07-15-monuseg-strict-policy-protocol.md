# MoNuSeg Strict Policy Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one reproducible, case-level MoNuSeg protocol that supplies leakage-free train, validation, and held-out test episodes for reference-conditioned prompt-policy research.

**Architecture:** The protocol starts from the audited 37-train/14-test local official-download manifest. It deterministically stratifies the observed training pool into train and validation cases, chooses the reference only from the train cases, and writes one manifest per query role. Candidate generation and all future learning consume these manifests without moving individual proposals across roles.

**Tech Stack:** Python standard library, CSV, JSON, pytest.

## Global Constraints

- Preserve the 14 downloaded test cases as held-out queries; do not select hyperparameters from them.
- A query case and every proposal generated from it must belong to exactly one role.
- The reference image and its manual mask must always come from the training role.
- Report the observed local package as 37 train + 14 test; do not silently relabel it as a 30/14 protocol.

---

### Task 1: Build deterministic case-level assignments and episode manifests

**Files:**
- Create: `code/benchmarks/build_monuseg_strict_policy_splits.py`
- Test: `code/benchmarks/test_build_monuseg_strict_policy_splits.py`

**Interfaces:**
- Consumes: rows with `split`, `sample_id`, `image_path`, `mask_path`, `mask_area_ratio`, and `nuclei_region_count`.
- Produces: `build_protocol(rows, val_count) -> (assignments, episodes_by_role, metadata)`.

- [x] Write a failing test with six train and two test cases.
- [x] Verify that validation stays inside the train pool, test stays held out, and a training reference is never queried against itself.
- [x] Implement deterministic area-stratified validation assignment and median-density training-reference selection.
- [x] Run the focused pytest module.

### Task 2: Materialize the protocol for the audited MoNuSeg manifest

**Files:**
- Modify: `code/benchmarks/build_monuseg_strict_policy_splits.py`
- Modify: `docs/track/checklist_pub.md`

- [x] Add a CLI that reads `runs/research_direction/dataset_audit/monuseg_official_manifest.csv`.
- [x] Write case assignments, train/val/test episode CSV files, and metadata JSON under `runs/research_direction/prompt_policy_splits/monuseg_strict_official/`.
- [x] Run the CLI with eight validation cases and validate counts from the generated artifacts.
- [x] Record the materialized protocol and its exact counts in the paper checklist.

### Task 3: Prepare the train-only candidate-oracle run manifest

**Files:**
- Modify: `code/benchmarks/run_monuseg_dense_prompt_experiment.py`
- Test: `code/benchmarks/test_dense_prompt_manifest_filter.py`

- [ ] Add role-aware manifest filtering without changing the default behavior.
- [ ] Generate candidate-oracle artifacts separately for training, validation, and test roles.
- [ ] Verify that no test candidate artifact is read by the training command.
