# Cross-Model Matcher Generalization Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify whether the same reference-guided foreground-background matcher improves multiple feature-backbone and SAM-family segmenter combinations on the fixed official MoNuSeg screening protocol.

**Architecture:** Keep one support/query manifest and one dense-point candidate grid. First run a controlled DINOv2-L cross-segmenter sweep, then combine it with existing DINOv2-B results. Evaluate each model against its own prompt baselines and report absolute metrics plus relative improvement, without selecting settings on the test set for a final paper claim.

**Tech Stack:** Python, PyTorch CPU environment, DINOv2, Segment Anything, SAM2.1, SAM-Med2D, CSV/HTML reports.

## Global Constraints

- Dataset: existing official MoNuSeg screening manifest, one train support and fourteen test queries.
- Point protocol: `K={20,30,40,50,75,100}`, `lambda={0,0.25,0.5,0.75,1,1.25,1.5,2}` and existing layout methods.
- Do not tune an individual model on test cases and then describe the result as a final held-out result.
- MedSAM remains a box-prompt comparator; it is not a failed point-prompt method.
- Report both mean Dice and the difference to each model's matched prompt baseline.

---

### Task 1: Validate inputs and runnable checkpoints

**Files:**
- Read: `runs/research_direction/monuseg_official_screening/official_monuseg_screening_manifest.csv`
- Read: `assets/checkpoints/`
- Read: `code/benchmarks/run_monuseg_dense_prompt_experiment.py`

- [ ] Verify the manifest has fourteen query cases and one fixed support case.
- [ ] Verify DINOv2-L and SAM ViT-L, ViT-H, SAM2.1 Hiera Large checkpoints exist before launching a long run.
- [ ] Run a one-case smoke prediction for every missing combination; require an output `summary.json` and one case row.

### Task 2: Run controlled DINOv2-L cross-segmenter screen

**Files:**
- Write: `runs/research_direction/monuseg_official_screening/dinov2l_samvitl_kxlambda_test14/`
- Write: `runs/research_direction/monuseg_official_screening/dinov2l_samvith_kxlambda_test14/`
- Write: `runs/research_direction/monuseg_official_screening/dinov2l_sam21large_kxlambda_test14/`

- [ ] Run the exact shared K/lambda grid for SAM ViT-L.
- [ ] Run the exact shared K/lambda grid for SAM ViT-H.
- [ ] Run the exact shared K/lambda grid for SAM2.1 Hiera Large.
- [ ] Preserve candidates, per-case CSVs, reports and summary JSON for every run.

### Task 3: Create a cross-model evidence table

**Files:**
- Create: `runs/research_direction/monuseg_official_screening/cross_model_generalization_summary.csv`
- Modify: `runs/research_direction/monuseg_official_screening/model_weight_screening_summary.csv`

- [ ] Collect DINOv2-B and DINOv2-L results under the same metrics.
- [ ] Join each segmenter with its valid non-reference prompt baseline.
- [ ] Add `delta_dice_vs_baseline`, `relative_dice_gain_pct`, candidate count and protocol label.
- [ ] Flag comparisons that are not prompt-compatible instead of producing a misleading difference.

### Task 4: Update the research protocol and progress checklist

**Files:**
- Modify: `docs/track/monuseg_sam123_medsam_paper_experiment_design.md`
- Modify: `docs/track/checklist.md`

- [ ] Replace the single-model winner narrative with a cross-model generalization claim to be tested.
- [ ] Add the two-level protocol: fixed cross-model policy and model-adaptive policy.
- [ ] Record completed combinations, failures, and the remaining non-training scope.

### Task 5: Verify and report honestly

**Files:**
- Read: all Task 2 `summary.json` and generated CSVs.

- [ ] Confirm each complete run has fourteen cases and its expected grid rows.
- [ ] Confirm all summary values are derived from the saved experiment outputs.
- [ ] State whether the evidence supports universal improvement, partial generalization, or model-dependent response; do not claim universality without positive comparisons.
