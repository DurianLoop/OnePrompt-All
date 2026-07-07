# Cleanup Todo

## Done

- Added SAM3 point prompt controls in the web demo:
  - `Point Run Mode`: `auto` or `manual`.
  - `Point Scope`: `cumulative` or `latest`.
  - SAM3 defaults to manual cumulative mode so multiple point prompts can be accumulated before running.
- Added `code/benchmarks/check_demo_prompt_controls.py` as a static regression guard for those UI controls.
- Collected an initial SAM3 natural-scene smoke check on `code/gradio_demo/images/elephant1.png`.
- Moved historical phase reports to `docs/archive/phase_reports/`.
- Moved historical prompt-selector notes to `docs/archive/prompt_selector/`.
- Added current workspace and code-entrypoint indexes.

## Keep Untouched Unless Explicitly Requested

- `assets/`: checkpoints and local assets.
- `runs/`: generated experiment outputs.
- `conda_envs/`: local environments.
- External baselines: `sam3/`, `sam2/`, `Matcher/`, `MedSAM/`, `SAM-Med2D/`, `Medical-SAM-Adapter/`, `nnUNet/`.

## Near-Term Cleanup

1. Split `code/demos/medical_sam_click_app.py` into state, backend, rendering, and web-route modules.
2. Add a small FastAPI route-level smoke test for `/segment` and the compare path.
3. Add a dedicated teacher-direction experiment script under `code/benchmarks/`:
   - natural-scene SAM3 point/text contrast;
   - sliding-window large-image protocol;
   - Matcher top-N point sampling;
   - stage 2 repeated SAM prompt sampling;
   - stage 3 instance-level threshold filtering.
4. Write experiment outputs to `runs/teacher_direction/`.
5. Update `docs/ops/sam_workspace_cleanup_experiment_plan.md` after each run.

## Research Directions To Try Next

| Direction | Why It Matters | Minimal Next Test |
| --- | --- | --- |
| SAM3 cumulative vs latest point prompts | Directly addresses the web-demo issue and teacher point-prompt claim. | Same image, same clicked points, compare cumulative/latest masks and IoU/area changes. |
| Positive + negative point feedback | Tests whether manual feedback improves target feature matching. | Select 1-2 correct points and 1-2 wrong points, rerun matching score ranking. |
| Sliding-window no-downsample inference | Handles images larger than DINO/SAM input constraints. | Tile a large image, merge masks, compare to downsample baseline. |
| Matcher top-N sampling | Tests the teacher's Stage 2 idea. | Sample top 5%, 10%, 20% matched points; run SAM repeatedly; compare mask diversity and stability. |
| Instance-level thresholding | Tests Stage 3 filtering. | Keep masks above similarity threshold and discard low-score masks; report instance count and retained area. |
