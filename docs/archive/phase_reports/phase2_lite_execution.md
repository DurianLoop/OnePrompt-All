# Phase2-lite Execution Plan

Phase2-lite is the small, fast benchmark lane before the full Phase 2 benchmark.
It fixes the data and prompt protocol first, then compares SAM-family models under
the same image and prompt conditions.

## Scope

- Datasets: IDRID + MonuSeg.
- Case count: 40 cases total, 20 from IDRID and 20 from MonuSeg.
- Prompt protocols per case:
  - `gt_box`: tight bounding box from ground-truth mask.
  - `perturbed_box`: deterministic expanded and shifted GT box.
  - `centroid_point`: foreground centroid positive point.
  - `grid_point`: up to four positive foreground points sampled from a 2x2 grid.
  - `auto_box`: weak image-derived box, with deterministic fallback when the
    image heuristic is unusable.
- First model set:
  - `sam_vit_b`
  - `medsam_vit_b`
  - `sam2_tiny`

## Outputs

All generated artifacts are under `runs/phase2_lite/`.

- `manifests/phase2_lite_40_cases.csv`: case list and GT metadata.
- `prompts/phase2_lite_prompts.jsonl`: prompt records, five per case.
- `predictions/<model>/<case>/<protocol>.png`: predicted binary masks.
- `metrics/phase2_lite_metrics.csv`: Dice, IoU, precision, recall, runtime, status.
- `README.md`: run summary and mean Dice by model.

## Commands

Build the manifest and prompt set only:

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\phase2_lite.py build
```

Run a two-case smoke test:

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\phase2_lite.py run --limit-cases 2 --max-side 512
```

Run the full 40-case phase2-lite benchmark:

```powershell
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\phase2_lite.py all --max-side 768
```

MedSAM is box-first. Its point-prompt rows are intentionally recorded as
`unsupported` rather than silently skipped.
