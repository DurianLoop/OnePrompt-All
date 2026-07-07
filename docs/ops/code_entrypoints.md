# Code Entrypoints

This file records the current runnable paths after workspace cleanup.

## Primary Demo

| Task | Entrypoint | Notes |
| --- | --- | --- |
| Web promptable segmentation demo | `code/demos/medical_sam_click_app.py` | Main browser app. Supports SAM, MedSAM, SAM2, SAM3, Matcher-style flows, click/box/text prompts, and the new point accumulation setting. |
| Demo control regression | `code/benchmarks/check_demo_prompt_controls.py` | Static check for the SAM3 cumulative/latest point controls and compare path. |
| Teacher minimal validation | `code/benchmarks/sam3_teacher_minimal_validation.py` | Runs the small SAM3 point/text/feedback validation loop and saves CSV, JSON, and overlays. |

Run locally:

```powershell
cd D:\SAM\code
D:\SAM\conda_envs\sam_gpu\python.exe demos\medical_sam_click_app.py --host 127.0.0.1 --port 7860
```

Run the current lightweight check:

```powershell
cd D:\SAM
D:\SAM\conda_envs\sam_gpu\python.exe -B code\benchmarks\check_demo_prompt_controls.py
```

## Teacher-Direction Experiment Code

| Purpose | Entrypoint |
| --- | --- |
| Existing Phase 2 lite benchmark | `code/benchmarks/phase2_lite.py` |
| Analyze Phase 2 lite results | `code/benchmarks/analyze_phase2_lite.py` |
| Prompt selector V2/V3 evaluation | `code/benchmarks/evaluate_prompt_selector_v2.py`, `code/benchmarks/evaluate_prompt_selector_v3.py` |
| Phase 3 experiments | `code/benchmarks/phase3_*.py` |
| SAM3/Matcher report evidence | See `docs/ops/sam_workspace_cleanup_experiment_plan.md` |
| Teacher minimal validation | `code/benchmarks/sam3_teacher_minimal_validation.py` |

## Matcher-Related Code

| Path | Role |
| --- | --- |
| `Matcher/` | Upstream Matcher reference. Keep as source-of-truth reference. |
| `code/matcher/` | Local integration and wrappers. Prefer adding project-specific logic here. |
| `code/demos/matcher_oneshot_worker.py` | Worker used by the web demo for one-shot Matcher-style calls. |

## Code Areas To Treat As Generated Or Local Data

These paths are not the first place to edit source behavior:

- `code/demo_outputs/`
- `code/output/`
- `code/vis/`
- `code/debug_figure/`
- `code/.ipynb_checkpoints/`
- `code/__pycache__/`
- dataset folders under `code/IDRID`, `code/MonuSeg`, `code/ddr_datasets`, and `code/datasets`

## Next Refactor Target

`code/demos/medical_sam_click_app.py` is now functionally important but too large. The next safe refactor is to split it into:

- `code/demos/web_app.py`: FastAPI routes and static page assembly.
- `code/demos/prompt_state.py`: click/box/text state handling and cumulative/latest prompt policy.
- `code/demos/model_backends.py`: SAM, MedSAM, SAM2, SAM3, and Matcher backend loading.
- `code/demos/rendering.py`: mask overlay, visualization, and response encoding.

Do this only after preserving the current regression check and adding at least one route-level smoke test.
