"""Phase U WP1 Multi-Model Segmenter Sensitivity Audit.

Runs the reference-instability benchmark (WP1) on the MoNuSAC train split
with multiple SAM-family segmenters, using DINOv2-L as the fixed matcher.

Supported models:
  SAM1: sam_b, sam_l, sam_h, medsam_b
  SAM2: sam2_t, sam2_s, sam2_b+, sam2_l
  SAM3: NOT AVAILABLE (torch.jit.script requires .py source)

Pipeline per model:
  Phase 1 (CPU):  reference resampling (5 seeds, train split)
  Phase 2 (GPU):  DINOv2-L matching + SAM segmentation -> episode CSV
  M2 Training:    dual-model (instability + catastrophic) anti-correlation check

CLI:
  Smoke test:
    D:\SAM\conda_envs\sam_gpu\python.exe ^
      src/adapromptset/scripts/run_phaseU_wp1_segmenter_audit.py ^
      --models sam_b,sam2_t --smoke --device cuda

  Full SAM1 audit:
    D:\SAM\conda_envs\sam_gpu\python.exe ^
      src/adapromptset/scripts/run_phaseU_wp1_segmenter_audit.py ^
      --models sam_b,sam_l,sam_h,medsam_b --device cuda

  Full SAM2 audit:
    D:\SAM\conda_envs\sam_gpu\python.exe ^
      src/adapromptset/scripts/run_phaseU_wp1_segmenter_audit.py ^
      --models sam2_t,sam2_s,sam2_b+,sam2_l --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = Path(r"D:\SAM")
SCRIPTS_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch

# Reuse WP1 pipeline from Phase T
import run_phaseT_wp1_holdout as wp1
import run_phaseT_wp3_m2_selective as m2

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
CKPT = WORKSPACE_ROOT / "assets" / "checkpoints"

SAM1_MODELS: dict[str, dict[str, Any]] = {
    "sam_b": {
        "checkpoint": CKPT / "sam_vit_b.pth",
        "architecture": "vit_b",
        "family": "sam1",
        "description": "SAM-B (vit_b, 94M)",
    },
    "sam_l": {
        "checkpoint": CKPT / "sam_vit_l_0b3195.pth",
        "architecture": "vit_l",
        "family": "sam1",
        "description": "SAM-L (vit_l, 312M)",
    },
    "sam_h": {
        "checkpoint": CKPT / "sam_vit_h_4b8939.pth",
        "architecture": "vit_h",
        "family": "sam1",
        "description": "SAM-H (vit_h, 641M)",
    },
    "medsam_b": {
        "checkpoint": CKPT / "medsam_vit_b.pth",
        "architecture": "vit_b",
        "family": "sam1",
        "description": "MedSAM-B (vit_b, 94M, medical fine-tuned)",
    },
}

SAM2_MODELS: dict[str, dict[str, Any]] = {
    "sam2_t": {
        "checkpoint": CKPT / "sam2.1_hiera_tiny.pt",
        "config": "sam2.1/sam2.1_hiera_t",
        "family": "sam2",
        "description": "SAM2.1-Hiera-Tiny (39M)",
    },
    "sam2_s": {
        "checkpoint": CKPT / "sam2.1_hiera_small.pt",
        "config": "sam2.1/sam2.1_hiera_s",
        "family": "sam2",
        "description": "SAM2.1-Hiera-Small (46M)",
    },
    "sam2_b+": {
        "checkpoint": CKPT / "sam2.1_hiera_base_plus.pt",
        "config": "sam2.1/sam2.1_hiera_b+",
        "family": "sam2",
        "description": "SAM2.1-Hiera-Base+ (81M)",
    },
    "sam2_l": {
        "checkpoint": CKPT / "sam2.1_hiera_large.pt",
        "config": "sam2.1/sam2.1_hiera_l",
        "family": "sam2",
        "description": "SAM2.1-Hiera-Large (224M)",
    },
}

ALL_SAM = {**SAM1_MODELS, **SAM2_MODELS}

# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _load_sam1_predictor(device: torch.device, architecture: str, checkpoint: Path) -> Any:
    """Load a SAM1-family predictor."""
    wp1.configure_external_paths(WORKSPACE_ROOT)
    from segment_anything import SamPredictor, sam_model_registry

    model = sam_model_registry[architecture](checkpoint=str(checkpoint))
    model.to(device)
    model.eval()
    return SamPredictor(model)


def _load_sam2_predictor(device_str: str, config_name: str, checkpoint: Path) -> Any:
    """Load a SAM2 image predictor using Hydra compose + instantiate."""
    sys.path.insert(0, str(WORKSPACE_ROOT / "sam2"))
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # Clear any previous Hydra initialization (e.g., from DINOv2/SAM1)
    GlobalHydra.instance().clear()

    config_dir = WORKSPACE_ROOT / "sam2" / "sam2" / "configs"
    try:
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = compose(config_name=config_name)
            model = instantiate(cfg.model, _recursive_=True)
    finally:
        GlobalHydra.instance().clear()

    ckpt_data = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt_data["model"], strict=True)
    model.to(device_str)
    model.eval()
    return SAM2ImagePredictor(model)


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark_for_model(
    sam_key: str,
    out_dir: Path,
    device: torch.device,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run Phase 1 + Phase 2 + M2 for one model.

    Returns a dict with benchmark results.
    """
    sam_info = ALL_SAM[sam_key]
    family = sam_info["family"]
    model_out = out_dir / sam_key
    model_out.mkdir(parents=True, exist_ok=True)

    seeds = list(range(2 if smoke else 5))
    max_queries = 2 if smoke else None

    # Monkey-patch WP1 constants
    _orig_sam_ckpt = wp1.SAM_CHECKPOINT
    _orig_dinov2_ckpt = wp1.DINOV2_CHECKPOINT
    _orig_query_manifest = wp1.HOLDOUT_QUERY_MANIFEST
    _orig_manifest_dir = wp1.PHASET_MANIFEST_DIR
    _orig_seed_count = wp1.SEED_COUNT
    _orig_load_sam = wp1.load_sam_predictor

    wp1.DINOV2_CHECKPOINT = CKPT / "dinov2_vitl14_pretrain.pth"
    train_manifest = PROJECT_ROOT / "manifests" / "phaseY" / "policy_train_queries.csv"
    wp1.HOLDOUT_QUERY_MANIFEST = train_manifest
    wp1.PHASET_MANIFEST_DIR = model_out / "manifests"
    wp1.PHASET_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    wp1.SEED_COUNT = len(seeds)

    try:
        # --- Phase 1: reference resampling ---
        print(f"  [{sam_key}] Phase 1: reference resampling ({len(seeds)} seeds)")
        wp1.run_phase1(seeds=seeds, max_queries=max_queries)

        # --- Phase 2: DINOv2 matching + SAM segmentation ---
        print(f"  [{sam_key}] Phase 2: DINOv2-L + {sam_info['description']}")
        t0 = time.perf_counter()

        # Load model-specific predictor
        if family == "sam1":
            wp1.SAM_CHECKPOINT = sam_info["checkpoint"]
            predictor = _load_sam1_predictor(device, sam_info["architecture"], sam_info["checkpoint"])
            wp1.load_sam_predictor = lambda d: predictor
        elif family == "sam2":
            wp1.SAM_CHECKPOINT = sam_info["checkpoint"]
            sam2_predictor = _load_sam2_predictor(str(device), sam_info["config"], sam_info["checkpoint"])
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else __import__("contextlib").nullcontext()
            wp1.load_sam_predictor = lambda d: _Sam2PredictorWrapper(sam2_predictor, autocast_ctx)
        else:
            raise ValueError(f"Unknown family: {family}")

        feature_cache = wp1.FeatureCache(device)
        predictor = wp1.load_sam_predictor(device)

        wp1.run_phase2(
            feature_cache=feature_cache,
            predictor=predictor,
            out_dir=model_out,
            seeds=range(len(seeds)),
            max_queries=max_queries,
        )

        elapsed = time.perf_counter() - t0
        episode_csv = model_out / "phase2_episode_instability.csv"
        if episode_csv.is_file():
            with open(str(episode_csv), encoding="utf-8") as f:
                n_rows = sum(1 for _ in f) - 1  # minus header
        else:
            n_rows = 0
        print(f"  [{sam_key}] Phase 2 done in {elapsed:.0f}s, {n_rows} episode rows")

        # --- M2 Training ---
        train_rows = m2.load_episodes(episode_csv)
        if len(train_rows) < 50:
            result = {
                "sam_key": sam_key, "family": family,
                "n_rows": len(train_rows), "status": "INSUFFICIENT_DATA",
                "phase2_elapsed_s": elapsed,
            }
        else:
            train_labels = m2.construct_labels(train_rows)
            val_labels = m2.construct_val_labels(train_rows)

            models = m2.build_all_models()
            # Train catastrophic model
            _, cat_model, cat_result = m2.evaluate_all_models(
                models, train_rows, train_rows, train_labels, val_labels,
                train_label_key="catastrophic",
            )
            # Train instability model
            _, inst_model, inst_result = m2.evaluate_all_models(
                models, train_rows, train_rows, train_labels, val_labels,
                train_label_key="instability",
            )

            cat_auroc_inst = float(cat_result["auroc_instability"])
            inst_auroc_inst = float(inst_result["auroc_instability"])
            cat_auprc_cat = float(cat_result["auprc_catastrophic"])
            inst_auprc_cat = float(inst_result["auprc_catastrophic"])
            prevalence_cat = float(np.mean(val_labels["catastrophic"]))
            prevalence_inst = float(np.mean(val_labels["instability"]))

            anti_corr = (cat_auroc_inst < 0.60) and (inst_auprc_cat < prevalence_cat)

            result = {
                "sam_key": sam_key, "family": family,
                "n_rows": len(train_rows), "status": "OK",
                "phase2_elapsed_s": elapsed,
                "prevalence_catastrophic": prevalence_cat,
                "prevalence_instability": prevalence_inst,
                "cat_model": cat_model.name,
                "inst_model": inst_model.name,
                "cat_auroc_instability": cat_auroc_inst,
                "cat_auprc_catastrophic": cat_auprc_cat,
                "inst_auroc_instability": inst_auroc_inst,
                "inst_auprc_catastrophic": inst_auprc_cat,
                "anti_correlation_confirmed": anti_corr,
                "gate_cond1_auroc_ge_0p70": inst_auroc_inst >= 0.70,
                "gate_cond2_auprc_gt_prevalence": cat_auprc_cat > prevalence_cat,
            }

        return result

    finally:
        wp1.SAM_CHECKPOINT = _orig_sam_ckpt
        wp1.DINOV2_CHECKPOINT = _orig_dinov2_ckpt
        wp1.HOLDOUT_QUERY_MANIFEST = _orig_query_manifest
        wp1.PHASET_MANIFEST_DIR = _orig_manifest_dir
        wp1.SEED_COUNT = _orig_seed_count
        wp1.load_sam_predictor = _orig_load_sam


# ---------------------------------------------------------------------------
# SAM2 Predictor Wrapper (duck-types SamPredictor)
# ---------------------------------------------------------------------------

class _Sam2PredictorWrapper:
    """Duck-types segment_anything.SamPredictor for SAM2.

    NOTE: SAM2 bfloat16 autocast conflicts with DINOv2 float32 feature
    extraction when both run in the same CUDA context. We run SAM2 in
    float32 (no autocast) to avoid illegal memory access errors.
    """

    def __init__(self, predictor: Any, autocast_ctx: Any):
        self._predictor = predictor
        self._autocast_ctx = autocast_ctx
        self._image: np.ndarray | None = None
        self._use_autocast = False  # Disabled to avoid DINOv2 conflict

    def set_image(self, image: np.ndarray) -> None:
        self._image = image

    def predict(
        self,
        point_coords: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
        box: np.ndarray | None = None,
        mask_input: Any = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import contextlib
        ctx = self._autocast_ctx if self._use_autocast else contextlib.nullcontext()
        with torch.inference_mode(), ctx:
            self._predictor.set_image(self._image)
            masks, scores, logits = self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=multimask_output,
            )
        return masks.astype(bool), scores.astype(np.float32), logits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase U WP1 Multi-Model Segmenter Audit")
    parser.add_argument("--models", type=str, default="sam_b",
                        help="Comma-separated model keys")
    parser.add_argument("--out-dir", type=Path,
                        default=PROJECT_ROOT / "reports" / "phaseU_wp1")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: 2 queries x 2 seeds")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    model_keys = [k.strip() for k in args.models.split(",")]

    for mk in model_keys:
        if mk not in ALL_SAM:
            raise ValueError(f"Unknown model: {mk}. Choose from: {list(ALL_SAM.keys())}")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[phaseU-wp1] device={device}, smoke={args.smoke}")
    print(f"[phaseU-wp1] models={model_keys}")
    print(f"[phaseU-wp1] out_dir={out_dir}")

    results: list[dict[str, Any]] = []

    for sam_key in model_keys:
        info = ALL_SAM[sam_key]
        print(f"\n{'='*60}")
        print(f"[phaseU-wp1] {sam_key}: {info['description']} ({info['family']})")
        print(f"{'='*60}")

        result = run_benchmark_for_model(
            sam_key=sam_key, out_dir=out_dir, device=device, smoke=args.smoke,
        )
        results.append(result)

    # --- Summary ---
    print("\n" + "=" * 85)
    print("[phaseU-wp1] ANTI-CORRELATION SUMMARY")
    print("=" * 85)
    header = (f"{'Model':<12} {'Family':<6} {'n_rows':>6} {'Inst_AUROC':>10} "
              f"{'Cat_AUPRC':>10} {'Prevalence':>10} {'Anti-Corr':>10} {'Gate1':>6} {'Gate2':>6}")
    print(header)
    print("-" * 85)
    for r in results:
        if r.get("status") == "INSUFFICIENT_DATA":
            print(f"{r['sam_key']:<12} {r['family']:<6} {r['n_rows']:>6} {'INSUFFICIENT_DATA'}")
            continue
        print(f"{r['sam_key']:<12} {r['family']:<6} {r['n_rows']:>6} "
              f"{r['inst_auroc_instability']:>10.4f} "
              f"{r['cat_auprc_catastrophic']:>10.4f} "
              f"{r['prevalence_catastrophic']:>10.4f} "
              f"{str(r['anti_correlation_confirmed']):>10} "
              f"{'PASS' if r['gate_cond1_auroc_ge_0p70'] else 'FAIL':>6} "
              f"{'PASS' if r['gate_cond2_auprc_gt_prevalence'] else 'FAIL':>6}")

    n_ok = sum(1 for r in results if r.get("status") == "OK")
    n_anti = sum(1 for r in results if r.get("anti_correlation_confirmed", False))
    print(f"\nAnti-correlation confirmed in {n_anti}/{n_ok} models")

    # Save summary
    summary = {
        "phase": "U", "work_package": "WP1",
        "experiment": "segmenter_sensitivity_audit",
        "date": datetime.now().isoformat(timespec="seconds"),
        "device": str(device), "smoke": args.smoke,
        "models": model_keys, "results": results,
        "summary": {"n_models_tested": n_ok, "n_anti_correlation": n_anti},
    }
    summary_path = out_dir / "phaseU_wp1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[phaseU-wp1] Summary -> {summary_path}")


if __name__ == "__main__":
    main()