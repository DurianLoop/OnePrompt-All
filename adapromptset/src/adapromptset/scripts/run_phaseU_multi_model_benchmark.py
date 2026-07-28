"""Phase U Multi-Model Instability Benchmark.

Runs the WP1 reference-instability pipeline (phases 1-3) on the MoNuSAC
train split with different (matcher, segmenter) pairs, then trains M2
dual-model selective prediction on each to verify the universality of the
instability-catastrophic anti-correlation across model variants.

Supported models
----------------
SAM1 family (current Python 3.13 env):
    sam_b       SAM-B   (vit_b, 91M)  -- Phase T baseline
    sam_l       SAM-L   (vit_l, 312M)
    sam_h       SAM-H   (vit_h, 636M)
    medsam_b    MedSAM-B (vit_b, 91M, medical fine-tuned)
    sam_med2d_b SAM-Med2D-B (vit_b, 91M, medical adapter)

SAM2 family (requires sam_gpu env, Python 3.11):
    sam2_t      SAM2.1-Hiera-Tiny
    sam2_s      SAM2.1-Hiera-Small
    sam2_b      SAM2.1-Hiera-Base+
    sam2_l      SAM2.1-Hiera-Large

SAM3 (requires sam_gpu env, Python 3.11):
    sam3        SAM3 (ModelScope)

Matchers:
    dinov2_l    DINOv2-L (vitl14, 304M)
    dinov2_b    DINOv2-B (vitb14, 86M)

Pipeline (per model pair)
-------------------------
Phase 1 (CPU):  reference resampling (5 seeds, train split)
Phase 2 (GPU):  DINOv2 matching + SAM segmentation -> episode CSV
Phase 3 (CPU):  metric computation (8 WP1 metric CSVs)
M2 training:    dual-model (instability + catastrophic) for anti-correlation check

CLI::

    # Smoke test SAM-B only
    python src/adapromptset/scripts/run_phaseU_multi_model_benchmark.py \
        --models sam_b --matcher dinov2_l --phase 2 --smoke --device cuda

    # Full SAM1 benchmark
    python src/adapromptset/scripts/run_phaseU_multi_model_benchmark.py \
        --models sam_b,sam_l,sam_h,medsam_b,sam_med2d_b \
        --matcher dinov2_l --phase all --device cuda

    # SAM2/SAM3 benchmark (requires sam_gpu env)
    D:\\SAM\\conda_envs\\sam_gpu\\python.exe \
        src/adapromptset/scripts/run_phaseU_multi_model_benchmark.py \
        --models sam2_t,sam2_s,sam2_b,sam2_l,sam3 \
        --matcher dinov2_l --phase 2 --device cuda --env sam_gpu
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
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import torch  # noqa: E402
import run_phaseT_wp1_holdout as wp1  # noqa: E402
import run_phaseT_wp3_m2_selective as m2  # noqa: E402

# ---------------------------------------------------------------------------
# Model definitions (correct checkpoint paths)
# ---------------------------------------------------------------------------
CKPT = WORKSPACE_ROOT / "assets" / "checkpoints"

SAM1_MODELS: dict[str, dict[str, Any]] = {
    "sam_b": {
        "checkpoint": CKPT / "sam_vit_b.pth",
        "architecture": "vit_b",
        "family": "sam1",
        "description": "SAM-B (vit_b, 91M)",
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
        "description": "SAM-H (vit_h, 636M)",
    },
    "medsam_b": {
        "checkpoint": CKPT / "medsam_vit_b.pth",
        "architecture": "vit_b",
        "family": "sam1",
        "description": "MedSAM-B (vit_b, medical fine-tuned)",
    },
    "sam_med2d_b": {
        "checkpoint": CKPT / "sam_med2d_b.pth",
        "architecture": "vit_b",
        "family": "sam1",
        "description": "SAM-Med2D-B (vit_b, medical adapter)",
    },
}

SAM2_MODELS: dict[str, dict[str, Any]] = {
    "sam2_t": {
        "checkpoint": CKPT / "sam2.1_hiera_tiny.pt",
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "family": "sam2",
        "description": "SAM2.1-Hiera-Tiny",
    },
    "sam2_s": {
        "checkpoint": CKPT / "sam2.1_hiera_small.pt",
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "family": "sam2",
        "description": "SAM2.1-Hiera-Small",
    },
    "sam2_b": {
        "checkpoint": CKPT / "sam2.1_hiera_base_plus.pt",
        "config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "family": "sam2",
        "description": "SAM2.1-Hiera-Base+",
    },
    "sam2_l": {
        "checkpoint": CKPT / "sam2.1_hiera_large.pt",
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "family": "sam2",
        "description": "SAM2.1-Hiera-Large",
    },
}

SAM3_MODELS: dict[str, dict[str, Any]] = {
    "sam3": {
        "checkpoint": CKPT / "sam3_modelscope" / "sam3.pt",
        "family": "sam3",
        "description": "SAM3 (ModelScope)",
    },
}

MATCHER_MODELS: dict[str, dict[str, Any]] = {
    "dinov2_l": {
        "checkpoint": CKPT / "dinov2_vitl14_pretrain.pth",
        "architecture": "large",
        "description": "DINOv2-L (vitl14, 304M)",
    },
    "dinov2_b": {
        "checkpoint": CKPT / "dinov2_vitb14_pretrain.pth",
        "architecture": "base",
        "description": "DINOv2-B (vitb14, 86M)",
    },
}

# ---------------------------------------------------------------------------
# SAM1 predictor (architecture-aware, replaces wp1.load_sam_predictor)
# ---------------------------------------------------------------------------

def _load_sam1_predictor(device: torch.device, architecture: str, checkpoint: Path) -> Any:
    """Load a SAM1-family predictor with the given architecture and checkpoint."""
    wp1.configure_external_paths(WORKSPACE_ROOT)
    from segment_anything import SamPredictor, sam_model_registry

    model = sam_model_registry[architecture](checkpoint=str(checkpoint))
    model.to(device)
    model.eval()
    return SamPredictor(model)


def _load_sam2_predictor(device_str: str, config: str, checkpoint: Path) -> Any:
    """Load a SAM2 image predictor. Must be called from sam_gpu env (Python 3.11).
    
    Uses Hydra initialize_config_dir + compose to bypass the missing config
    search path issue. SAM2 is distributed as .pyc files without full Hydra
    config infrastructure.
    """
    import sys as _sys
    _sys.path.insert(0, str(WORKSPACE_ROOT / "sam2"))
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    config_dir = WORKSPACE_ROOT / "sam2" / "sam2" / "configs"
    # config is e.g. "sam2.1/sam2.1_hiera_t"
    config_name = config.replace(".yaml", "").replace("configs/", "")
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=config_name)
        model = instantiate(cfg.model, _recursive_=True)
    ckpt_data = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(ckpt_data["model"], strict=True)
    model.to(device_str)
    model.eval()
    return SAM2ImagePredictor(model)


def _load_sam3_predictor(device_str: str, checkpoint: Path) -> Any:
    """Load a SAM3 processor. Must be called from sam_gpu env (Python 3.11).
    
    NOTE: SAM3 uses torch.jit.script internally (box_ops.py) which requires
    .py source files. Since only .pyc bytecode is available, SAM3 point-prompt
    mode is NOT AVAILABLE for the current benchmark.
    """
    raise NotImplementedError(
        "SAM3 point-prompt mode unavailable: torch.jit.script requires .py "
        "source files (box_ops.py). Only .pyc bytecode is distributed."
    )


# ---------------------------------------------------------------------------
# Phase 2: DINOv2 matching + SAM segmentation for one model pair
# ---------------------------------------------------------------------------

def _segment_with_sam1(
    predictor: Any,
    image: Any,
    point_coords: np.ndarray,
    point_labels: np.ndarray,
) -> np.ndarray:
    """Run SAM1 point-prompt segmentation. Returns binary mask (H, W)."""
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    best_idx = int(np.argmax(scores))
    return masks[best_idx].astype(bool)


def _segment_with_sam2(
    predictor: Any,
    image: np.ndarray,
    point_coords: np.ndarray,
    point_labels: np.ndarray,
    autocast_ctx: Any,
) -> np.ndarray:
    """Run SAM2 point-prompt segmentation. Returns binary mask (H, W)."""
    import contextlib
    with torch.inference_mode(), autocast_ctx:
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=(len(point_labels) <= 1),
        )
    best_idx = int(np.argmax(scores))
    return masks[best_idx].astype(bool)


def _segment_with_sam3(
    processor: Any,
    image: np.ndarray,
    point_coords: np.ndarray,
    point_labels: np.ndarray,
    autocast_ctx: Any,
) -> np.ndarray:
    """Run SAM3 point-prompt segmentation. Returns binary mask (H, W)."""
    import contextlib
    from PIL import Image as PILImage

    with autocast_ctx, torch.inference_mode():
        state = processor.set_image(PILImage.fromarray(image.astype(np.uint8), mode="RGB"))
        if "language_features" not in state["backbone_out"]:
            state["backbone_out"].update(
                processor.model.backbone.forward_text(["visual"], device=processor.device)
            )
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = processor.model._get_dummy_prompt()

        h, w = image.shape[:2]
        norm_points = point_coords.astype(np.float32).copy()
        norm_points[:, 0] /= float(w)
        norm_points[:, 1] /= float(h)
        points_t = torch.tensor(norm_points, device=processor.device, dtype=torch.float32).view(-1, 1, 2)
        labels_t = torch.tensor(point_labels, device=processor.device, dtype=torch.long).view(-1, 1)
        state["geometric_prompt"].append_points(points_t, labels_t)
        state = processor._forward_grounding(state)

    masks_t = state["masks"].detach().cpu()
    scores_t = state["scores"].detach().float().cpu()
    if masks_t.numel() == 0:
        return np.zeros(image.shape[:2], dtype=bool)
    best_idx = int(torch.argmax(scores_t).item())
    return masks_t[best_idx, 0].numpy().astype(bool)


def run_phase2_for_model(
    sam_key: str,
    matcher_key: str,
    out_dir: Path,
    device: torch.device,
    smoke: bool = False,
) -> Path:
    """Run Phase 2 (DINOv2 matching + SAM segmentation) for one model pair.

    Monkey-patches wp1.SAM_CHECKPOINT and wp1.DINOV2_CHECKPOINT, then calls
    wp1.run_phase2() with overridden constants.

    Returns the path to the generated episode CSV.
    """
    all_sam = {**SAM1_MODELS, **SAM2_MODELS, **SAM3_MODELS}
    sam_info = all_sam[sam_key]
    matcher_info = MATCHER_MODELS[matcher_key]

    # Validate checkpoints exist.
    for label, path in [("SAM", sam_info["checkpoint"]), ("Matcher", matcher_info["checkpoint"])]:
        if not path.is_file():
            raise FileNotFoundError(f"{label} checkpoint not found: {path}")

    family = sam_info["family"]
    seeds = list(range(2 if smoke else 5))

    # Monkey-patch WP1 constants.
    wp1.SAM_CHECKPOINT = sam_info["checkpoint"]
    wp1.DINOV2_CHECKPOINT = matcher_info["checkpoint"]
    train_manifest = PROJECT_ROOT / "manifests" / "phaseY" / "policy_train_queries.csv"
    wp1.HOLDOUT_QUERY_MANIFEST = train_manifest
    wp1.PHASET_MANIFEST_DIR = out_dir / "manifests"
    wp1.PHASET_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    wp1.SEED_COUNT = len(seeds)

    # Save original load_sam_predictor and FeatureCache for restoration.
    _original_load_sam = wp1.load_sam_predictor
    _original_feature_cache_init = wp1.FeatureCache.__init__

    if family == "sam1":
        # Patch load_sam_predictor for SAM1 architecture.
        arch = sam_info["architecture"]
        ckpt = sam_info["checkpoint"]

        def _patched_load_sam(device_override: torch.device) -> Any:
            return _load_sam1_predictor(device_override, arch, ckpt)

        wp1.load_sam_predictor = _patched_load_sam

    elif family == "sam2":
        # SAM2 requires a different predictor interface.
        # We patch load_sam_predictor to return a SAM2 predictor wrapper.
        import contextlib as _cl

        config = sam_info["config"]
        ckpt = sam_info["checkpoint"]
        device_str = str(device)

        sam2_predictor = _load_sam2_predictor(device_str, config, ckpt)
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device_str.startswith("cuda")
            else _cl.nullcontext()
        )

        def _patched_load_sam(device_override: torch.device) -> Any:
            # Return a wrapper that duck-types SamPredictor.
            return _Sam2PredictorWrapper(sam2_predictor, autocast_ctx)

        wp1.load_sam_predictor = _patched_load_sam

    elif family == "sam3":
        import contextlib as _cl

        ckpt = sam_info["checkpoint"]
        device_str = str(device)

        sam3_processor = _load_sam3_predictor(device_str, ckpt)
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device_str.startswith("cuda")
            else _cl.nullcontext()
        )

        def _patched_load_sam(device_override: torch.device) -> Any:
            return _Sam3PredictorWrapper(sam3_processor, autocast_ctx)

        wp1.load_sam_predictor = _patched_load_sam

    else:
        raise ValueError(f"Unknown model family: {family}")

    try:
        # ---- Phase 1: reference resampling ---------------------------------
        print(f"  [{sam_key}+{matcher_key}] Phase 1: reference resampling "
              f"({len(seeds)} seeds)")
        max_queries = 2 if smoke else None
        wp1.run_phase1(seeds=seeds, max_queries=max_queries)

        # ---- Phase 2: DINOv2 matching + SAM segmentation -------------------
        print(f"  [{sam_key}+{matcher_key}] Starting Phase 2 ({family}) ...")
        t0 = time.perf_counter()

        feature_cache = wp1.FeatureCache(device)
        predictor = wp1.load_sam_predictor(device)

        wp1.run_phase2(
            feature_cache=feature_cache,
            predictor=predictor,
            out_dir=out_dir,
            seeds=range(len(seeds)),
            max_queries=max_queries,
        )

        elapsed = time.perf_counter() - t0
        episode_csv = out_dir / "phase2_episode_instability.csv"
        if episode_csv.is_file():
            with open(str(episode_csv), encoding="utf-8") as f:
                n_rows = sum(1 for _ in f)
        else:
            n_rows = 0
        print(f"  [{sam_key}+{matcher_key}] Phase 2 done in {elapsed:.0f}s, "
              f"{n_rows} episode rows")
        return episode_csv
    finally:
        wp1.load_sam_predictor = _original_load_sam


# ---------------------------------------------------------------------------
# Predictor wrappers for SAM2/SAM3 (duck-type SamPredictor)
# ---------------------------------------------------------------------------

class _Sam2PredictorWrapper:
    """Duck-types segment_anything.SamPredictor for SAM2."""

    def __init__(self, predictor: Any, autocast_ctx: Any):
        self._predictor = predictor
        self._autocast_ctx = autocast_ctx
        self._image: np.ndarray | None = None

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
        with torch.inference_mode(), self._autocast_ctx:
            self._predictor.set_image(self._image)
            masks, scores, logits = self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=multimask_output,
            )
        return masks.astype(bool), scores.astype(np.float32), logits


class _Sam3PredictorWrapper:
    """Duck-types segment_anything.SamPredictor for SAM3."""

    def __init__(self, processor: Any, autocast_ctx: Any):
        self._processor = processor
        self._autocast_ctx = autocast_ctx
        self._image: np.ndarray | None = None

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
        from PIL import Image as PILImage

        with self._autocast_ctx, torch.inference_mode():
            state = self._processor.set_image(
                PILImage.fromarray(self._image.astype(np.uint8), mode="RGB")
            )
            if "language_features" not in state["backbone_out"]:
                state["backbone_out"].update(
                    self._processor.model.backbone.forward_text(
                        ["visual"], device=self._processor.device
                    )
                )
            if "geometric_prompt" not in state:
                state["geometric_prompt"] = self._processor.model._get_dummy_prompt()

            h, w = self._image.shape[:2]
            norm_points = point_coords.astype(np.float32).copy()
            norm_points[:, 0] /= float(w)
            norm_points[:, 1] /= float(h)
            points_t = torch.tensor(
                norm_points, device=self._processor.device, dtype=torch.float32
            ).view(-1, 1, 2)
            labels_t = torch.tensor(
                point_labels, device=self._processor.device, dtype=torch.long
            ).view(-1, 1)
            state["geometric_prompt"].append_points(points_t, labels_t)
            state = self._processor._forward_grounding(state)

        masks_t = state["masks"].detach().cpu()
        scores_t = state["scores"].detach().float().cpu()
        if masks_t.numel() == 0:
            h, w = self._image.shape[:2]
            empty = np.zeros((1, h, w), dtype=bool)
            return empty, np.zeros(1, dtype=np.float32), np.zeros((1, h, w), dtype=np.float32)
        masks = masks_t[:, 0].numpy().astype(bool)
        scores = scores_t.numpy().astype(np.float32)
        logits = masks_t[:, 0].numpy().astype(np.float32)
        return masks, scores, logits


# ---------------------------------------------------------------------------
# M2 dual-model training for anti-correlation check
# ---------------------------------------------------------------------------

def run_m2_for_model(
    sam_key: str,
    train_csv: Path,
    val_csv: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Train M2 dual-model and check anti-correlation."""
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = m2.load_episodes(train_csv)
    val_rows = m2.load_episodes(val_csv)

    if len(train_rows) < 50 or len(val_rows) < 50:
        return {
            "sam_key": sam_key,
            "status": "INSUFFICIENT_DATA",
            "n_train": len(train_rows),
            "n_val": len(val_rows),
        }

    train_labels = m2.construct_labels(train_rows)
    val_labels = m2.construct_val_labels(val_rows)

    # Train catastrophic model.
    cat_results, cat_model, cat_result_val = m2.evaluate_all_models(
        m2.build_all_models(), train_rows, val_rows, train_labels, val_labels,
        train_label_key="catastrophic",
    )

    # Train instability model.
    inst_results, inst_model, inst_result_val = m2.evaluate_all_models(
        m2.build_all_models(), train_rows, val_rows, train_labels, val_labels,
        train_label_key="instability",
    )

    cat_auroc_inst = cat_result_val["auroc_instability"]
    inst_auroc_inst = inst_result_val["auroc_instability"]
    cat_auprc_cat = cat_result_val["auprc_catastrophic"]
    inst_auprc_cat = inst_result_val["auprc_catastrophic"]

    anti_corr = (cat_auroc_inst < 0.60) and (inst_auprc_cat < 0.15)

    prevalence_cat = float(np.mean(val_labels["catastrophic"]))
    gate_cond1 = inst_auroc_inst >= m2.GATE_AUROC_THRESHOLD
    gate_cond2 = cat_auprc_cat > prevalence_cat

    return {
        "sam_key": sam_key,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "prevalence_catastrophic": float(prevalence_cat),
        "prevalence_instability": float(np.mean(val_labels["instability"])),
        "cat_model": cat_model.name,
        "inst_model": inst_model.name,
        "cat_auroc_instability": float(cat_auroc_inst),
        "cat_auprc_catastrophic": float(cat_auprc_cat),
        "inst_auroc_instability": float(inst_auroc_inst),
        "inst_auprc_catastrophic": float(inst_auprc_cat),
        "anti_correlation_confirmed": anti_corr,
        "gate_cond1_auroc_ge_0p70": gate_cond1,
        "gate_cond2_auprc_gt_prevalence": gate_cond2,
        "best_cat_auprc": float(cat_auprc_cat),
        "best_inst_auroc": float(inst_auroc_inst),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase U Multi-Model Instability Benchmark",
    )
    parser.add_argument(
        "--models", type=str, default="sam_b",
        help="Comma-separated model keys. SAM1: sam_b,sam_l,sam_h,medsam_b,sam_med2d_b. "
             "SAM2: sam2_t,sam2_s,sam2_b,sam2_l. SAM3: sam3.",
    )
    parser.add_argument(
        "--matcher", type=str, default="dinov2_l",
        help="Matcher key: dinov2_l, dinov2_b",
    )
    parser.add_argument(
        "--phase", type=str, default="all",
        choices=["2", "m2", "all"],
        help="2=benchmark only, m2=M2 only, all=both",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=PROJECT_ROOT / "reports" / "phaseU_multi_model",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test: 2 queries x 2 seeds.",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: auto, cuda, cpu",
    )
    parser.add_argument(
        "--env", type=str, default="current",
        choices=["current", "sam_gpu"],
        help="Python environment: current (3.13 for SAM1), sam_gpu (3.11 for SAM2/SAM3)",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model_keys = [k.strip() for k in args.models.split(",")]
    matcher_key = args.matcher

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_sam = {**SAM1_MODELS, **SAM2_MODELS, **SAM3_MODELS}

    # Validate model keys.
    for mk in model_keys:
        if mk not in all_sam:
            raise ValueError(f"Unknown model key: {mk}. Choose from: {list(all_sam.keys())}")
    if matcher_key not in MATCHER_MODELS:
        raise ValueError(f"Unknown matcher key: {matcher_key}. "
                         f"Choose from: {list(MATCHER_MODELS.keys())}")

    # Check env compatibility.
    sam1_keys = [k for k in model_keys if k in SAM1_MODELS]
    sam2_keys = [k for k in model_keys if k in SAM2_MODELS]
    sam3_keys = [k for k in model_keys if k in SAM3_MODELS]

    if (sam2_keys or sam3_keys) and args.env != "sam_gpu":
        print("WARNING: SAM2/SAM3 models require Python 3.11 (sam_gpu env).")
        print("  Run with: D:\\SAM\\conda_envs\\sam_gpu\\python.exe ... --env sam_gpu")
        if not sam1_keys:
            raise RuntimeError("No SAM1 models selected; cannot run with current env.")

    if sam1_keys and args.env == "sam_gpu":
        print("WARNING: SAM1 models can run in current env. Use --env current for SAM1.")

    print(f"[phaseU-multi-model] device={device}")
    print(f"[phaseU-multi-model] models={model_keys}")
    print(f"[phaseU-multi-model] matcher={matcher_key}")
    print(f"[phaseU-multi-model] env={args.env}")
    print(f"[phaseU-multi-model] smoke={args.smoke}")
    print(f"[phaseU-multi-model] out_dir={out_dir}")

    # Load train queries.
    train_manifest = PROJECT_ROOT / "manifests" / "phaseY" / "policy_train_queries.csv"
    train_queries: list[dict[str, str]] = []
    with open(str(train_manifest), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            train_queries.append(row)
    print(f"[phaseU-multi-model] loaded {len(train_queries)} train queries")

    seeds = list(range(2 if args.smoke else 5))

    results: list[dict[str, Any]] = []

    for sam_key in model_keys:
        sam_info = all_sam[sam_key]
        print()
        print(f"{'='*60}")
        print(f"[phaseU-multi-model] {sam_key}: {sam_info['description']}")
        print(f"  family: {sam_info['family']}")
        print(f"  checkpoint: {sam_info['checkpoint']}")
        print(f"{'='*60}")

        model_out_dir = out_dir / sam_key / matcher_key
        model_out_dir.mkdir(parents=True, exist_ok=True)

        train_csv = model_out_dir / "phase2_episode_instability.csv"

        if args.phase in ("2", "all"):
            run_phase2_for_model(
                sam_key=sam_key,
                matcher_key=matcher_key,
                out_dir=model_out_dir,
                device=device,
                smoke=args.smoke,
            )

        if args.phase in ("m2", "all"):
            if not train_csv.is_file():
                print(f"  [{sam_key}] No episode data at {train_csv}, skipping M2")
                continue

            val_csv = (
                PROJECT_ROOT / "reports" / "phaseT_wp3_validation"
                / "phase2_episode_instability.csv"
            )
            if not val_csv.is_file():
                print(f"  [{sam_key}] No val data at {val_csv}, using train as val")
                val_csv = train_csv

            m2_out_dir = model_out_dir / "m2"
            m2_result = run_m2_for_model(
                sam_key=sam_key,
                train_csv=train_csv,
                val_csv=val_csv,
                out_dir=m2_out_dir,
            )
            results.append(m2_result)

    # Print summary.
    if results:
        print()
        print("=" * 85)
        print("[phaseU-multi-model] ANTI-CORRELATION SUMMARY")
        print("=" * 85)
        header = (f"{'Model':<14} {'Inst_AUROC':>10} {'Cat_AUPRC':>10} "
                  f"{'Cat_AUROC':>10} {'Inst_AUPRC':>10} "
                  f"{'Anti-Corr':>10} {'Gate1':>6} {'Gate2':>6}")
        print(header)
        print("-" * 85)
        for r in results:
            if r.get("status") == "INSUFFICIENT_DATA":
                print(f"{r['sam_key']:<14} {'INSUFFICIENT_DATA':>10} "
                      f"(n_train={r.get('n_train',0)}, n_val={r.get('n_val',0)})")
                continue
            print(f"{r['sam_key']:<14} "
                  f"{r['best_inst_auroc']:>10.4f} "
                  f"{r['best_cat_auprc']:>10.4f} "
                  f"{r['cat_auroc_instability']:>10.4f} "
                  f"{r['inst_auprc_catastrophic']:>10.4f} "
                  f"{str(r['anti_correlation_confirmed']):>10} "
                  f"{'PASS' if r['gate_cond1_auroc_ge_0p70'] else 'FAIL':>6} "
                  f"{'PASS' if r['gate_cond2_auprc_gt_prevalence'] else 'FAIL':>6}")
        print("-" * 85)

        n_anti_corr = sum(1 for r in results if r.get("anti_correlation_confirmed", False))
        print(f"\nAnti-correlation confirmed in {n_anti_corr}/{len(results)} "
              f"models ({n_anti_corr/max(len(results),1)*100:.0f}%)")

    # Write summary JSON.
    summary = {
        "phase": "U",
        "experiment": "multi_model_instability_benchmark",
        "date": datetime.now().isoformat(timespec="seconds"),
        "matcher": matcher_key,
        "env": args.env,
        "models": model_keys,
        "n_train_queries": len(train_queries),
        "n_seeds": len(seeds),
        "smoke": args.smoke,
        "results": results,
        "summary": {
            "n_models_tested": len(results),
            "n_anti_correlation_confirmed": sum(
                1 for r in results if r.get("anti_correlation_confirmed", False)
            ),
        } if results else {},
    }
    (out_dir / "phaseU_multi_model_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(f"\n[phaseU-multi-model] summary -> {out_dir / 'phaseU_multi_model_summary.json'}")


if __name__ == "__main__":
    main()