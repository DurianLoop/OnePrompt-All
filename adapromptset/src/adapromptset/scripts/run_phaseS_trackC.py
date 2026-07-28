"""Phase S Track C: Stable Candidate Scoring and Pixel-wise Agreement Gate.

Track C (research.md §8.2) intervenes at two levels DIFFERENT from Track B:

* **C3 — Pixel-wise Agreement Gate** (pixel aggregation level): for each
  reference *r*, compute a per-pixel weight based on agreement with other
  references, then produce a gated consensus

      C_mean(x) = (1/R) Σ_r C_r(x)
      d_r(x)    = |C_r(x) - C_mean(x)|
      w_r(x)    = 1 / (1 + α · d_r(x))
      C_gated(x)= Σ_r w_r(x)·C_r(x) / Σ_r w_r(x)

* **C4 — Stable Candidate Scoring** (candidate selection level): re-rank
  candidates by a stability score

      μ_i = mean contrast at position i across R references
      σ_i = std of contrast at position i across R references
      a_i = agreement = fraction of references where position i is in their
            individual top-K positives
      s_i = μ_i - β·σ_i + γ·a_i

  Select top-K candidates by *s_i* instead of by *μ_i*.

* **Combined C3+C4**: apply C3 gating first, then C4 scoring on the gated
  consensus.

Track B intervened at the *reference weighting* level (C_weighted = Σ w_r C_r
with learned w_r) and completed with ``PHASES_TRACK_B_NO_GAIN`` — reliability
weighting is predictable (Spearman=0.478) but does not improve Dice
(ΔDice=-0.0024).  Track C starts because Track B has stable positive signal
but plateaued.

Pipeline (3 phases):

* **Phase 1 (CPU)** — run SAM-B on TRAIN with all (α, β, γ) combos (seed 0
  only for efficiency), select the best combo by mean Dice.  80 queries × 3
  configs × 27 combos = 6480 SAM calls.
* **Phase 2 (CPU)** — run SAM-B on validation with the selected (α, β, γ).
  41 queries × 3 configs = 123 SAM calls.  Track B weighted consensus
  (M4_gbm) is the baseline for Gate C.
* **Phase 3 (CPU)** — evaluate Gate C (§8.4): 8 conditions relative to Track
  B best, plus cond4 (3-seed consistency), cond5 (reference-count variation),
  and cond6 (leave-one-reference-out) which require additional SAM calls.

Outputs (under ``reports/phaseS_trackC/``)::

    Phase 1:
        train_grid_results.csv     (query, config, alpha, beta, gamma, dice, ...)
        train_selection.json       (best_alpha, best_beta, best_gamma, ...)
    Phase 2:
        val_trackC_consensus_metrics.csv  (query, config, method, alpha, beta, gamma, ...)
    Phase 3:
        gateC_summary.json         (full gate evaluation)
        gateC_summary.md           (human-readable summary)
        val_comparison.csv         (per-query Track C vs Track B comparison)
        run_metadata.json          (device, runtime, SAM calls, GPU hours)
        loo_stability.csv          (leave-one-reference-out results for cond6)
        seed_consistency.csv       (3-seed train results for cond4)
        refcount_stability.csv     (k=3 vs k=6 results for cond5)

CPU-only safe (torch 2.12.0+cpu).  SAM-B runs on CPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

# OpenMP runtime conflict workaround (conda + torch on Windows).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from PIL import Image

# Make ``adapromptset`` importable when run as ``python src/.../script.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parent
_SRC = PROJECT_ROOT / "src"
import sys
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from adapromptset.candidates.points import (  # noqa: E402
    CandidatePools,
    build_candidate_pools,
)
from adapromptset.evaluation.metrics import (  # noqa: E402
    dice_score,
    iou_score,
)
from adapromptset.matching.evidence import (  # noqa: E402
    IMAGE_SIZE,
    PATCH_SIZE,
    extract_dinov2_features,
    load_dinov2_model,
    matching_scores_from_features,
    patch_weights_from_mask,
)
from adapromptset.matching.runtime import configure_external_paths  # noqa: E402

SAM_CHECKPOINT = WORKSPACE_ROOT / "assets" / "checkpoints" / "sam_vit_b.pth"
DINOV2_CHECKPOINT = WORKSPACE_ROOT / "assets" / "checkpoints" / "dinov2_vitl14_pretrain.pth"
MONUSAC_ROOT = WORKSPACE_ROOT / "code" / "datasets" / "MoNuSAC" / "processed"

POSITIVE_LIMIT = 100
NEGATIVE_LIMIT = 5
MIN_POSITIVE_DISTANCE = 4.0

CATASTROPHIC_DICE_THRESHOLD = 0.1
SEED_COUNT = 5
REFS_PER_QUERY = 6
GRID_SIZE = IMAGE_SIZE // PATCH_SIZE  # 37

# Track C hyperparameter grids (§8.2).
ALPHA_GRID = (0.0, 1.0, 5.0)       # C3 gate strength (0 = off = baseline)
BETA_GRID = (0.0, 0.5, 1.0)        # C4 stability penalty (0 = off)
GAMMA_GRID = (0.0, 0.5, 1.0)       # C4 agreement bonus (0 = off)

# Smoke-test reduced grid.
ALPHA_GRID_SMOKE = (0.0, 1.0)
BETA_GRID_SMOKE = (0.0, 0.5)
GAMMA_GRID_SMOKE = (0.0, 0.5)

N_BOOTSTRAP = 10000

# Track B best baseline (frozen from Phase S Track B gateB_summary.json).
TRACK_B_MEAN_DICE = 0.473183
TRACK_B_MEAN_REF_STD = 0.086751
TRACK_B_CFR = 0.073171
TRACK_B_PHASE2_RUNTIME_SECONDS = 308.5


# ---------------------------------------------------------------------------
# Frozen prompt configs (only 3, not 210) — copied from R2
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PromptConfig:
    name: str
    k_positive: int
    k_negative: int
    positive_selector: str
    negative_selector: str
    grouping: str
    sam_calls: int


FROZEN_CONFIGS: tuple[PromptConfig, ...] = (
    PromptConfig("P20-N1", 20, 1, "P_TOPK", "N_BOTTOMK", "G_ONE_GROUP", 1),
    PromptConfig("P30-N0", 30, 0, "P_TOPK", "N_NONE", "G_ONE_GROUP", 1),
    PromptConfig("P50-N0", 50, 0, "P_TOPK", "N_NONE", "G_ONE_GROUP", 1),
)


# ---------------------------------------------------------------------------
# Case ID -> path resolution (copied from R2)
# ---------------------------------------------------------------------------

def _split_from_case_id(case_id: str) -> str:
    """train_065 -> train, test_001 -> test."""
    return case_id.rsplit("_", 1)[0]


def resolve_case_paths(case_id: str) -> dict[str, Path]:
    """Resolve image and mask paths for a MoNuSAC case_id."""
    split = _split_from_case_id(case_id)
    return {
        "image": MONUSAC_ROOT / split / "images" / f"{case_id}.png",
        "mask": MONUSAC_ROOT / split / "masks" / f"{case_id}.png",
    }


# ---------------------------------------------------------------------------
# Image / mask loading (copied from R2)
# ---------------------------------------------------------------------------

def load_rgb_518(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC
    )


def load_mask_518(path: Path) -> list[list[bool]]:
    """Load binary mask at 518x518 as list[list[bool]] for metric functions."""
    image = Image.open(path).convert("L").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST
    )
    values = list(image.getdata())
    return [[bool(values[row * IMAGE_SIZE + col]) for col in range(IMAGE_SIZE)]
            for row in range(IMAGE_SIZE)]


def load_mask_tensor(path: Path) -> torch.Tensor:
    """Load binary mask at 518x518 as torch tensor for patch weight computation."""
    image = Image.open(path).convert("L").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST
    )
    values = torch.tensor(list(image.getdata()), dtype=torch.float32).reshape(
        IMAGE_SIZE, IMAGE_SIZE
    )
    return (values > 0).to(torch.float32)


# ---------------------------------------------------------------------------
# Component labels for recall computation (copied from R2)
# ---------------------------------------------------------------------------

def _component_labels(mask: Sequence[Sequence[bool]]) -> dict[tuple[int, int], int]:
    rows = len(mask)
    cols = len(mask[0]) if rows else 0
    labels: dict[tuple[int, int], int] = {}
    current = 0
    for row in range(rows):
        for col in range(cols):
            if not mask[row][col] or (row, col) in labels:
                continue
            current += 1
            queue: deque[tuple[int, int]] = deque([(row, col)])
            labels[(row, col)] = current
            while queue:
                r, c = queue.popleft()
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (
                        0 <= nr < rows
                        and 0 <= nc < cols
                        and mask[nr][nc]
                        and (nr, nc) not in labels
                    ):
                        labels[(nr, nc)] = current
                        queue.append((nr, nc))
    return labels


def _component_recall(
    prediction: Sequence[Sequence[bool]],
    target_labels: dict[tuple[int, int], int],
) -> tuple[float, int, int]:
    all_components = set(target_labels.values())
    total = len(all_components)
    if total == 0:
        return 0.0, 0, 0
    hit: set[int] = set()
    for (row, col), label in target_labels.items():
        if prediction[row][col]:
            hit.add(label)
    return len(hit) / total, len(hit), total


# ---------------------------------------------------------------------------
# SAM predictor (copied from R2)
# ---------------------------------------------------------------------------

def load_sam_predictor(device: torch.device) -> Any:
    configure_external_paths(WORKSPACE_ROOT)
    from segment_anything import SamPredictor, sam_model_registry

    architecture = "vit_b"
    model = sam_model_registry[architecture](checkpoint=str(SAM_CHECKPOINT))
    model.to(device)
    model.eval()
    return SamPredictor(model)


# ---------------------------------------------------------------------------
# DINOv2 feature cache (copied from R2)
# ---------------------------------------------------------------------------

class FeatureCache:
    """Per-case DINOv2-L feature cache to avoid redundant forward passes."""

    def __init__(self, device: torch.device):
        self._device = device
        self._model: torch.nn.Module | None = None
        self._features: dict[str, torch.Tensor] = {}
        self._mask_tensors: dict[str, torch.Tensor] = {}

    def _ensure_model(self) -> torch.nn.Module:
        if self._model is None:
            self._model = load_dinov2_model(
                DINOV2_CHECKPOINT, workspace_root=WORKSPACE_ROOT, device=self._device
            )
        return self._model

    def get_features(self, case_id: str) -> torch.Tensor:
        if case_id not in self._features:
            paths = resolve_case_paths(case_id)
            image = load_rgb_518(paths["image"])
            model = self._ensure_model()
            with torch.inference_mode():
                features = extract_dinov2_features(model, image, self._device)
            self._features[case_id] = features.cpu()
        return self._features[case_id]

    def get_mask_tensor(self, case_id: str) -> torch.Tensor:
        if case_id not in self._mask_tensors:
            paths = resolve_case_paths(case_id)
            self._mask_tensors[case_id] = load_mask_tensor(paths["mask"])
        return self._mask_tensors[case_id]


# ---------------------------------------------------------------------------
# Point selection (copied from R2)
# ---------------------------------------------------------------------------

def select_positives(pools: CandidatePools, k: int) -> list[tuple[int, int]]:
    """Return K positive pixel coordinates for the P_TOPK pool."""
    coords = [c.coordinate for c in pools.positives]
    return coords[:k] if k <= len(coords) else coords


def select_negatives(
    pools: CandidatePools,
    selector: str,
    k: int,
) -> list[tuple[int, int]]:
    """Return K negative pixel coordinates. All pools share the same negatives."""
    if k == 0 or selector == "N_NONE":
        return []
    neg_coords = [c.coordinate for c in pools.negatives]
    if k > len(neg_coords):
        k = len(neg_coords)
    if selector == "N_BOTTOMK":
        return neg_coords[:k]
    raise ValueError(f"unknown negative selector: {selector}")


# ---------------------------------------------------------------------------
# Config execution: SAM predict + metrics (copied from R2)
# ---------------------------------------------------------------------------

@dataclass
class ConfigRecord:
    """Result of running one config on one (query, seed, reference)."""
    dice: float | None
    iou: float | None
    component_recall: float | None
    target_component_count: int
    hit_components: int | None
    sam_score: float | None
    runtime_ms: float
    status: str
    n_pos_points: int
    n_neg_points: int
    pos_coords: list[tuple[int, int]]
    pred_mask: np.ndarray | None  # bool 2D array at 518x518


def run_config(
    predictor: Any,
    config: PromptConfig,
    pools: CandidatePools,
    target: list[list[bool]],
    target_labels: dict[tuple[int, int], int],
    target_component_count: int,
) -> ConfigRecord:
    """Run one prompt config on one reference and return a metrics record."""
    pos_points = select_positives(pools, config.k_positive)
    if not pos_points:
        return ConfigRecord(
            dice=None, iou=None, component_recall=None,
            target_component_count=target_component_count, hit_components=None,
            sam_score=None, runtime_ms=0.0, status="SKIPPED_NO_POSITIVES",
            n_pos_points=0, n_neg_points=0, pos_coords=[], pred_mask=None,
        )

    neg_points = select_negatives(pools, config.negative_selector, config.k_negative)
    all_points = pos_points + neg_points
    labels = [1] * len(pos_points) + [0] * len(neg_points)

    t0 = time.perf_counter()
    try:
        masks, scores, _ = predictor.predict(
            point_coords=np.asarray(all_points, dtype=np.float32),
            point_labels=np.asarray(labels, dtype=np.int32),
            multimask_output=True,
        )
    except Exception as exc:
        return ConfigRecord(
            dice=None, iou=None, component_recall=None,
            target_component_count=target_component_count, hit_components=None,
            sam_score=None,
            runtime_ms=round((time.perf_counter() - t0) * 1000.0, 3),
            status=f"SAM_ERROR:{type(exc).__name__}",
            n_pos_points=len(pos_points), n_neg_points=len(neg_points),
            pos_coords=pos_points, pred_mask=None,
        )
    best = int(np.argmax(scores))
    mask_np = masks[best].astype(bool)
    mask = mask_np.tolist()
    score = float(scores[best])
    runtime_ms = (time.perf_counter() - t0) * 1000.0

    dice = dice_score(mask, target)
    iou = iou_score(mask, target)
    recall, hit_count, total_count = _component_recall(mask, target_labels)

    return ConfigRecord(
        dice=dice, iou=iou, component_recall=recall,
        target_component_count=total_count, hit_components=hit_count,
        sam_score=score, runtime_ms=round(runtime_ms, 3), status="OK",
        n_pos_points=len(pos_points), n_neg_points=len(neg_points),
        pos_coords=pos_points, pred_mask=mask_np,
    )


def run_config_with_coords(
    predictor: Any,
    config: PromptConfig,
    pools: CandidatePools,
    pos_coords: list[tuple[int, int]],
    target: list[list[bool]],
    target_labels: dict[tuple[int, int], int],
    target_component_count: int,
) -> ConfigRecord:
    """Run SAM with explicit positive coordinates (for C4 scoring)."""
    if not pos_coords:
        return ConfigRecord(
            dice=None, iou=None, component_recall=None,
            target_component_count=target_component_count, hit_components=None,
            sam_score=None, runtime_ms=0.0, status="SKIPPED_NO_POSITIVES",
            n_pos_points=0, n_neg_points=0, pos_coords=[], pred_mask=None,
        )

    neg_points = select_negatives(pools, config.negative_selector, config.k_negative)
    all_points = pos_coords + neg_points
    labels = [1] * len(pos_coords) + [0] * len(neg_points)

    t0 = time.perf_counter()
    try:
        masks, scores, _ = predictor.predict(
            point_coords=np.asarray(all_points, dtype=np.float32),
            point_labels=np.asarray(labels, dtype=np.int32),
            multimask_output=True,
        )
    except Exception as exc:
        return ConfigRecord(
            dice=None, iou=None, component_recall=None,
            target_component_count=target_component_count, hit_components=None,
            sam_score=None,
            runtime_ms=round((time.perf_counter() - t0) * 1000.0, 3),
            status=f"SAM_ERROR:{type(exc).__name__}",
            n_pos_points=len(pos_coords), n_neg_points=len(neg_points),
            pos_coords=pos_coords, pred_mask=None,
        )
    best = int(np.argmax(scores))
    mask_np = masks[best].astype(bool)
    mask = mask_np.tolist()
    score = float(scores[best])
    runtime_ms = (time.perf_counter() - t0) * 1000.0

    dice = dice_score(mask, target)
    iou = iou_score(mask, target)
    recall, hit_count, total_count = _component_recall(mask, target_labels)

    return ConfigRecord(
        dice=dice, iou=iou, component_recall=recall,
        target_component_count=total_count, hit_components=hit_count,
        sam_score=score, runtime_ms=round(runtime_ms, 3), status="OK",
        n_pos_points=len(pos_coords), n_neg_points=len(neg_points),
        pos_coords=pos_coords, pred_mask=mask_np,
    )


# ---------------------------------------------------------------------------
# CSV helpers (copied from R2)
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def _jaccard_masks(a: np.ndarray, b: np.ndarray) -> float:
    a_ = a.ravel()
    b_ = b.ravel()
    inter = int(np.logical_and(a_, b_).sum())
    union = int(np.logical_or(a_, b_).sum())
    if union == 0:
        return 1.0
    return inter / union


def _jaccard_sets(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


# ---------------------------------------------------------------------------
# Manifest loading (copied from R2)
# ---------------------------------------------------------------------------

def load_train_resampling(seed_index: int) -> list[dict[str, str]]:
    """Load the train resampling manifest for the given seed index (0-4)."""
    path = PROJECT_ROOT / "manifests" / "phaseR" / f"train_resampling_seed_{seed_index}.csv"
    return _read_csv(path)


def load_validation_ab_fixed() -> list[dict[str, str]]:
    """Load the validation A/B fixed manifest (41 queries x 6 refs)."""
    path = PROJECT_ROOT / "manifests" / "phaseR" / "validation_AB_fixed.csv"
    return _read_csv(path)


def queries_from_train_manifests(seeds: range) -> list[str]:
    """Return the ordered, unique list of query_case_ids across the loaded seeds."""
    seen: list[str] = []
    seen_set: set[str] = set()
    for s in seeds:
        for row in load_train_resampling(s):
            qid = row["query_case_id"]
            if qid not in seen_set:
                seen_set.add(qid)
                seen.append(qid)
    return seen


def validation_queries() -> list[str]:
    """Return the ordered, unique list of validation query_case_ids."""
    seen: list[str] = []
    seen_set: set[str] = set()
    for row in load_validation_ab_fixed():
        qid = row["query_case_id"]
        if qid not in seen_set:
            seen_set.add(qid)
            seen.append(qid)
    return seen


def refs_for_query_seed(seed_index: int, query_case_id: str) -> list[dict[str, str]]:
    """Return the 6 reference rows for one (query, seed)."""
    rows = load_train_resampling(seed_index)
    return [r for r in rows if r["query_case_id"] == query_case_id]


def refs_for_query_val(query_case_id: str) -> list[dict[str, str]]:
    """Return the 6 reference rows for one validation query."""
    rows = load_validation_ab_fixed()
    return [r for r in rows if r["query_case_id"] == query_case_id]


# ---------------------------------------------------------------------------
# Contrast maps + per-reference pair state (copied from R2)
# ---------------------------------------------------------------------------

def _zscore_np(grid: np.ndarray) -> np.ndarray:
    """Per-grid standardization matching points._zscore_grid (population std)."""
    mean = float(grid.mean())
    std = float(grid.std())  # ddof=0 == population std
    if std == 0.0:
        return np.zeros_like(grid)
    return (grid - mean) / std


@dataclass(frozen=True)
class PairStateR2:
    """Per-(query, ref) pair state: pools + s_fg/s_bg + contrast map C_r."""
    pools: CandidatePools
    s_fg: np.ndarray  # grid_size x grid_size float64
    s_bg: np.ndarray  # grid_size x grid_size float64
    contrast: np.ndarray  # C_r = zscore(s_fg) - zscore(s_bg)


def build_pair_state_r2(
    query_case_id: str,
    reference_case_id: str,
    feature_cache: FeatureCache,
) -> PairStateR2:
    """Build per-reference pool, s_fg, s_bg, and contrast map for one (query, ref)."""
    query_features = feature_cache.get_features(query_case_id)
    ref_features = feature_cache.get_features(reference_case_id)
    ref_mask = feature_cache.get_mask_tensor(reference_case_id)

    fg_weights = patch_weights_from_mask(ref_mask, patch_size=PATCH_SIZE, foreground=True)
    bg_weights = patch_weights_from_mask(ref_mask, patch_size=PATCH_SIZE, foreground=False)
    grid_size = IMAGE_SIZE // PATCH_SIZE
    maps = matching_scores_from_features(
        support_features=ref_features,
        query_features=query_features,
        foreground_weights=fg_weights,
        background_weights=bg_weights,
        grid_size=grid_size,
    )
    s_fg = maps["s_fg"].numpy().astype(np.float64)
    s_bg = maps["s_bg"].numpy().astype(np.float64)
    contrast = _zscore_np(s_fg) - _zscore_np(s_bg)

    pools = build_candidate_pools(
        s_fg.tolist(), s_bg.tolist(),
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        positive_limit=POSITIVE_LIMIT,
        negative_limit=NEGATIVE_LIMIT,
        positive_layout="topk",
        min_positive_distance=MIN_POSITIVE_DISTANCE,
    )
    return PairStateR2(pools=pools, s_fg=s_fg, s_bg=s_bg, contrast=contrast)


def build_consensus_pool(contrast_map: np.ndarray, config: PromptConfig) -> CandidatePools:
    """Build candidate pool from an aggregated contrast map (config-independent)."""
    del config  # config-independent pool; kept for signature parity
    grid_size = contrast_map.shape[0]
    fg = contrast_map.tolist()
    bg = [[0.0 for _ in range(grid_size)] for _ in range(grid_size)]
    return build_candidate_pools(
        fg, bg,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        positive_limit=POSITIVE_LIMIT,
        negative_limit=NEGATIVE_LIMIT,
        positive_layout="topk",
        min_positive_distance=MIN_POSITIVE_DISTANCE,
    )


def _aggregate(stacked: np.ndarray, method: str, hyperparam: str = "") -> np.ndarray:
    """Aggregate a (m, H, W) stack of contrast maps for one consensus variant."""
    if method == "B2_mean":
        return np.mean(stacked, axis=0)
    if method == "B3_median":
        return np.median(stacked, axis=0)
    raise ValueError(f"unknown consensus method: {method}")


# ---------------------------------------------------------------------------
# Checkpointing helpers (copied/adapted from R2)
# ---------------------------------------------------------------------------

def _open_csv(path: Path, fields: list[str]) -> tuple[Any, Any]:
    is_append = path.is_file()
    handle = path.open("a" if is_append else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fields)
    if not is_append:
        writer.writeheader()
    return handle, writer


def _load_grid_checkpoint(path: Path) -> set[tuple[str, str, str, str, str]]:
    """Return set of (query, config, alpha, beta, gamma) done in the grid CSV."""
    if not path.is_file():
        return set()
    done: set[tuple[str, str, str, str, str]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            done.add((
                row["query_case_id"], row["config_name"],
                row["alpha"], row["beta"], row["gamma"],
            ))
    return done


def _load_trackc_checkpoint(path: Path) -> set[tuple[str, str]]:
    """Return set of (query, config) done for the Track C val CSV."""
    if not path.is_file():
        return set()
    done: set[tuple[str, str]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            done.add((row["query_case_id"], row["config_name"]))
    return done


def _row_dice(row: dict[str, str]) -> float | None:
    s = row.get("dice", "")
    return float(s) if s not in ("", None) else None


def _safe_float(s: str | None) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def _cvar(dices: list[float], quantile: float = 0.1) -> float:
    """Mean of the worst `quantile` fraction of Dice values."""
    if not dices:
        return 0.0
    arr = np.asarray(dices, dtype=np.float64)
    k = max(1, int(math.ceil(quantile * len(arr))))
    worst = np.sort(arr)[:k]
    return float(worst.mean())


def _paired_bootstrap(
    deltas: list[float], n_bootstrap: int = N_BOOTSTRAP, seed: int = 20260722,
) -> dict[str, float]:
    if not deltas:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    rng = np.random.RandomState(seed)
    arr = np.asarray(deltas, dtype=np.float64)
    n = len(arr)
    means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        means[i] = arr[idx].mean()
    return {
        "mean": float(arr.mean()),
        "ci_lower": float(np.percentile(means, 2.5)),
        "ci_upper": float(np.percentile(means, 97.5)),
    }


def _resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _fmt_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return repr(v)
    return str(v)


# ---------------------------------------------------------------------------
# C3 — Pixel-wise Agreement Gate
# ---------------------------------------------------------------------------

def c3_gated_consensus(contrast_stack: np.ndarray, alpha: float) -> np.ndarray:
    """Compute pixel-wise agreement-gated consensus.

    Args:
        contrast_stack: (R, H, W) per-reference contrast maps.
        alpha: gate strength (0 = uniform mean = baseline).

    Returns: (H, W) gated consensus map.
    """
    c_mean = np.mean(contrast_stack, axis=0)
    if alpha == 0.0:
        return c_mean
    deviations = np.abs(contrast_stack - c_mean)  # (R, H, W)
    weights = 1.0 / (1.0 + alpha * deviations)  # (R, H, W)
    weighted_sum = np.sum(weights * contrast_stack, axis=0)
    weight_sum = np.sum(weights, axis=0)
    return weighted_sum / np.maximum(weight_sum, 1e-8)


# ---------------------------------------------------------------------------
# C4 — Stable Candidate Scoring
# ---------------------------------------------------------------------------

def _precompute_ref_topk_sets(
    contrast_stack: np.ndarray,
    config: PromptConfig,
) -> list[set[tuple[int, int]]]:
    """Precompute each reference's individual top-K positive coordinate sets.

    These are independent of (alpha, beta, gamma) and can be reused across the
    full grid.
    """
    k = config.k_positive
    ref_topk_sets: list[set[tuple[int, int]]] = []
    for r in range(contrast_stack.shape[0]):
        ref_pool = build_consensus_pool(contrast_stack[r], config)
        ref_topk_sets.append(set(c.coordinate for c in ref_pool.positives[:k]))
    return ref_topk_sets


def c4_score_candidates(
    contrast_stack: np.ndarray,
    consensus_map: np.ndarray,
    config: PromptConfig,
    beta: float,
    gamma: float,
    ref_topk_sets: list[set[tuple[int, int]]] | None = None,
) -> list[tuple[int, int]]:
    """Select top-K positive candidates using stability score s_i = μ_i - β·σ_i + γ·a_i.

    Args:
        contrast_stack: (R, H, W) per-reference contrast maps (H=W=grid_size=37).
        consensus_map: (H, W) aggregated consensus (possibly C3-gated).
        config: prompt config (determines K).
        beta: stability penalty weight.
        gamma: agreement bonus weight.
        ref_topk_sets: precomputed per-reference top-K coordinate sets (optional
            optimisation; computed from contrast_stack if None).

    Returns: list of (x, y) pixel coordinates for top-K candidates (suitable for
    SAM point_coords).
    """
    R = contrast_stack.shape[0]
    k = config.k_positive

    # μ and σ at each pixel (grid space).
    mu = consensus_map  # (H, W) in grid space
    sigma = np.std(contrast_stack, axis=0)  # (H, W) in grid space

    # Build candidate pool from consensus to get candidate positions.
    pool = build_consensus_pool(consensus_map, config)
    # Use grid_index for scoring (row, col in 37x37), coordinate for SAM (x, y in 518x518).
    candidate_grid = [c.grid_index for c in pool.positives]  # up to POSITIVE_LIMIT
    candidate_coords = [c.coordinate for c in pool.positives]

    # Precompute per-reference top-K sets if not provided.
    if ref_topk_sets is None:
        ref_topk_sets = _precompute_ref_topk_sets(contrast_stack, config)

    # Agreement: fraction of references that include each candidate in their top-K.
    agreement = np.zeros(len(candidate_coords), dtype=np.float64)
    for ref_topk in ref_topk_sets:
        for i, coord in enumerate(candidate_coords):
            if coord in ref_topk:
                agreement[i] += 1.0
    agreement /= R

    # Score each candidate using grid_index to index mu/sigma.
    scores: list[tuple[float, tuple[int, int]]] = []
    for i, (grow, gcol) in enumerate(candidate_grid):
        mu_i = mu[grow, gcol]
        sigma_i = sigma[grow, gcol]
        a_i = agreement[i]
        s_i = mu_i - beta * sigma_i + gamma * a_i
        scores.append((s_i, candidate_coords[i]))

    # Sort by score descending and take top-K.
    scores.sort(key=lambda x: -x[0])
    return [coord for _, coord in scores[:k]]


# ---------------------------------------------------------------------------
# Unified consensus runner (B2_mean baseline or C3+C4)
# ---------------------------------------------------------------------------

def run_consensus_for_query(
    predictor: Any,
    config: PromptConfig,
    pair_states: list[PairStateR2],
    target: list[list[bool]],
    target_labels: dict[tuple[int, int], int],
    target_component_count: int,
    alpha: float = 0.0,
    beta: float = 0.0,
    gamma: float = 0.0,
) -> ConfigRecord:
    """Run consensus (B2_mean when α=β=γ=0, or C3+C4 otherwise) for a query.

    ``pair_states`` is the list of valid per-reference pair states (None entries
    already filtered out).
    """
    if not pair_states:
        return ConfigRecord(
            dice=None, iou=None, component_recall=None,
            target_component_count=target_component_count, hit_components=None,
            sam_score=None, runtime_ms=0.0, status="SKIPPED_NO_PAIR_STATES",
            n_pos_points=0, n_neg_points=0, pos_coords=[], pred_mask=None,
        )

    stack = np.stack([ps.contrast for ps in pair_states], axis=0)  # (R, H, W)
    consensus = c3_gated_consensus(stack, alpha)

    if beta == 0.0 and gamma == 0.0:
        # Baseline path: select positives by contrast value (B2_mean or C3-only).
        pools = build_consensus_pool(consensus, config)
        pos_coords = select_positives(pools, config.k_positive)
    else:
        # C4 path: re-rank candidates by stability score.
        ref_topk_sets = _precompute_ref_topk_sets(stack, config)
        pos_coords = c4_score_candidates(
            stack, consensus, config, beta, gamma, ref_topk_sets,
        )
        pools = build_consensus_pool(consensus, config)

    return run_config_with_coords(
        predictor, config, pools, pos_coords, target, target_labels,
        target_component_count,
    )


def _build_pair_states(
    query_case_id: str,
    ref_rows: list[dict[str, str]],
    feature_cache: FeatureCache,
) -> list[PairStateR2]:
    """Build pair states for all references; skip failures."""
    pair_states: list[PairStateR2] = []
    for ref_row in ref_rows:
        ref_id = ref_row["reference_case_id"]
        try:
            ps = build_pair_state_r2(query_case_id, ref_id, feature_cache)
            pair_states.append(ps)
        except Exception as exc:
            print(f"[phaseS-TrackC] pair error {query_case_id}/{ref_id}: "
                  f"{type(exc).__name__}: {exc}")
    return pair_states


# ---------------------------------------------------------------------------
# CSV field names
# ---------------------------------------------------------------------------

TRAIN_GRID_FIELDS = [
    "query_case_id", "config_name", "alpha", "beta", "gamma",
    "dice", "iou", "component_recall", "target_component_count",
    "hit_components", "sam_score", "runtime_ms", "status",
    "n_pos_points", "n_neg_points",
]

VAL_TRACKC_FIELDS = [
    "query_case_id", "config_name", "method", "alpha", "beta", "gamma",
    "dice", "iou", "component_recall", "target_component_count",
    "hit_components", "sam_score", "runtime_ms", "status",
    "n_pos_points", "n_neg_points",
]

VAL_COMPARISON_FIELDS = [
    "query_case_id", "dice_trackC", "dice_trackB", "delta",
    "alpha", "beta", "gamma",
]

SEED_CONSISTENCY_FIELDS = [
    "seed", "query_case_id", "config_name", "method",
    "alpha", "beta", "gamma", "dice", "dice_b2mean", "delta", "status",
]

REFCOUNT_STABILITY_FIELDS = [
    "query_case_id", "config_name", "n_refs", "method",
    "alpha", "beta", "gamma", "dice", "dice_b2mean", "delta", "status",
]

LOO_STABILITY_FIELDS = [
    "query_case_id", "config_name", "removed_ref_index", "removed_ref_id",
    "method", "alpha", "beta", "gamma",
    "dice_full", "dice_loo", "drop", "status",
]


# ---------------------------------------------------------------------------
# Phase 1: Train Hyperparameter Selection
# ---------------------------------------------------------------------------

def _grid_row(query_case_id: str, config: PromptConfig, alpha: float, beta: float,
              gamma: float, rec: ConfigRecord) -> dict[str, Any]:
    return {
        "query_case_id": query_case_id,
        "config_name": config.name,
        "alpha": repr(alpha),
        "beta": repr(beta),
        "gamma": repr(gamma),
        "dice": rec.dice if rec.dice is not None else "",
        "iou": rec.iou if rec.iou is not None else "",
        "component_recall": rec.component_recall if rec.component_recall is not None else "",
        "target_component_count": rec.target_component_count,
        "hit_components": rec.hit_components if rec.hit_components is not None else "",
        "sam_score": rec.sam_score if rec.sam_score is not None else "",
        "runtime_ms": rec.runtime_ms,
        "status": rec.status,
        "n_pos_points": rec.n_pos_points,
        "n_neg_points": rec.n_neg_points,
    }


def run_phase1(
    *,
    out_dir: Path,
    feature_cache: FeatureCache,
    predictor: Any,
    seeds: range,
    max_queries: int | None,
    alpha_grid: tuple[float, ...],
    beta_grid: tuple[float, ...],
    gamma_grid: tuple[float, ...],
) -> dict[str, Any]:
    """Phase 1: run SAM-B on train with all (α, β, γ) combos, select best by mean Dice.

    Uses seed=0 only for efficiency (80 queries × 3 configs × 27 combos).
    SAM set_image is called once per query; predict is called 3×27=81 times.
    """
    grid_path = out_dir / "train_grid_results.csv"
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_grid_checkpoint(grid_path)
    grid_handle, grid_writer = _open_csv(grid_path, TRAIN_GRID_FIELDS)

    # Use seed 0 only (first in seeds range).
    seed = list(seeds)[0] if seeds else 0
    queries = queries_from_train_manifests(range(seed, seed + 1))
    if max_queries is not None:
        queries = queries[:max_queries]

    total_sam_calls = 0
    start = time.perf_counter()

    # Accumulate dice per (alpha, beta, gamma) for selection.
    dice_by_combo: dict[tuple[float, float, float], list[float]] = {}
    for a in alpha_grid:
        for b in beta_grid:
            for g in gamma_grid:
                dice_by_combo[(a, b, g)] = []

    for qi, query_case_id in enumerate(queries):
        query_paths = resolve_case_paths(query_case_id)
        query_image = load_rgb_518(query_paths["image"])
        predictor.set_image(np.asarray(query_image))
        target = load_mask_518(query_paths["mask"])
        target_labels = _component_labels(target)
        target_component_count = len(set(target_labels.values()))

        ref_rows = refs_for_query_seed(seed, query_case_id)
        if not ref_rows:
            continue

        for config in FROZEN_CONFIGS:
            pair_states = _build_pair_states(query_case_id, ref_rows, feature_cache)
            if not pair_states:
                continue
            stack = np.stack([ps.contrast for ps in pair_states], axis=0)

            # Precompute per-reference top-K sets once per (query, config).
            ref_topk_sets = _precompute_ref_topk_sets(stack, config)

            for alpha in alpha_grid:
                for beta in beta_grid:
                    for gamma in gamma_grid:
                        ck = (query_case_id, config.name,
                              repr(alpha), repr(beta), repr(gamma))
                        if ck in done:
                            # Load existing dice for selection.
                            continue

                        consensus = c3_gated_consensus(stack, alpha)
                        if beta == 0.0 and gamma == 0.0:
                            pools = build_consensus_pool(consensus, config)
                            pos_coords = select_positives(pools, config.k_positive)
                        else:
                            pos_coords = c4_score_candidates(
                                stack, consensus, config, beta, gamma,
                                ref_topk_sets,
                            )
                            pools = build_consensus_pool(consensus, config)

                        rec = run_config_with_coords(
                            predictor, config, pools, pos_coords, target,
                            target_labels, target_component_count,
                        )
                        if rec.status == "OK":
                            total_sam_calls += 1
                        grid_writer.writerow(
                            _grid_row(query_case_id, config, alpha, beta, gamma, rec)
                        )
                        if rec.dice is not None:
                            dice_by_combo[(alpha, beta, gamma)].append(rec.dice)

        grid_handle.flush()
        elapsed = time.perf_counter() - start
        print(f"[phaseS-TrackC] phase1 query {qi + 1}/{len(queries)} done "
              f"({query_case_id}) sam_calls={total_sam_calls} elapsed={elapsed:.1f}s")

    grid_handle.close()

    # Reload full grid results to get dice for all combos (including cached).
    grid_rows = _read_csv(grid_path)
    for row in grid_rows:
        d = _row_dice(row)
        if d is not None:
            key = (float(row["alpha"]), float(row["beta"]), float(row["gamma"]))
            if key not in dice_by_combo:
                dice_by_combo[key] = []
            # Avoid double-counting (only append if not already there from this run).
            # Since we reload, some may be duplicated; we'll deduplicate by using
            # the reloaded data as the source of truth.
    # Rebuild dice_by_combo from reloaded rows (source of truth).
    dice_by_combo = {}
    for a in alpha_grid:
        for b in beta_grid:
            for g in gamma_grid:
                dice_by_combo[(a, b, g)] = []
    for row in grid_rows:
        d = _row_dice(row)
        if d is not None:
            key = (float(row["alpha"]), float(row["beta"]), float(row["gamma"]))
            if key in dice_by_combo:
                dice_by_combo[key].append(d)

    # Select best (alpha, beta, gamma) by mean Dice.
    best_combo: tuple[float, float, float] = (0.0, 0.0, 0.0)
    best_mean_dice = -1.0
    combo_summary: list[dict[str, Any]] = []
    for a in alpha_grid:
        for b in beta_grid:
            for g in gamma_grid:
                dices = dice_by_combo[(a, b, g)]
                md = _mean(dices) if dices else 0.0
                combo_summary.append({
                    "alpha": a, "beta": b, "gamma": g,
                    "mean_dice": round(md, 6),
                    "n_samples": len(dices),
                })
                if md > best_mean_dice:
                    best_mean_dice = md
                    best_combo = (a, b, g)

    best_alpha, best_beta, best_gamma = best_combo
    sel_payload = {
        "best_alpha": best_alpha,
        "best_beta": best_beta,
        "best_gamma": best_gamma,
        "best_mean_dice": round(best_mean_dice, 6),
        "grid": combo_summary,
        "seed": seed,
        "n_queries": len(queries),
        "n_configs": len(FROZEN_CONFIGS),
        "reasoning": (
            f"Selected (α={best_alpha}, β={best_beta}, γ={best_gamma}) maximising "
            f"mean Dice on train (seed {seed})."
        ),
    }
    (out_dir / "train_selection.json").write_text(
        json.dumps(sel_payload, indent=2), encoding="utf-8"
    )

    elapsed = time.perf_counter() - start
    print(f"[phaseS-TrackC] phase1 done: best=(α={best_alpha}, β={best_beta}, "
          f"γ={best_gamma}) mean_dice={best_mean_dice:.4f} elapsed={elapsed:.1f}s")
    return {
        "query_count": len(queries),
        "best_alpha": best_alpha,
        "best_beta": best_beta,
        "best_gamma": best_gamma,
        "best_mean_dice": best_mean_dice,
        "elapsed_seconds": round(elapsed, 3),
        "total_sam_calls": total_sam_calls,
    }


# ---------------------------------------------------------------------------
# Phase 2: Validation
# ---------------------------------------------------------------------------

def _val_trackc_row(query_case_id: str, config: PromptConfig, alpha: float,
                    beta: float, gamma: float, rec: ConfigRecord) -> dict[str, Any]:
    method = "C3C4" if (alpha != 0.0 or beta != 0.0 or gamma != 0.0) else "B2_mean"
    return {
        "query_case_id": query_case_id,
        "config_name": config.name,
        "method": method,
        "alpha": repr(alpha),
        "beta": repr(beta),
        "gamma": repr(gamma),
        "dice": rec.dice if rec.dice is not None else "",
        "iou": rec.iou if rec.iou is not None else "",
        "component_recall": rec.component_recall if rec.component_recall is not None else "",
        "target_component_count": rec.target_component_count,
        "hit_components": rec.hit_components if rec.hit_components is not None else "",
        "sam_score": rec.sam_score if rec.sam_score is not None else "",
        "runtime_ms": rec.runtime_ms,
        "status": rec.status,
        "n_pos_points": rec.n_pos_points,
        "n_neg_points": rec.n_neg_points,
    }


def run_phase2(
    *,
    out_dir: Path,
    feature_cache: FeatureCache,
    predictor: Any,
    best_alpha: float,
    best_beta: float,
    best_gamma: float,
    configs: tuple[PromptConfig, ...],
    max_queries: int | None,
) -> dict[str, Any]:
    """Phase 2: run SAM-B on validation with selected (α, β, γ)."""
    val_path = out_dir / "val_trackC_consensus_metrics.csv"
    val_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_trackc_checkpoint(val_path)
    val_handle, val_writer = _open_csv(val_path, VAL_TRACKC_FIELDS)

    queries = validation_queries()
    if max_queries is not None:
        queries = queries[:max_queries]

    total_sam_calls = 0
    start = time.perf_counter()

    for qi, query_case_id in enumerate(queries):
        query_paths = resolve_case_paths(query_case_id)
        query_image = load_rgb_518(query_paths["image"])
        predictor.set_image(np.asarray(query_image))
        target = load_mask_518(query_paths["mask"])
        target_labels = _component_labels(target)
        target_component_count = len(set(target_labels.values()))

        ref_rows = refs_for_query_val(query_case_id)
        pair_states = _build_pair_states(query_case_id, ref_rows, feature_cache)

        for config in configs:
            ck = (query_case_id, config.name)
            if ck in done:
                continue

            rec = run_consensus_for_query(
                predictor, config, pair_states, target, target_labels,
                target_component_count,
                alpha=best_alpha, beta=best_beta, gamma=best_gamma,
            )
            if rec.status == "OK":
                total_sam_calls += 1
            val_writer.writerow(
                _val_trackc_row(query_case_id, config, best_alpha, best_beta,
                                best_gamma, rec)
            )
            done.add(ck)

        val_handle.flush()
        elapsed = time.perf_counter() - start
        print(f"[phaseS-TrackC] phase2 query {qi + 1}/{len(queries)} done "
              f"({query_case_id}) sam_calls={total_sam_calls} elapsed={elapsed:.1f}s")

    val_handle.close()
    elapsed = time.perf_counter() - start
    print(f"[phaseS-TrackC] phase2 done: {total_sam_calls} SAM calls, "
          f"{elapsed:.1f}s")
    return {
        "query_count": len(queries),
        "config_count": len(configs),
        "total_sam_calls": total_sam_calls,
        "elapsed_seconds": round(elapsed, 3),
    }


# ---------------------------------------------------------------------------
# Phase 3: Gate C helper runs (cond4, cond5, cond6)
# ---------------------------------------------------------------------------

def run_seed_consistency(
    *,
    predictor: Any,
    feature_cache: FeatureCache,
    best_alpha: float,
    best_beta: float,
    best_gamma: float,
    best_config: PromptConfig,
    train_queries: list[str],
    seeds: list[int],
    out_dir: Path,
) -> list[dict[str, str]]:
    """cond4: run Track C on train with multiple seeds, check direction vs B2_mean.

    Writes seed_consistency.csv and returns the rows.
    """
    path = out_dir / "seed_consistency.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, writer = _open_csv(path, SEED_CONSISTENCY_FIELDS)

    # Load R2 train B2_mean baselines for comparison.
    r2_train_path = PROJECT_ROOT / "reports" / "phaseR_gateR2" / "train_consensus_metrics.csv"
    r2_train_rows = _read_csv(r2_train_path) if r2_train_path.is_file() else []
    b2mean_by_q_seed: dict[tuple[str, int], float] = {}
    for row in r2_train_rows:
        if row.get("method") != "B2_mean":
            continue
        q = row["query_case_id"]
        s = int(row["seed"])
        d = _row_dice(row)
        if d is not None:
            b2mean_by_q_seed[(q, s)] = d

    rows_out: list[dict[str, str]] = []
    for seed in seeds:
        for query_case_id in train_queries:
            query_paths = resolve_case_paths(query_case_id)
            query_image = load_rgb_518(query_paths["image"])
            predictor.set_image(np.asarray(query_image))
            target = load_mask_518(query_paths["mask"])
            target_labels = _component_labels(target)
            target_component_count = len(set(target_labels.values()))

            ref_rows = refs_for_query_seed(seed, query_case_id)
            if not ref_rows:
                continue
            pair_states = _build_pair_states(query_case_id, ref_rows, feature_cache)

            rec = run_consensus_for_query(
                predictor, best_config, pair_states, target, target_labels,
                target_component_count,
                alpha=best_alpha, beta=best_beta, gamma=best_gamma,
            )
            dice_c = rec.dice
            dice_b = b2mean_by_q_seed.get((query_case_id, seed))
            method = "C3C4" if (best_alpha != 0.0 or best_beta != 0.0
                                 or best_gamma != 0.0) else "B2_mean"
            row = {
                "seed": str(seed),
                "query_case_id": query_case_id,
                "config_name": best_config.name,
                "method": method,
                "alpha": repr(best_alpha),
                "beta": repr(best_beta),
                "gamma": repr(best_gamma),
                "dice": repr(dice_c) if dice_c is not None else "",
                "dice_b2mean": repr(dice_b) if dice_b is not None else "",
                "delta": repr(dice_c - dice_b) if (dice_c is not None
                                                    and dice_b is not None) else "",
                "status": rec.status,
            }
            writer.writerow(row)
            rows_out.append(row)
        handle.flush()
    handle.close()
    return rows_out


def run_refcount_stability(
    *,
    predictor: Any,
    feature_cache: FeatureCache,
    best_alpha: float,
    best_beta: float,
    best_gamma: float,
    best_config: PromptConfig,
    val_queries: list[str],
    out_dir: Path,
) -> list[dict[str, str]]:
    """cond5: run Track C with k=3 and k=6 references, check direction stable.

    Also runs B2_mean (α=β=γ=0) with the same reference counts for comparison.
    Writes refcount_stability.csv and returns the rows.
    """
    path = out_dir / "refcount_stability.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, writer = _open_csv(path, REFCOUNT_STABILITY_FIELDS)

    rows_out: list[dict[str, str]] = []
    for query_case_id in val_queries:
        query_paths = resolve_case_paths(query_case_id)
        query_image = load_rgb_518(query_paths["image"])
        predictor.set_image(np.asarray(query_image))
        target = load_mask_518(query_paths["mask"])
        target_labels = _component_labels(target)
        target_component_count = len(set(target_labels.values()))

        ref_rows = refs_for_query_val(query_case_id)
        full_pair_states = _build_pair_states(query_case_id, ref_rows, feature_cache)

        for n_refs in (6, 3):
            pair_states = full_pair_states[:n_refs]
            if not pair_states:
                continue

            # Track C with selected hyperparameters.
            rec_c = run_consensus_for_query(
                predictor, best_config, pair_states, target, target_labels,
                target_component_count,
                alpha=best_alpha, beta=best_beta, gamma=best_gamma,
            )
            # B2_mean baseline (α=β=γ=0) with the same reference count.
            rec_b = run_consensus_for_query(
                predictor, best_config, pair_states, target, target_labels,
                target_component_count,
                alpha=0.0, beta=0.0, gamma=0.0,
            )
            dice_c = rec_c.dice
            dice_b = rec_b.dice
            method = "C3C4" if (best_alpha != 0.0 or best_beta != 0.0
                                 or best_gamma != 0.0) else "B2_mean"
            row = {
                "query_case_id": query_case_id,
                "config_name": best_config.name,
                "n_refs": str(n_refs),
                "method": method,
                "alpha": repr(best_alpha),
                "beta": repr(best_beta),
                "gamma": repr(best_gamma),
                "dice": repr(dice_c) if dice_c is not None else "",
                "dice_b2mean": repr(dice_b) if dice_b is not None else "",
                "delta": repr(dice_c - dice_b) if (dice_c is not None
                                                    and dice_b is not None) else "",
                "status": rec_c.status,
            }
            writer.writerow(row)
            rows_out.append(row)
        handle.flush()
    handle.close()
    return rows_out


def run_loo_stability(
    *,
    predictor: Any,
    feature_cache: FeatureCache,
    best_alpha: float,
    best_beta: float,
    best_gamma: float,
    best_config: PromptConfig,
    val_queries: list[str],
    full_ref_dice: dict[str, float],
    out_dir: Path,
) -> list[dict[str, str]]:
    """cond6: leave-one-reference-out on val queries, check no crash / no big drop.

    Writes loo_stability.csv and returns the rows.
    """
    path = out_dir / "loo_stability.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, writer = _open_csv(path, LOO_STABILITY_FIELDS)

    rows_out: list[dict[str, str]] = []
    method = "C3C4" if (best_alpha != 0.0 or best_beta != 0.0
                         or best_gamma != 0.0) else "B2_mean"
    for query_case_id in val_queries:
        query_paths = resolve_case_paths(query_case_id)
        query_image = load_rgb_518(query_paths["image"])
        predictor.set_image(np.asarray(query_image))
        target = load_mask_518(query_paths["mask"])
        target_labels = _component_labels(target)
        target_component_count = len(set(target_labels.values()))

        ref_rows = refs_for_query_val(query_case_id)
        full_pair_states = _build_pair_states(query_case_id, ref_rows, feature_cache)
        dice_full = full_ref_dice.get(query_case_id)

        for loo_idx in range(len(full_pair_states)):
            loo_states = [
                ps for i, ps in enumerate(full_pair_states) if i != loo_idx
            ]
            if not loo_states:
                continue
            removed_ref_id = ref_rows[loo_idx]["reference_case_id"] if loo_idx < len(ref_rows) else ""
            rec = run_consensus_for_query(
                predictor, best_config, loo_states, target, target_labels,
                target_component_count,
                alpha=best_alpha, beta=best_beta, gamma=best_gamma,
            )
            dice_loo = rec.dice
            drop = (dice_full - dice_loo) if (dice_full is not None
                                               and dice_loo is not None) else None
            row = {
                "query_case_id": query_case_id,
                "config_name": best_config.name,
                "removed_ref_index": str(loo_idx),
                "removed_ref_id": removed_ref_id,
                "method": method,
                "alpha": repr(best_alpha),
                "beta": repr(best_beta),
                "gamma": repr(best_gamma),
                "dice_full": repr(dice_full) if dice_full is not None else "",
                "dice_loo": repr(dice_loo) if dice_loo is not None else "",
                "drop": repr(drop) if drop is not None else "",
                "status": rec.status,
            }
            writer.writerow(row)
            rows_out.append(row)
        handle.flush()
    handle.close()
    return rows_out


# ---------------------------------------------------------------------------
# Phase 3: Gate C Evaluation
# ---------------------------------------------------------------------------

def compute_gate_c(
    *,
    trackc_rows: list[dict[str, str]],
    trackb_rows: list[dict[str, str]],
    val_consensus_rows: list[dict[str, str]],
    best_config: str,
    best_alpha: float,
    best_beta: float,
    best_gamma: float,
    train_grid_rows: list[dict[str, str]],
    seed_consistency_rows: list[dict[str, str]],
    refcount_stability_rows: list[dict[str, str]],
    loo_stability_rows: list[dict[str, str]],
    phase2_runtime_seconds: float,
) -> dict[str, Any]:
    """Evaluate Gate C (§8.4): 8 conditions relative to Track B best."""
    # --- Track C per-query Dice (best config, selected hyperparameters) ---
    trackc_by_q: dict[str, float] = {}
    for r in trackc_rows:
        if r["config_name"] != best_config:
            continue
        d = _row_dice(r)
        if d is not None:
            trackc_by_q[r["query_case_id"]] = d

    # --- Track B weighted per-query Dice (best config = baseline) ---
    trackb_by_q: dict[str, float] = {}
    for r in trackb_rows:
        if r["config_name"] != best_config:
            continue
        d = _row_dice(r)
        if d is not None:
            trackb_by_q[r["query_case_id"]] = d

    # --- B2_mean uniform per-query Dice (best config) ---
    b2mean_by_q: dict[str, float] = {}
    for r in val_consensus_rows:
        if (r.get("method") != "B2_mean" or r.get("config_name") != best_config
                or r.get("reference_scope", "all6") != "all6"):
            continue
        d = _row_dice(r)
        if d is not None:
            b2mean_by_q[r["query_case_id"]] = d

    common_q = sorted(q for q in (set(trackc_by_q) & set(trackb_by_q)))
    trackc_dices = [trackc_by_q[q] for q in common_q]
    trackb_dices = [trackb_by_q[q] for q in common_q]
    deltas = [trackc_by_q[q] - trackb_by_q[q] for q in common_q]

    mean_c = _mean(trackc_dices)
    mean_b = _mean(trackb_dices)
    delta_dice = mean_c - mean_b

    # cond1: Mean Dice +0.005 relative to Track B best.
    cond1 = delta_dice >= 0.005

    # cond2: CVaR +0.01 OR reference std further reduction ≥10% (vs Track B).
    cvar_c = _cvar(trackc_dices, 0.1)
    cvar_b = _cvar(trackb_dices, 0.1)
    cvar_improvement = cvar_c - cvar_b
    # Reference std: Track C's per-query std across references isn't directly
    # available (Track C doesn't produce per-reference Dice). Use per-query Dice
    # std across the 3 configs as a proxy, or check CVaR improvement.
    # Per the task: "CVaR +0.01, OR reference std further reduction ≥10% (vs Track B)"
    # Since Track C doesn't have per-reference weights, use CVaR as primary.
    cond2 = (cvar_improvement >= 0.01)

    # cond3: Catastrophic failure not increase (vs Track B).
    cfr_c = (sum(1 for d in trackc_dices if d < CATASTROPHIC_DICE_THRESHOLD)
             / len(trackc_dices)) if trackc_dices else 0.0
    cfr_b = (sum(1 for d in trackb_dices if d < CATASTROPHIC_DICE_THRESHOLD)
             / len(trackb_dices)) if trackb_dices else 0.0
    cond3 = cfr_c <= cfr_b

    # cond4: 3 seeds consistent (Track C beats B2_mean in same direction ≥2/3 seeds).
    seed_results: dict[int, list[float]] = {}
    seed_b2mean: dict[int, list[float]] = {}
    for row in seed_consistency_rows:
        if row["config_name"] != best_config:
            continue
        seed = int(row["seed"])
        d = _safe_float(row.get("dice"))
        db = _safe_float(row.get("dice_b2mean"))
        if d is not None:
            seed_results.setdefault(seed, []).append(d)
        if db is not None:
            seed_b2mean.setdefault(seed, []).append(db)
    seeds_consistent = 0
    seeds_total = 0
    for seed in sorted(set(list(seed_results.keys()) + list(seed_b2mean.keys()))):
        tc = _mean(seed_results.get(seed, []))
        tb = _mean(seed_b2mean.get(seed, []))
        if seed_results.get(seed) and seed_b2mean.get(seed):
            seeds_total += 1
            if tc > tb:
                seeds_consistent += 1
    cond4 = seeds_total > 0 and seeds_consistent >= math.ceil(2 / 3 * seeds_total)

    # cond5: Reference count variation stable (k=3 vs k=6 direction stable).
    refcount_by_q_n: dict[tuple[str, int], float] = {}
    refcount_b2_by_q_n: dict[tuple[str, int], float] = {}
    for row in refcount_stability_rows:
        if row["config_name"] != best_config:
            continue
        q = row["query_case_id"]
        n = int(row["n_refs"])
        d = _safe_float(row.get("dice"))
        db = _safe_float(row.get("dice_b2mean"))
        if d is not None:
            refcount_by_q_n[(q, n)] = d
        if db is not None:
            refcount_b2_by_q_n[(q, n)] = db
    # Compare direction for k=3 and k=6.
    deltas_k6 = [refcount_by_q_n.get((q, 6), 0.0) - refcount_b2_by_q_n.get((q, 6), 0.0)
                 for q in {q for q, _ in refcount_by_q_n} if (q, 6) in refcount_by_q_n]
    deltas_k3 = [refcount_by_q_n.get((q, 3), 0.0) - refcount_b2_by_q_n.get((q, 3), 0.0)
                 for q in {q for q, _ in refcount_by_q_n} if (q, 3) in refcount_by_q_n]
    dir_k6 = _mean(deltas_k6) > 0 if deltas_k6 else False
    dir_k3 = _mean(deltas_k3) > 0 if deltas_k3 else False
    cond5 = (dir_k6 == dir_k3) if (deltas_k6 and deltas_k3) else False

    # cond6: Leave-one-reference-out not crash (drop < 0.05 for 5 val queries).
    loo_crashes = 0
    loo_big_drops = 0
    loo_total = 0
    for row in loo_stability_rows:
        if row["config_name"] != best_config:
            continue
        loo_total += 1
        status = row.get("status", "")
        if not status.startswith("OK"):
            loo_crashes += 1
            continue
        drop = _safe_float(row.get("drop"))
        if drop is not None and drop > 0.05:
            loo_big_drops += 1
    cond6 = (loo_crashes == 0 and loo_big_drops == 0) if loo_total > 0 else False

    # cond7: Params and runtime acceptable.
    # Track C has 0 learned params (C3/C4 are parameter-free given α, β, γ).
    # 1 SAM call per (query, config). Runtime < 2x Track B.
    n_params = 0
    one_sam_call = True
    runtime_ok = phase2_runtime_seconds < 2.0 * TRACK_B_PHASE2_RUNTIME_SECONDS
    cond7 = (n_params < 5_000_000) and one_sam_call and runtime_ok

    # cond8: Not just high-capacity train fitting (train Dice - val Dice < 0.05).
    train_dices = []
    for row in train_grid_rows:
        if (row["config_name"] != best_config
                or float(row["alpha"]) != best_alpha
                or float(row["beta"]) != best_beta
                or float(row["gamma"]) != best_gamma):
            continue
        d = _row_dice(row)
        if d is not None:
            train_dices.append(d)
    mean_train = _mean(train_dices) if train_dices else 0.0
    mean_val = mean_c
    train_val_gap = mean_train - mean_val
    cond8 = train_val_gap < 0.05

    conditions = {
        "cond1_delta_dice_ge_0p005_vs_trackB": bool(cond1),
        "cond2_cvar_improvement_ge_0p01": bool(cond2),
        "cond3_catastrophic_failure_not_increase": bool(cond3),
        "cond4_3_seeds_consistent": bool(cond4),
        "cond5_refcount_variation_stable": bool(cond5),
        "cond6_loo_no_crash": bool(cond6),
        "cond7_params_runtime_acceptable": bool(cond7),
        "cond8_train_val_gap_lt_0p05": bool(cond8),
    }

    n_pass = sum(1 for v in conditions.values() if v)
    cond1_pass = conditions["cond1_delta_dice_ge_0p005_vs_trackB"]
    cond2_pass = conditions["cond2_cvar_improvement_ge_0p01"]

    # Decision: PASS if cond1 AND cond2 AND at least 4/8 total conditions pass.
    gate_pass = cond1_pass and cond2_pass and (n_pass >= 4)
    decision = "PHASES_TRACK_C_STABLE_CANDIDATE_PASS" if gate_pass else "PHASES_TRACK_C_NO_GAIN"

    boot = _paired_bootstrap(deltas)

    # --- val_comparison.csv payload ---
    comparison: list[dict[str, Any]] = []
    for q in common_q:
        comparison.append({
            "query_case_id": q,
            "dice_trackC": round(trackc_by_q[q], 6),
            "dice_trackB": round(trackb_by_q[q], 6),
            "delta": round(trackc_by_q[q] - trackb_by_q[q], 6),
            "alpha": repr(best_alpha),
            "beta": repr(best_beta),
            "gamma": repr(best_gamma),
        })

    return {
        "best_config": best_config,
        "best_alpha": best_alpha,
        "best_beta": best_beta,
        "best_gamma": best_gamma,
        "query_count": len(common_q),
        "mean_trackC_dice": round(mean_c, 6),
        "mean_trackB_dice": round(mean_b, 6),
        "delta_dice_vs_trackB": round(delta_dice, 6),
        "mean_b2mean_dice": round(_mean(list(b2mean_by_q.values())), 6)
            if b2mean_by_q else 0.0,
        "cvar_0p1_trackC": round(cvar_c, 6),
        "cvar_0p1_trackB": round(cvar_b, 6),
        "cvar_improvement": round(cvar_improvement, 6),
        "cfr_trackC": round(cfr_c, 6),
        "cfr_trackB": round(cfr_b, 6),
        "mean_train_dice": round(mean_train, 6),
        "train_val_gap": round(train_val_gap, 6),
        "seed_consistency": {
            "seeds_total": seeds_total,
            "seeds_consistent": seeds_consistent,
        },
        "refcount_direction": {
            "k6_positive": bool(dir_k6),
            "k3_positive": bool(dir_k3),
        },
        "loo_stability": {
            "total": loo_total,
            "crashes": loo_crashes,
            "big_drops": loo_big_drops,
        },
        "phase2_runtime_seconds": round(phase2_runtime_seconds, 3),
        "trackB_phase2_runtime_seconds": TRACK_B_PHASE2_RUNTIME_SECONDS,
        "n_learned_params": n_params,
        "bootstrap": {k: round(v, 6) for k, v in boot.items()},
        "conditions": conditions,
        "n_conditions_pass": n_pass,
        "gate_pass": bool(gate_pass),
        "decision": decision,
        "comparison": comparison,
    }


def write_gate_c_summary(gate: dict[str, Any], out_dir: Path) -> None:
    json_path = out_dir / "gateC_summary.json"
    json_path.write_text(json.dumps({"gate_c": gate}, indent=2), encoding="utf-8")

    md: list[str] = []
    md.append("# Phase S Track C: Stable Candidate Scoring — Gate C Summary")
    md.append("")
    md.append(f"Date: {datetime.now().isoformat(timespec='seconds')}")
    md.append(f"Validation queries: {gate['query_count']}")
    md.append(f"Best config: {gate['best_config']}")
    md.append(f"Hyperparameters: α={gate['best_alpha']}, β={gate['best_beta']}, "
              f"γ={gate['best_gamma']}")
    md.append("")
    md.append("## Track C vs Track B (baseline)")
    md.append("")
    md.append("| Metric | Track C | Track B |")
    md.append("|---|---:|---:|")
    md.append(f"| Mean Dice | {gate['mean_trackC_dice']:.4f} | "
              f"{gate['mean_trackB_dice']:.4f} |")
    md.append(f"| CVaR_0.1 | {gate['cvar_0p1_trackC']:.4f} | "
              f"{gate['cvar_0p1_trackB']:.4f} |")
    md.append(f"| CFR (Dice<0.1) | {gate['cfr_trackC']:.4f} | "
              f"{gate['cfr_trackB']:.4f} |")
    md.append("")
    md.append(f"Delta Dice vs Track B: {gate['delta_dice_vs_trackB']:.4f}")
    md.append(f"CVaR improvement: {gate['cvar_improvement']:.4f}")
    md.append(f"Mean train Dice: {gate['mean_train_dice']:.4f} "
              f"(train-val gap: {gate['train_val_gap']:.4f})")
    md.append(f"Bootstrap mean delta: {gate['bootstrap']['mean']:.4f} "
              f"(CI [{gate['bootstrap']['ci_lower']:.4f}, "
              f"{gate['bootstrap']['ci_upper']:.4f}])")
    md.append("")
    md.append("## Gate C Conditions (§8.4)")
    md.append("")
    md.append("| # | Condition | Pass |")
    md.append("|---|---|---|")
    cond_labels = {
        "cond1_delta_dice_ge_0p005_vs_trackB": "Mean Dice +0.005 vs Track B",
        "cond2_cvar_improvement_ge_0p01": "CVaR +0.01 vs Track B",
        "cond3_catastrophic_failure_not_increase": "Catastrophic failure not increase",
        "cond4_3_seeds_consistent": "3 seeds consistent (≥2/3 beat B2_mean)",
        "cond5_refcount_variation_stable": "Reference count variation stable (k=3 vs k=6)",
        "cond6_loo_no_crash": "LOO no crash / no big drop (<0.05)",
        "cond7_params_runtime_acceptable": "Params <5M, 1 SAM call, <2x Track B runtime",
        "cond8_train_val_gap_lt_0p05": "Train-val gap < 0.05",
    }
    for i, key in enumerate(cond_labels, start=1):
        md.append(f"| {i} | {cond_labels[key]} | "
                  f"{'YES' if gate['conditions'][key] else 'NO'} |")
    md.append("")
    md.append(f"Conditions passed: {gate['n_conditions_pass']}/8")
    md.append(f"Gate pass: {gate['gate_pass']}")
    md.append(f"## Decision: **{gate['decision']}**")
    md.append("")
    (out_dir / "gateC_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def write_val_comparison(gate: dict[str, Any], out_dir: Path) -> None:
    path = out_dir / "val_comparison.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=VAL_COMPARISON_FIELDS)
        writer.writeheader()
        for row in gate["comparison"]:
            writer.writerow(row)


def write_run_metadata(
    out_dir: Path, *, device: torch.device, phases_run: list[int],
    phase1_meta: dict[str, Any] | None, phase2_meta: dict[str, Any] | None,
    phase3_meta: dict[str, Any] | None, gate: dict[str, Any] | None,
) -> None:
    elapsed = 0.0
    total_sam = 0
    if phase1_meta:
        elapsed += phase1_meta.get("elapsed_seconds", 0.0)
        total_sam += phase1_meta.get("total_sam_calls", 0)
    if phase2_meta:
        elapsed += phase2_meta.get("elapsed_seconds", 0.0)
        total_sam += phase2_meta.get("total_sam_calls", 0)
    if phase3_meta:
        elapsed += phase3_meta.get("elapsed_seconds", 0.0)
        total_sam += phase3_meta.get("total_sam_calls", 0)
    gpu_hours = (elapsed / 3600.0) if str(device).startswith("cuda") else 0.0
    metadata = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "phases_run": phases_run,
        "runtime_seconds": round(elapsed, 3),
        "gpu_hours": round(gpu_hours, 6),
        "total_sam_calls": total_sam,
        "phase1_query_count": (phase1_meta or {}).get("query_count", 0),
        "phase2_query_count": (phase2_meta or {}).get("query_count", 0),
        "gate_c_decision": (gate or {}).get("decision"),
        "gate_c_pass": (gate or {}).get("gate_pass"),
        "best_alpha": (gate or {}).get("best_alpha"),
        "best_beta": (gate or {}).get("best_beta"),
        "best_gamma": (gate or {}).get("best_gamma"),
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_selection(out_dir: Path) -> dict[str, Any]:
    path = out_dir / "train_selection.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"train_selection.json not found at {path}; run Phase 1 first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase S Track C: Stable Candidate Scoring and Pixel-wise Agreement Gate"
    )
    parser.add_argument("--phase", choices=["1", "2", "3", "all"], default="all",
                        help="Which phase(s) to run (default: all).")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick verification: 2 train + 2 val, reduced grid.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--val-configs", choices=["all", "best"], default="all",
                        help="Phase 2 configs: all 3 or only best (P30-N0).")
    parser.add_argument(
        "--out-dir", type=Path,
        default=PROJECT_ROOT / "reports" / "phaseS_trackC",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        seeds = range(1)  # seed 0 only
        train_max = 2
        val_max = 2
        alpha_grid = ALPHA_GRID_SMOKE
        beta_grid = BETA_GRID_SMOKE
        gamma_grid = GAMMA_GRID_SMOKE
        loo_queries_max = 2
        cond4_seeds = [0, 1, 2]
    else:
        seeds = range(SEED_COUNT)
        train_max = None
        val_max = None
        alpha_grid = ALPHA_GRID
        beta_grid = BETA_GRID
        gamma_grid = GAMMA_GRID
        loo_queries_max = 5
        cond4_seeds = [0, 1, 2]

    device = _resolve_device(args.device)
    phases_run: list[int] = []
    run_phase1_flag = args.phase in ("1", "all")
    run_phase2_flag = args.phase in ("2", "all")
    run_phase3_flag = args.phase in ("3", "all")

    print(f"[phaseS-TrackC] device={device} seeds={list(seeds)} smoke={args.smoke} "
          f"phase={args.phase} val_configs={args.val_configs} out_dir={out_dir}")
    print(f"[phaseS-TrackC] grid: α={alpha_grid} β={beta_grid} γ={gamma_grid} "
          f"({len(alpha_grid)*len(beta_grid)*len(gamma_grid)} combos)")

    feature_cache = FeatureCache(device=device)

    best_alpha = 0.0
    best_beta = 0.0
    best_gamma = 0.0
    phase1_meta: dict[str, Any] | None = None
    phase2_meta: dict[str, Any] | None = None

    # --- Phase 1 ---
    if run_phase1_flag:
        phases_run.append(1)
        predictor = load_sam_predictor(device)
        phase1_meta = run_phase1(
            out_dir=out_dir,
            feature_cache=feature_cache,
            predictor=predictor,
            seeds=seeds,
            max_queries=train_max,
            alpha_grid=alpha_grid,
            beta_grid=beta_grid,
            gamma_grid=gamma_grid,
        )
        best_alpha = phase1_meta["best_alpha"]
        best_beta = phase1_meta["best_beta"]
        best_gamma = phase1_meta["best_gamma"]
        del predictor  # free SAM model memory before reloading in Phase 2
    else:
        # Load selection from Phase 1.
        selection = _load_selection(out_dir)
        best_alpha = selection["best_alpha"]
        best_beta = selection["best_beta"]
        best_gamma = selection["best_gamma"]

    # --- Phase 2 ---
    if run_phase2_flag:
        phases_run.append(2)
        if args.val_configs == "best":
            configs = tuple(c for c in FROZEN_CONFIGS if c.name == "P30-N0")
        else:
            configs = FROZEN_CONFIGS
        predictor = load_sam_predictor(device)
        phase2_meta = run_phase2(
            out_dir=out_dir,
            feature_cache=feature_cache,
            predictor=predictor,
            best_alpha=best_alpha,
            best_beta=best_beta,
            best_gamma=best_gamma,
            configs=configs,
            max_queries=val_max,
        )
        print(f"[phaseS-TrackC] phase2 done: {phase2_meta['total_sam_calls']} SAM calls, "
              f"{phase2_meta['elapsed_seconds']}s")
        del predictor

    # --- Phase 3 ---
    gate: dict[str, Any] | None = None
    phase3_meta: dict[str, Any] | None = None
    if run_phase3_flag:
        phases_run.append(3)
        best_config_name = "P30-N0"
        best_config = next(c for c in FROZEN_CONFIGS if c.name == best_config_name)

        # Load CSV data for gate evaluation.
        trackc_rows = _read_csv(out_dir / "val_trackC_consensus_metrics.csv")
        trackb_rows = _read_csv(
            PROJECT_ROOT / "reports" / "phaseS_trackB" / "val_weighted_consensus_metrics.csv"
        )
        val_consensus_rows = _read_csv(
            PROJECT_ROOT / "reports" / "phaseR_gateR2" / "val_consensus_metrics.csv"
        )
        train_grid_rows = _read_csv(out_dir / "train_grid_results.csv")

        # Run cond4/5/6 SAM calls.
        predictor = load_sam_predictor(device)
        p3_start = time.perf_counter()
        p3_sam = 0

        # cond4: 3-seed consistency on train.
        train_queries = queries_from_train_manifests(range(1))  # seed 0 queries
        if args.smoke:
            train_queries = train_queries[:2]
        else:
            train_queries = train_queries[:20]  # subset for efficiency
        seed_rows = run_seed_consistency(
            predictor=predictor,
            feature_cache=feature_cache,
            best_alpha=best_alpha,
            best_beta=best_beta,
            best_gamma=best_gamma,
            best_config=best_config,
            train_queries=train_queries,
            seeds=cond4_seeds,
            out_dir=out_dir,
        )
        p3_sam += sum(1 for r in seed_rows if r.get("status", "").startswith("OK"))

        # cond5: reference count variation on val.
        val_queries = validation_queries()
        if args.smoke:
            val_queries = val_queries[:2]
        else:
            val_queries = val_queries[:10]
        refcount_rows = run_refcount_stability(
            predictor=predictor,
            feature_cache=feature_cache,
            best_alpha=best_alpha,
            best_beta=best_beta,
            best_gamma=best_gamma,
            best_config=best_config,
            val_queries=val_queries,
            out_dir=out_dir,
        )
        p3_sam += sum(1 for r in refcount_rows if r.get("status", "").startswith("OK"))

        # cond6: LOO on val queries (subset).
        loo_queries = validation_queries()[:loo_queries_max]
        # Get full-reference Track C dice from Phase 2 CSV.
        full_ref_dice: dict[str, float] = {}
        for r in trackc_rows:
            if r["config_name"] == best_config_name:
                d = _row_dice(r)
                if d is not None:
                    full_ref_dice[r["query_case_id"]] = d
        loo_rows = run_loo_stability(
            predictor=predictor,
            feature_cache=feature_cache,
            best_alpha=best_alpha,
            best_beta=best_beta,
            best_gamma=best_gamma,
            best_config=best_config,
            val_queries=loo_queries,
            full_ref_dice=full_ref_dice,
            out_dir=out_dir,
        )
        p3_sam += sum(1 for r in loo_rows if r.get("status", "").startswith("OK"))

        del predictor
        p3_elapsed = time.perf_counter() - p3_start
        phase3_meta = {
            "elapsed_seconds": round(p3_elapsed, 3),
            "total_sam_calls": p3_sam,
        }

        phase2_runtime = (phase2_meta or {}).get("elapsed_seconds", 0.0)
        gate = compute_gate_c(
            trackc_rows=trackc_rows,
            trackb_rows=trackb_rows,
            val_consensus_rows=val_consensus_rows,
            best_config=best_config_name,
            best_alpha=best_alpha,
            best_beta=best_beta,
            best_gamma=best_gamma,
            train_grid_rows=train_grid_rows,
            seed_consistency_rows=seed_rows,
            refcount_stability_rows=refcount_rows,
            loo_stability_rows=loo_rows,
            phase2_runtime_seconds=phase2_runtime,
        )
        write_gate_c_summary(gate, out_dir)
        write_val_comparison(gate, out_dir)
        print(f"[phaseS-TrackC] decision: {gate['decision']} "
              f"(pass={gate['gate_pass']})")

    write_run_metadata(
        out_dir, device=device, phases_run=phases_run,
        phase1_meta=phase1_meta, phase2_meta=phase2_meta,
        phase3_meta=phase3_meta, gate=gate,
    )
    print(f"[phaseS-TrackC] artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
