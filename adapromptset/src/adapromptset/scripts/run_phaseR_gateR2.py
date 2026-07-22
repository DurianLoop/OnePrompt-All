"""Phase R Gate R2: Deterministic Consensus Feasibility.

Gate R1 confirmed reference instability (REFERENCE_INSTABILITY confirmed on
train). Gate R2 tests whether deterministic consensus methods (B1-B6) can
reduce reference-induced variance while preserving (or improving) Dice and
keeping the budget to a single SAM call.

For each (query, seed) on train the script aggregates the per-reference
contrast maps ``C_r(x) = zscore(s_fg_r) - zscore(s_bg_r)`` across the 6
references using several deterministic operators (mean, median, trimmed mean,
disagreement-penalized, quantile), builds a candidate pool from the aggregated
map, and runs SAM-B once per consensus variant. B1 (random single reference)
reuses R1's per-reference Dice to avoid redundant SAM calls. Hyperparameters
(best config, method, beta, tau) are selected on train, frozen, and evaluated
on the validation split.

Outputs (under ``reports/phaseR_gateR2/``)::

    train_consensus_metrics.csv     (query x seed x config x method x hyperparam)
    train_selection_summary.json    (best config/method/beta/tau on train)
    val_consensus_metrics.csv       (query x config x method x scope)
    val_per_reference_metrics.csv   (query x config x reference for B0)
    val_cross_set_stability.csv     (query x config x method A/B comparison)
    gateR2_summary.json             (Gate R2 decision, 8 conditions)
    gateR2_summary.md               (human-readable summary)
    run_metadata.json               (device, runtime, gpu_hours, total_sam_calls)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
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

# Gate R2 specific constants.
BETA_GRID = (0.25, 0.5, 1.0)  # for B5 Disagreement-Penalized
TAU_GRID = (0.50, 0.67, 0.83)  # for B6 Quantile
RANDOM_SINGLE_SEED = 20260722  # for B1 random reference selection
N_BOOTSTRAP = 10000  # for paired bootstrap CI


# ---------------------------------------------------------------------------
# Frozen prompt configs (only 3, not 210)
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
# Case ID -> path resolution (copied from R1)
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
# Image / mask loading (copied from R1)
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
# Component labels for recall computation (copied from R1)
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
# SAM predictor (copied from R1)
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
# DINOv2 feature cache (copied from R1)
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
# Point selection (copied from R1, trimmed to P_TOPK / N_BOTTOMK / N_NONE)
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
        # First K = most confident background (lowest foreground score)
        return neg_coords[:k]
    raise ValueError(f"unknown negative selector: {selector}")


# ---------------------------------------------------------------------------
# Config execution: SAM predict + metrics (copied from R1)
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


# ---------------------------------------------------------------------------
# CSV helpers (copied from R1)
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
# Manifest loading
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
# Contrast maps + per-reference pair state for R2
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


# ---------------------------------------------------------------------------
# Consensus aggregation + pool building
# ---------------------------------------------------------------------------

def build_consensus_pool(contrast_map: np.ndarray, config: PromptConfig) -> CandidatePools:
    """Build candidate pool from an aggregated contrast map.

    The aggregated map is passed as ``foreground`` and a zero grid as
    ``background``. Inside ``build_candidate_pools`` ``_contrast_grid`` z-scores
    the foreground (preserving ranking) and the zero background yields zeros.
    ``config`` is unused for pool construction (pools are config-independent,
    selected via ``select_positives``/``select_negatives``) but retained to match
    the directive signature.
    """
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


def _aggregate(stacked: np.ndarray, method: str, hyperparam: str) -> np.ndarray:
    """Aggregate a (m, H, W) stack of contrast maps for one consensus variant."""
    if method == "B2_mean":
        return np.mean(stacked, axis=0)
    if method == "B3_median":
        return np.median(stacked, axis=0)
    if method == "B4_trimmed":
        m = stacked.shape[0]
        if m >= 5:
            # remove max and min per position, then mean over the rest
            return np.sort(stacked, axis=0)[1:-1].mean(axis=0)
        # <= 3 refs: trimmed mean removes 1 max + 1 min -> single value == median
        return np.median(stacked, axis=0)
    if method == "B5_disagreement":
        beta = float(hyperparam)
        mu = stacked.mean(axis=0)
        sigma = stacked.std(axis=0)  # ddof=0 population std
        return mu - beta * sigma
    if method == "B6_quantile":
        tau = float(hyperparam)
        return np.quantile(stacked, tau, axis=0)
    raise ValueError(f"unknown consensus method: {method}")


def _train_consensus_variants(stacked: np.ndarray):
    """Yield (method, hyperparam, agg_map) for all train consensus variants."""
    yield ("B2_mean", "", np.mean(stacked, axis=0))
    yield ("B3_median", "", np.median(stacked, axis=0))
    m = stacked.shape[0]
    if m >= 5:
        yield ("B4_trimmed", "", np.sort(stacked, axis=0)[1:-1].mean(axis=0))
    else:
        yield ("B4_trimmed", "", np.median(stacked, axis=0))
    mu = stacked.mean(axis=0)
    sigma = stacked.std(axis=0)
    for beta in BETA_GRID:
        yield ("B5_disagreement", str(beta), mu - float(beta) * sigma)
    for tau in TAU_GRID:
        yield ("B6_quantile", str(tau), np.quantile(stacked, float(tau), axis=0))


def _val_consensus_variants(stacked: np.ndarray, best_beta: float, best_tau: float):
    """Yield (method, hyperparam, agg_map) for validation (frozen beta/tau)."""
    yield ("B2_mean", "", np.mean(stacked, axis=0))
    yield ("B3_median", "", np.median(stacked, axis=0))
    m = stacked.shape[0]
    if m >= 5:
        yield ("B4_trimmed", "", np.sort(stacked, axis=0)[1:-1].mean(axis=0))
    else:
        yield ("B4_trimmed", "", np.median(stacked, axis=0))
    mu = stacked.mean(axis=0)
    sigma = stacked.std(axis=0)
    yield ("B5_disagreement", str(best_beta), mu - float(best_beta) * sigma)
    yield ("B6_quantile", str(best_tau), np.quantile(stacked, float(best_tau), axis=0))


# ---------------------------------------------------------------------------
# B1 random single reference selection (deterministic across runs)
# ---------------------------------------------------------------------------

def _b1_index(query_case_id: str, seed: int, n_refs: int) -> int:
    """Deterministic index of the random single reference for B1.

    Uses a stable SHA-256 hash of the query id (rather than Python's
    process-randomized ``hash``) so the selection reproduces across runs and
    supports reliable checkpointing.
    """
    digest = int(hashlib.sha256(query_case_id.encode("utf-8")).hexdigest(), 16)
    rng = random.Random(RANDOM_SINGLE_SEED + (digest % (2**31)) + seed)
    return rng.randrange(n_refs)


def _b1_record_from_r1(r1_row: dict[str, str], config: PromptConfig) -> ConfigRecord:
    """Reconstruct a ConfigRecord from R1's per-reference row (no new SAM call)."""
    del config  # k_positive/k_negative already encoded in R1's row
    dice_str = r1_row.get("dice", "")
    iou_str = r1_row.get("iou", "")
    cr_str = r1_row.get("component_recall", "")
    hc_str = r1_row.get("hit_components", "")
    ss_str = r1_row.get("sam_score", "")
    return ConfigRecord(
        dice=float(dice_str) if dice_str != "" else None,
        iou=float(iou_str) if iou_str != "" else None,
        component_recall=float(cr_str) if cr_str != "" else None,
        target_component_count=int(r1_row.get("target_component_count", "0") or "0"),
        hit_components=int(hc_str) if hc_str != "" else None,
        sam_score=float(ss_str) if ss_str != "" else None,
        runtime_ms=0.0,
        status=r1_row.get("status", "OK"),
        n_pos_points=int(r1_row.get("n_pos_points", "0") or "0"),
        n_neg_points=int(r1_row.get("n_neg_points", "0") or "0"),
        pos_coords=[],
        pred_mask=None,
    )


def _load_r1_per_reference(
    path: Path,
) -> dict[tuple[str, int, str, str], dict[str, str]]:
    """Return {(query_case_id, seed, reference_case_id, config_name): row}."""
    result: dict[tuple[str, int, str, str], dict[str, str]] = {}
    if not path.is_file():
        return result
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["query_case_id"],
                int(row["seed"]),
                row["reference_case_id"],
                row["config_name"],
            )
            result[key] = row
    return result


# ---------------------------------------------------------------------------
# CSV field schemas + checkpointing
# ---------------------------------------------------------------------------

TRAIN_CONSENSUS_FIELDS = [
    "query_case_id", "seed", "config_name", "method", "hyperparam",
    "dice", "iou", "component_recall", "target_component_count",
    "hit_components", "sam_score", "runtime_ms", "status",
    "n_pos_points", "n_neg_points",
]

VAL_CONSENSUS_FIELDS = [
    "query_case_id", "config_name", "method", "hyperparam", "reference_scope",
    "dice", "iou", "component_recall", "target_component_count",
    "hit_components", "sam_score", "runtime_ms", "status",
    "n_pos_points", "n_neg_points",
]

VAL_PER_REFERENCE_FIELDS = [
    "query_case_id", "config_name", "reference_set", "reference_index",
    "reference_case_id", "k_positive", "k_negative",
    "dice", "iou", "component_recall", "target_component_count",
    "hit_components", "sam_score", "runtime_ms", "status",
    "n_pos_points", "n_neg_points",
]

VAL_CROSS_SET_FIELDS = [
    "query_case_id", "config_name", "method", "dice_A", "dice_B",
    "dice_diff", "prompt_overlap_AB", "prediction_overlap_AB",
    "single_prediction_overlap_AB",
]


def _open_csv(path: Path, fields: list[str]) -> tuple[Any, Any]:
    is_append = path.is_file()
    handle = path.open("a" if is_append else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fields)
    if not is_append:
        writer.writeheader()
    return handle, writer


def _load_train_checkpoint(path: Path) -> set[tuple[str, int, str, str, str]]:
    """Return set of (query, seed, config, method, hyperparam) done."""
    if not path.is_file():
        return set()
    done: set[tuple[str, int, str, str, str]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            done.add((
                row["query_case_id"], int(row["seed"]),
                row["config_name"], row["method"], row["hyperparam"],
            ))
    return done


def _load_val_checkpoint(path: Path) -> set[tuple[str, str, str, str, str]]:
    """Return set of (query, config, method, hyperparam, reference_scope) done."""
    if not path.is_file():
        return set()
    done: set[tuple[str, str, str, str, str]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            done.add((
                row["query_case_id"], row["config_name"], row["method"],
                row["hyperparam"], row["reference_scope"],
            ))
    return done


def _load_val_per_ref_checkpoint(path: Path) -> set[tuple[str, str, str]]:
    """Return set of (query, config, reference_case_id) done."""
    if not path.is_file():
        return set()
    done: set[tuple[str, str, str]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            done.add((
                row["query_case_id"], row["config_name"],
                row["reference_case_id"],
            ))
    return done


def _load_val_cross_set_checkpoint(path: Path) -> set[tuple[str, str, str]]:
    """Return set of (query, config, method) done."""
    if not path.is_file():
        return set()
    done: set[tuple[str, str, str]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            done.add((
                row["query_case_id"], row["config_name"], row["method"],
            ))
    return done


def _row_dice(row: dict[str, str]) -> float | None:
    s = row.get("dice", "")
    return float(s) if s != "" else None


# ---------------------------------------------------------------------------
# Phase 1: Train consensus evaluation
# ---------------------------------------------------------------------------

def _train_row_from_record(
    query_case_id: str,
    seed: int,
    config: PromptConfig,
    method: str,
    hyperparam: str,
    rec: ConfigRecord,
) -> dict[str, Any]:
    return {
        "query_case_id": query_case_id,
        "seed": seed,
        "config_name": config.name,
        "method": method,
        "hyperparam": hyperparam,
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


def run_train_consensus(
    *,
    configs: tuple[PromptConfig, ...],
    feature_cache: FeatureCache,
    predictor: Any,
    out_dir: Path,
    seeds: range,
    max_queries: int | None = None,
) -> dict[str, Any]:
    """Phase 1: train consensus evaluation (B1-B6 per query x seed x config)."""
    train_path = out_dir / "train_consensus_metrics.csv"
    train_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_train_checkpoint(train_path)

    r1_path = PROJECT_ROOT / "reports" / "phaseR_gateR1" / "per_reference_metrics.csv"
    r1_rows = _load_r1_per_reference(r1_path)

    train_handle, train_writer = _open_csv(train_path, TRAIN_CONSENSUS_FIELDS)

    queries = queries_from_train_manifests(seeds)
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

        # Cache contrast maps per (query, reference) across seeds.
        pair_cache: dict[str, PairStateR2] = {}

        for seed in seeds:
            ref_rows = refs_for_query_seed(seed, query_case_id)
            if not ref_rows:
                continue

            # Build / fetch pair state for each reference (cached per query+ref).
            pair_states: list[PairStateR2 | None] = []
            for ref_row in ref_rows:
                ref_id = ref_row["reference_case_id"]
                try:
                    if ref_id not in pair_cache:
                        pair_cache[ref_id] = build_pair_state_r2(
                            query_case_id, ref_id, feature_cache
                        )
                    pair_states.append(pair_cache[ref_id])
                except Exception as exc:
                    print(f"[phaseR-R2] pool error {query_case_id}/{ref_id}: "
                          f"{type(exc).__name__}: {exc}")
                    pair_states.append(None)

            contrasts = [
                ps.contrast for ps in pair_states if ps is not None
            ]
            if len(contrasts) == len(pair_states) and contrasts:
                stacked = np.stack(contrasts, axis=0)
            else:
                stacked = None

            # Precompute consensus pools per variant (config-independent).
            variant_pools: dict[tuple[str, str], CandidatePools] = {}
            if stacked is not None:
                for method, hyperparam, agg_map in _train_consensus_variants(stacked):
                    key = (method, hyperparam)
                    if key not in variant_pools:
                        try:
                            variant_pools[key] = build_consensus_pool(
                                agg_map, configs[0]
                            )
                        except Exception as exc:
                            print(f"[phaseR-R2] consensus pool error "
                                  f"{query_case_id} {method}: {exc}")

            for config in configs:
                # --- B1 random single (reuse R1 per-reference Dice) ---
                b1_key = (query_case_id, seed, config.name, "B1_random", "")
                if b1_key not in done:
                    idx = _b1_index(query_case_id, seed, len(ref_rows))
                    ref_row = ref_rows[idx]
                    r1_key = (
                        query_case_id, seed, ref_row["reference_case_id"],
                        config.name,
                    )
                    r1_row = r1_rows.get(r1_key)
                    if r1_row is not None:
                        rec = _b1_record_from_r1(r1_row, config)
                    else:
                        rec = ConfigRecord(
                            dice=None, iou=None, component_recall=None,
                            target_component_count=target_component_count,
                            hit_components=None, sam_score=None, runtime_ms=0.0,
                            status="MISSING_R1", n_pos_points=0, n_neg_points=0,
                            pos_coords=[], pred_mask=None,
                        )
                    train_writer.writerow(_train_row_from_record(
                        query_case_id, seed, config, "B1_random", "", rec
                    ))
                    done.add(b1_key)

                # --- B2-B6 consensus variants (1 SAM call each) ---
                if stacked is None:
                    continue
                for (method, hyperparam), pool in variant_pools.items():
                    ck = (query_case_id, seed, config.name, method, hyperparam)
                    if ck in done:
                        continue
                    rec = run_config(
                        predictor, config, pool, target, target_labels,
                        target_component_count,
                    )
                    if rec.status == "OK":
                        total_sam_calls += 1
                    train_writer.writerow(_train_row_from_record(
                        query_case_id, seed, config, method, hyperparam, rec
                    ))
                    done.add(ck)
            train_handle.flush()

        elapsed = time.perf_counter() - start
        print(f"[phaseR-R2] train query {qi + 1}/{len(queries)} done "
              f"({query_case_id}) sam_calls={total_sam_calls} "
              f"elapsed={elapsed:.1f}s")

    train_handle.close()
    elapsed = time.perf_counter() - start
    return {
        "query_count": len(queries),
        "seed_count": len(seeds),
        "config_count": len(configs),
        "total_sam_calls": total_sam_calls,
        "elapsed_seconds": round(elapsed, 3),
    }


# ---------------------------------------------------------------------------
# Phase 2: Select best hyperparameters on train
# ---------------------------------------------------------------------------

def _load_train_consensus_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_best_hyperparameters(
    train_rows: list[dict[str, str]],
    configs: tuple[PromptConfig, ...],
) -> dict[str, Any]:
    """Phase 2: select best config/method/beta/tau on train (max mean Dice)."""
    # Group dice by (config_name, method, hyperparam).
    by_key: dict[tuple[str, str, str], list[float]] = {}
    for row in train_rows:
        d = _row_dice(row)
        if d is None:
            continue
        key = (row["config_name"], row["method"], row["hyperparam"])
        by_key.setdefault(key, []).append(d)

    def mean_of(cfg: str, method: str, hp: str) -> float:
        vals = by_key.get((cfg, method, hp), [])
        return _mean(vals)

    config_best: dict[str, dict[str, Any]] = {}
    for config in configs:
        # Best beta for B5 at this config.
        b5_means = [
            (beta, mean_of(config.name, "B5_disagreement", str(beta)))
            for beta in BETA_GRID
        ]
        best_beta, best_b5_mean = max(b5_means, key=lambda x: x[1])
        # Best tau for B6 at this config.
        b6_means = [
            (tau, mean_of(config.name, "B6_quantile", str(tau)))
            for tau in TAU_GRID
        ]
        best_tau, best_b6_mean = max(b6_means, key=lambda x: x[1])

        candidates = [
            (mean_of(config.name, "B2_mean", ""), "B2_mean", "", "B2_mean"),
            (mean_of(config.name, "B3_median", ""), "B3_median", "", "B3_median"),
            (mean_of(config.name, "B4_trimmed", ""), "B4_trimmed", "", "B4_trimmed"),
            (best_b5_mean, "B5_disagreement", str(best_beta), "B5_disagreement"),
            (best_b6_mean, "B6_quantile", str(best_tau), "B6_quantile"),
        ]
        best = max(candidates, key=lambda x: x[0])
        config_best[config.name] = {
            "best_mean_dice": best[0],
            "best_method": best[3],
            "best_hyperparam": best[2],
            "best_beta": best_beta,
            "best_tau": best_tau,
            "b5_mean_at_best_beta": best_b5_mean,
            "b6_mean_at_best_tau": best_b6_mean,
        }

    best_config_name = max(
        config_best, key=lambda c: config_best[c]["best_mean_dice"]
    )
    chosen = config_best[best_config_name]
    selection = {
        "best_config": best_config_name,
        "best_method": chosen["best_method"],
        "best_hyperparam": chosen["best_hyperparam"],
        "best_beta": chosen["best_beta"],
        "best_tau": chosen["best_tau"],
        "best_config_mean_dice": round(chosen["best_mean_dice"], 6),
        "per_config": {
            c: {
                "best_mean_dice": round(config_best[c]["best_mean_dice"], 6),
                "best_method": config_best[c]["best_method"],
                "best_hyperparam": config_best[c]["best_hyperparam"],
                "best_beta": config_best[c]["best_beta"],
                "best_tau": config_best[c]["best_tau"],
                "b5_mean_at_best_beta": round(
                    config_best[c]["b5_mean_at_best_beta"], 6
                ),
                "b6_mean_at_best_tau": round(
                    config_best[c]["b6_mean_at_best_tau"], 6
                ),
            }
            for c in [cfg.name for cfg in configs]
        },
    }
    return selection


def write_train_selection_summary(selection: dict[str, Any], out_dir: Path) -> None:
    path = out_dir / "train_selection_summary.json"
    path.write_text(
        json.dumps({"train_selection": selection}, indent=2), encoding="utf-8"
    )


def load_train_selection_summary(out_dir: Path) -> dict[str, Any]:
    path = out_dir / "train_selection_summary.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"train_selection_summary.json not found at {path}; run train phase "
            f"first or remove --skip-train."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["train_selection"]


# ---------------------------------------------------------------------------
# Phase 3: Validation evaluation
# ---------------------------------------------------------------------------

def _val_row_from_record(
    query_case_id: str,
    config: PromptConfig,
    method: str,
    hyperparam: str,
    scope: str,
    rec: ConfigRecord,
) -> dict[str, Any]:
    return {
        "query_case_id": query_case_id,
        "config_name": config.name,
        "method": method,
        "hyperparam": hyperparam,
        "reference_scope": scope,
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


def _val_per_ref_row(
    query_case_id: str,
    config: PromptConfig,
    ref_row: dict[str, str],
    rec: ConfigRecord,
) -> dict[str, Any]:
    return {
        "query_case_id": query_case_id,
        "config_name": config.name,
        "reference_set": ref_row["reference_set"],
        "reference_index": ref_row["reference_index"],
        "reference_case_id": ref_row["reference_case_id"],
        "k_positive": config.k_positive,
        "k_negative": config.k_negative,
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


def run_validation(
    *,
    configs: tuple[PromptConfig, ...],
    feature_cache: FeatureCache,
    predictor: Any,
    out_dir: Path,
    selection: dict[str, Any],
    max_queries: int | None = None,
) -> dict[str, Any]:
    """Phase 3: validation evaluation (B0, B1, B2-B6, cross-set A/B)."""
    val_path = out_dir / "val_consensus_metrics.csv"
    per_ref_path = out_dir / "val_per_reference_metrics.csv"
    cross_set_path = out_dir / "val_cross_set_stability.csv"

    done_val = _load_val_checkpoint(val_path)
    done_per_ref = _load_val_per_ref_checkpoint(per_ref_path)
    done_cross = _load_val_cross_set_checkpoint(cross_set_path)

    val_handle, val_writer = _open_csv(val_path, VAL_CONSENSUS_FIELDS)
    per_ref_handle, per_ref_writer = _open_csv(per_ref_path, VAL_PER_REFERENCE_FIELDS)
    cross_handle, cross_writer = _open_csv(cross_set_path, VAL_CROSS_SET_FIELDS)

    best_beta = float(selection["best_beta"])
    best_tau = float(selection["best_tau"])
    best_method = selection["best_method"]
    best_hyperparam = selection["best_hyperparam"]

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
        # Split refs by reference set (A/B), preserving manifest order.
        a_rows = [r for r in ref_rows if r["reference_set"] == "A"]
        b_rows = [r for r in ref_rows if r["reference_set"] == "B"]

        # Build pair states for all 6 refs (cached per query).
        pair_cache: dict[str, PairStateR2] = {}
        pair_states: list[PairStateR2 | None] = []
        for ref_row in ref_rows:
            ref_id = ref_row["reference_case_id"]
            try:
                if ref_id not in pair_cache:
                    pair_cache[ref_id] = build_pair_state_r2(
                        query_case_id, ref_id, feature_cache
                    )
                pair_states.append(pair_cache[ref_id])
            except Exception as exc:
                print(f"[phaseR-R2] pool error {query_case_id}/{ref_id}: "
                      f"{type(exc).__name__}: {exc}")
                pair_states.append(None)

        # index map from ref_row position -> pair_states position (aligned)
        # pair_states is aligned with ref_rows order.
        ref_to_state = {
            r["reference_case_id"]: ps for r, ps in zip(ref_rows, pair_states)
        }

        for config in configs:
            # --- B0: run SAM on each of 6 refs individually ---
            b0_records: list[ConfigRecord | None] = []
            for ref_row in ref_rows:
                ref_id = ref_row["reference_case_id"]
                pr_key = (query_case_id, config.name, ref_id)
                ps = ref_to_state.get(ref_id)
                if pr_key in done_per_ref or ps is None:
                    if ps is None:
                        rec = ConfigRecord(
                            dice=None, iou=None, component_recall=None,
                            target_component_count=target_component_count,
                            hit_components=None, sam_score=None, runtime_ms=0.0,
                            status="POOL_ERROR", n_pos_points=0, n_neg_points=0,
                            pos_coords=[], pred_mask=None,
                        )
                        b0_records.append(rec)
                        if pr_key not in done_per_ref:
                            per_ref_writer.writerow(_val_per_ref_row(
                                query_case_id, config, ref_row, rec
                            ))
                            done_per_ref.add(pr_key)
                    else:
                        b0_records.append(None)  # checkpointed, skip prediction use
                    continue
                rec = run_config(
                    predictor, config, ps.pools, target, target_labels,
                    target_component_count,
                )
                if rec.status == "OK":
                    total_sam_calls += 1
                b0_records.append(rec)
                per_ref_writer.writerow(_val_per_ref_row(
                    query_case_id, config, ref_row, rec
                ))
                done_per_ref.add(pr_key)
            per_ref_handle.flush()

            # --- B1: random single (reuse B0 result) ---
            b1_key = (query_case_id, config.name, "B1_random", "", "all6")
            if b1_key not in done_val:
                idx = _b1_index(query_case_id, 0, len(ref_rows))
                rec_b1 = b0_records[idx] if idx < len(b0_records) else None
                if rec_b1 is None:
                    rec_b1 = ConfigRecord(
                        dice=None, iou=None, component_recall=None,
                        target_component_count=target_component_count,
                        hit_components=None, sam_score=None, runtime_ms=0.0,
                        status="MISSING_B0", n_pos_points=0, n_neg_points=0,
                        pos_coords=[], pred_mask=None,
                    )
                val_writer.writerow(_val_row_from_record(
                    query_case_id, config, "B1_random", "", "all6", rec_b1
                ))
                done_val.add(b1_key)

            # --- B2-B6 on all6 references (frozen beta/tau) ---
            all_contrasts = [
                ps.contrast for ps in pair_states if ps is not None
            ]
            if len(all_contrasts) == len(pair_states) and all_contrasts:
                stacked_all = np.stack(all_contrasts, axis=0)
                for method, hyperparam, agg_map in _val_consensus_variants(
                    stacked_all, best_beta, best_tau
                ):
                    ck = (query_case_id, config.name, method, hyperparam, "all6")
                    if ck in done_val:
                        continue
                    try:
                        pool = build_consensus_pool(agg_map, config)
                    except Exception as exc:
                        rec = ConfigRecord(
                            dice=None, iou=None, component_recall=None,
                            target_component_count=target_component_count,
                            hit_components=None, sam_score=None, runtime_ms=0.0,
                            status=f"POOL_ERROR:{type(exc).__name__}",
                            n_pos_points=0, n_neg_points=0,
                            pos_coords=[], pred_mask=None,
                        )
                        val_writer.writerow(_val_row_from_record(
                            query_case_id, config, method, hyperparam, "all6", rec
                        ))
                        done_val.add(ck)
                        continue
                    rec = run_config(
                        predictor, config, pool, target, target_labels,
                        target_component_count,
                    )
                    if rec.status == "OK":
                        total_sam_calls += 1
                    val_writer.writerow(_val_row_from_record(
                        query_case_id, config, method, hyperparam, "all6", rec
                    ))
                    done_val.add(ck)

            # --- Cross-set: best consensus method on A_only and B_only ---
            a_contrasts = [
                ref_to_state[r["reference_case_id"]].contrast
                for r in a_rows
                if ref_to_state.get(r["reference_case_id"]) is not None
            ]
            b_contrasts = [
                ref_to_state[r["reference_case_id"]].contrast
                for r in b_rows
                if ref_to_state.get(r["reference_case_id"]) is not None
            ]

            rec_a: ConfigRecord | None = None
            rec_b: ConfigRecord | None = None
            if a_contrasts:
                ck_a = (query_case_id, config.name, best_method,
                        best_hyperparam, "A_only")
                if ck_a not in done_val:
                    agg_a = _aggregate(np.stack(a_contrasts, axis=0),
                                       best_method, best_hyperparam)
                    try:
                        pool_a = build_consensus_pool(agg_a, config)
                        rec_a = run_config(
                            predictor, config, pool_a, target, target_labels,
                            target_component_count,
                        )
                        if rec_a.status == "OK":
                            total_sam_calls += 1
                    except Exception as exc:
                        rec_a = ConfigRecord(
                            dice=None, iou=None, component_recall=None,
                            target_component_count=target_component_count,
                            hit_components=None, sam_score=None, runtime_ms=0.0,
                            status=f"POOL_ERROR:{type(exc).__name__}",
                            n_pos_points=0, n_neg_points=0,
                            pos_coords=[], pred_mask=None,
                        )
                    val_writer.writerow(_val_row_from_record(
                        query_case_id, config, best_method,
                        best_hyperparam, "A_only", rec_a
                    ))
                    done_val.add(ck_a)
            if b_contrasts:
                ck_b = (query_case_id, config.name, best_method,
                        best_hyperparam, "B_only")
                if ck_b not in done_val:
                    agg_b = _aggregate(np.stack(b_contrasts, axis=0),
                                       best_method, best_hyperparam)
                    try:
                        pool_b = build_consensus_pool(agg_b, config)
                        rec_b = run_config(
                            predictor, config, pool_b, target, target_labels,
                            target_component_count,
                        )
                        if rec_b.status == "OK":
                            total_sam_calls += 1
                    except Exception as exc:
                        rec_b = ConfigRecord(
                            dice=None, iou=None, component_recall=None,
                            target_component_count=target_component_count,
                            hit_components=None, sam_score=None, runtime_ms=0.0,
                            status=f"POOL_ERROR:{type(exc).__name__}",
                            n_pos_points=0, n_neg_points=0,
                            pos_coords=[], pred_mask=None,
                        )
                    val_writer.writerow(_val_row_from_record(
                        query_case_id, config, best_method,
                        best_hyperparam, "B_only", rec_b
                    ))
                    done_val.add(ck_b)

            # --- Cross-set stability row ---
            cross_key = (query_case_id, config.name, best_method)
            if cross_key not in done_cross and rec_a is not None and rec_b is not None:
                dice_a = rec_a.dice if rec_a.dice is not None else 0.0
                dice_b = rec_b.dice if rec_b.dice is not None else 0.0
                pred_overlap = (
                    _jaccard_masks(rec_a.pred_mask, rec_b.pred_mask)
                    if rec_a.pred_mask is not None and rec_b.pred_mask is not None
                    else 0.0
                )
                prompt_overlap = _jaccard_sets(
                    set(rec_a.pos_coords), set(rec_b.pos_coords)
                )
                # Single-ref baseline: mean pairwise Jaccard over A x B pairs.
                a_preds = [
                    b0_records[i].pred_mask
                    for i, r in enumerate(ref_rows)
                    if r["reference_set"] == "A"
                    and i < len(b0_records)
                    and b0_records[i] is not None
                    and b0_records[i].pred_mask is not None
                ]
                b_preds = [
                    b0_records[i].pred_mask
                    for i, r in enumerate(ref_rows)
                    if r["reference_set"] == "B"
                    and i < len(b0_records)
                    and b0_records[i] is not None
                    and b0_records[i].pred_mask is not None
                ]
                pair_jacs: list[float] = []
                for pa in a_preds:
                    for pb in b_preds:
                        pair_jacs.append(_jaccard_masks(pa, pb))
                single_overlap = _mean(pair_jacs)

                cross_writer.writerow({
                    "query_case_id": query_case_id,
                    "config_name": config.name,
                    "method": best_method,
                    "dice_A": round(dice_a, 6),
                    "dice_B": round(dice_b, 6),
                    "dice_diff": round(abs(dice_a - dice_b), 6),
                    "prompt_overlap_AB": round(prompt_overlap, 6),
                    "prediction_overlap_AB": round(pred_overlap, 6),
                    "single_prediction_overlap_AB": round(single_overlap, 6),
                })
                done_cross.add(cross_key)
            val_handle.flush()
            cross_handle.flush()

        elapsed = time.perf_counter() - start
        print(f"[phaseR-R2] val query {qi + 1}/{len(queries)} done "
              f"({query_case_id}) sam_calls={total_sam_calls} "
              f"elapsed={elapsed:.1f}s")

    val_handle.close()
    per_ref_handle.close()
    cross_handle.close()
    elapsed = time.perf_counter() - start
    return {
        "query_count": len(queries),
        "config_count": len(configs),
        "total_sam_calls": total_sam_calls,
        "elapsed_seconds": round(elapsed, 3),
    }


# ---------------------------------------------------------------------------
# Phase 4: Gate R2 conditions
# ---------------------------------------------------------------------------

def _load_val_consensus_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_val_per_ref_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_val_cross_set_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in (
            "dice_A", "dice_B", "dice_diff", "prompt_overlap_AB",
            "prediction_overlap_AB", "single_prediction_overlap_AB",
        ):
            row[key] = float(row[key]) if row.get(key, "") != "" else 0.0
    return rows


def _paired_bootstrap_ci(
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


def compute_gate_r2(
    val_consensus_rows: list[dict[str, str]],
    val_per_ref_rows: list[dict[str, str]],
    val_cross_set_rows: list[dict[str, Any]],
    selection: dict[str, Any],
    configs: tuple[PromptConfig, ...],
) -> dict[str, Any]:
    """Phase 4: compute the 8 Gate R2 conditions + decision."""
    best_config = selection["best_config"]
    best_method = selection["best_method"]
    best_hyperparam = selection["best_hyperparam"]

    # Best config: per-query consensus vs B1 (scope all6).
    def _filter(rows, method, scope, cfg=None, hp=None):
        out = []
        for r in rows:
            if r["method"] != method or r["reference_scope"] != scope:
                continue
            if cfg is not None and r["config_name"] != cfg:
                continue
            if hp is not None and r["hyperparam"] != hp:
                continue
            out.append(r)
        return out

    b1_rows = _filter(val_consensus_rows, "B1_random", "all6", best_config)
    cons_rows = _filter(
        val_consensus_rows, best_method, "all6", best_config, best_hyperparam
    )

    b1_by_q = {r["query_case_id"]: _row_dice(r) for r in b1_rows}
    cons_by_q = {r["query_case_id"]: _row_dice(r) for r in cons_rows}
    common_q = sorted(
        q for q in (set(b1_by_q) & set(cons_by_q))
        if b1_by_q[q] is not None and cons_by_q[q] is not None
    )
    cons_dices = [cons_by_q[q] for q in common_q]
    b1_dices = [b1_by_q[q] for q in common_q]
    deltas = [cons_by_q[q] - b1_by_q[q] for q in common_q]

    mean_cons = _mean(cons_dices)
    mean_b1 = _mean(b1_dices)
    delta_dice = mean_cons - mean_b1

    # cond1: delta Dice >= 0.010
    cond1 = delta_dice >= 0.010

    # cond2: paired bootstrap 95% CI lower > 0
    boot = _paired_bootstrap_ci(deltas)
    cond2 = boot["ci_lower"] > 0

    # cond3: variance reduction >= 25%
    per_ref_dices = [
        _row_dice(r) for r in val_per_ref_rows
        if r["config_name"] == best_config and _row_dice(r) is not None
    ]
    var_single = float(np.var(per_ref_dices)) if len(per_ref_dices) > 1 else 0.0
    var_cons = float(np.var(cons_dices)) if len(cons_dices) > 1 else 0.0
    vr = 1.0 - var_cons / var_single if var_single > 0 else 0.0
    cond3 = vr >= 0.25

    # cond4: catastrophic failure rate not increase
    cfr_cons = (
        sum(1 for d in cons_dices if d < CATASTROPHIC_DICE_THRESHOLD) / len(cons_dices)
        if cons_dices else 1.0
    )
    cfr_b1 = (
        sum(1 for d in b1_dices if d < CATASTROPHIC_DICE_THRESHOLD) / len(b1_dices)
        if b1_dices else 1.0
    )
    cond4 = cfr_cons <= cfr_b1

    # cond5: A/B prediction overlap improve
    cs_best = [r for r in val_cross_set_rows if r["config_name"] == best_config]
    cons_overlap = _mean(
        [float(r["prediction_overlap_AB"]) for r in cs_best]
    )
    single_overlap = _mean(
        [float(r["single_prediction_overlap_AB"]) for r in cs_best]
    )
    cond5 = cons_overlap > single_overlap

    # cond6: 1 SAM call (always true by construction)
    cond6 = True

    # cond7: no query GT used (always true by construction)
    cond7 = True

    # cond8: >= 2 of 3 configs show consistent direction (consensus > B1)
    consistent_configs = 0
    for config in configs:
        b1_c = _filter(val_consensus_rows, "B1_random", "all6", config.name)
        cons_c = _filter(
            val_consensus_rows, best_method, "all6", config.name, best_hyperparam
        )
        b1_dc = [d for d in (_row_dice(r) for r in b1_c) if d is not None]
        cons_dc = [d for d in (_row_dice(r) for r in cons_c) if d is not None]
        if b1_dc and cons_dc and _mean(cons_dc) > _mean(b1_dc):
            consistent_configs += 1
    cond8 = consistent_configs >= 2

    conditions = {
        "cond1_delta_dice_ge_0p010": bool(cond1),
        "cond2_bootstrap_ci_lower_gt_0": bool(cond2),
        "cond3_variance_reduction_ge_25pct": bool(cond3),
        "cond4_catastrophic_rate_not_increase": bool(cond4),
        "cond5_ab_prediction_overlap_improve": bool(cond5),
        "cond6_single_sam_call": bool(cond6),
        "cond7_no_query_gt_used": bool(cond7),
        "cond8_2of3_configs_consistent_direction": bool(cond8),
    }

    if all(conditions.values()):
        decision = "PHASER_GATE_R2_DETERMINISTIC_CONSENSUS_PASS"
    else:
        decision = "PHASER_GATE_R2_CONSENSUS_NO_GAIN"

    return {
        "query_count": len(common_q),
        "best_config": best_config,
        "best_method": best_method,
        "best_hyperparam": best_hyperparam,
        "best_beta": selection["best_beta"],
        "best_tau": selection["best_tau"],
        "mean_consensus_dice": round(mean_cons, 6),
        "mean_b1_random_dice": round(mean_b1, 6),
        "delta_dice": round(delta_dice, 6),
        "bootstrap": {k: round(v, 6) for k, v in boot.items()},
        "var_single": round(var_single, 6),
        "var_consensus": round(var_cons, 6),
        "variance_reduction": round(vr, 6),
        "cfr_consensus": round(cfr_cons, 6),
        "cfr_b1": round(cfr_b1, 6),
        "mean_consensus_ab_overlap": round(cons_overlap, 6),
        "mean_single_ab_overlap": round(single_overlap, 6),
        "consistent_config_count": consistent_configs,
        "conditions": conditions,
        "decision": decision,
    }


# ---------------------------------------------------------------------------
# Summary writers
# ---------------------------------------------------------------------------

def _fmt(v: Any, fmt: str = ".4f") -> str:
    if v is None:
        return "N/A"
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return str(v)


def write_gate_r2_summary(gate: dict[str, Any], out_dir: Path) -> None:
    json_path = out_dir / "gateR2_summary.json"
    json_path.write_text(
        json.dumps({"gate_r2": gate}, indent=2), encoding="utf-8"
    )

    md: list[str] = []
    md.append("# Phase R Gate R2: Deterministic Consensus Feasibility Summary")
    md.append("")
    md.append(f"Date: {datetime.now().isoformat(timespec='seconds')}")
    md.append(f"Validation queries: {gate['query_count']}")
    md.append(f"Best config: {gate['best_config']}")
    md.append(f"Best consensus method: {gate['best_method']}"
              f" (hyperparam={gate['best_hyperparam'] or '-'}, "
              f"beta={gate['best_beta']}, tau={gate['best_tau']})")
    md.append("")
    md.append("## Consensus vs Random Single (B1)")
    md.append("")
    md.append("| Metric | Value |")
    md.append("|---|---:|")
    md.append(f"| Mean consensus Dice | {_fmt(gate['mean_consensus_dice'])} |")
    md.append(f"| Mean B1 random single Dice | {_fmt(gate['mean_b1_random_dice'])} |")
    md.append(f"| Delta Dice | {_fmt(gate['delta_dice'])} |")
    md.append(f"| Bootstrap 95% CI | "
              f"[{_fmt(gate['bootstrap']['ci_lower'])}, "
              f"{_fmt(gate['bootstrap']['ci_upper'])}] |")
    md.append(f"| Var(single) | {_fmt(gate['var_single'])} |")
    md.append(f"| Var(consensus) | {_fmt(gate['var_consensus'])} |")
    md.append(f"| Variance reduction | {_fmt(gate['variance_reduction'])} |")
    md.append(f"| CFR consensus | {_fmt(gate['cfr_consensus'])} |")
    md.append(f"| CFR B1 | {_fmt(gate['cfr_b1'])} |")
    md.append(f"| Mean consensus A/B overlap | {_fmt(gate['mean_consensus_ab_overlap'])} |")
    md.append(f"| Mean single A/B overlap | {_fmt(gate['mean_single_ab_overlap'])} |")
    md.append(f"| Consistent config count (of 3) | {gate['consistent_config_count']} |")
    md.append("")
    md.append("## Gate R2 Conditions")
    md.append("")
    md.append("| # | Condition | Pass |")
    md.append("|---|---|---|")
    labels = {
        "cond1_delta_dice_ge_0p010": "Delta Dice >= 0.010 (consensus - B1)",
        "cond2_bootstrap_ci_lower_gt_0": "Paired bootstrap 95% CI lower > 0",
        "cond3_variance_reduction_ge_25pct": "Variance reduction >= 25%",
        "cond4_catastrophic_rate_not_increase": "Catastrophic failure rate not increase",
        "cond5_ab_prediction_overlap_improve": "A/B prediction overlap improve",
        "cond6_single_sam_call": "Single SAM call (by construction)",
        "cond7_no_query_gt_used": "No query GT used (by construction)",
        "cond8_2of3_configs_consistent_direction": ">= 2 of 3 configs consistent direction",
    }
    for i, key in enumerate(labels, start=1):
        md.append(f"| {i} | {labels[key]} | "
                  f"{'YES' if gate['conditions'][key] else 'NO'} |")
    md.append("")
    md.append(f"## Decision: **{gate['decision']}**")
    md.append("")
    (out_dir / "gateR2_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def write_run_metadata(
    out_dir: Path,
    *,
    device: torch.device,
    train_meta: dict[str, Any],
    val_meta: dict[str, Any] | None,
    gate: dict[str, Any] | None,
) -> None:
    elapsed = train_meta.get("elapsed_seconds", 0.0)
    if val_meta is not None:
        elapsed += val_meta.get("elapsed_seconds", 0.0)
    total_sam = train_meta.get("total_sam_calls", 0)
    if val_meta is not None:
        total_sam += val_meta.get("total_sam_calls", 0)
    gpu_hours = (
        elapsed / 3600.0 if str(device).startswith("cuda") else 0.0
    )
    metadata = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "runtime_seconds": round(elapsed, 3),
        "gpu_hours": round(gpu_hours, 6),
        "total_sam_calls": total_sam,
        "train_query_count": train_meta.get("query_count", 0),
        "train_seed_count": train_meta.get("seed_count", 0),
        "config_count": train_meta.get("config_count", 0),
        "val_query_count": (val_meta or {}).get("query_count", 0),
        "gate_r2_decision": (gate or {}).get("decision"),
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase R Gate R2: Deterministic Consensus Feasibility"
    )
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="Quick verification: 2 train x 1 seed, 2 val.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip train phase; reuse train_selection_summary.json.")
    parser.add_argument("--train-only", action="store_true",
                        help="Run train phase only (Phase 1 + 2).")
    parser.add_argument(
        "--out-dir", type=Path,
        default=PROJECT_ROOT / "reports" / "phaseR_gateR2",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        seeds = range(1)  # seed 0 only
        train_max = args.max_queries or 2
        val_max = args.max_queries or 2
    else:
        seeds = range(SEED_COUNT)
        train_max = args.max_queries
        val_max = args.max_queries

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[phaseR-R2] device={device} seeds={list(seeds)} "
          f"train_max={train_max} val_max={val_max} "
          f"smoke={args.smoke} skip_train={args.skip_train} "
          f"train_only={args.train_only}")

    configs = FROZEN_CONFIGS
    feature_cache = FeatureCache(device=device)
    predictor = load_sam_predictor(device)

    train_meta: dict[str, Any] = {
        "query_count": 0, "seed_count": len(seeds), "config_count": len(configs),
        "total_sam_calls": 0, "elapsed_seconds": 0.0,
    }

    # --- Phase 1 + 2: train consensus + selection ---
    if not args.skip_train:
        train_meta = run_train_consensus(
            configs=configs,
            feature_cache=feature_cache,
            predictor=predictor,
            out_dir=out_dir,
            seeds=seeds,
            max_queries=train_max,
        )
        print(f"[phaseR-R2] train done: {train_meta['total_sam_calls']} SAM calls, "
              f"{train_meta['elapsed_seconds']}s")

        train_rows = _load_train_consensus_rows(
            out_dir / "train_consensus_metrics.csv"
        )
        selection = select_best_hyperparameters(train_rows, configs)
        write_train_selection_summary(selection, out_dir)
        print(f"[phaseR-R2] selection: config={selection['best_config']} "
              f"method={selection['best_method']} "
              f"hyperparam={selection['best_hyperparam'] or '-'} "
              f"beta={selection['best_beta']} tau={selection['best_tau']}")
    else:
        selection = load_train_selection_summary(out_dir)
        print(f"[phaseR-R2] loaded selection: config={selection['best_config']} "
              f"method={selection['best_method']}")

    if args.train_only:
        write_run_metadata(out_dir, device=device, train_meta=train_meta,
                           val_meta=None, gate=None)
        print(f"[phaseR-R2] train-only mode; artifacts written to {out_dir}")
        return

    # --- Phase 3: validation ---
    val_meta = run_validation(
        configs=configs,
        feature_cache=feature_cache,
        predictor=predictor,
        out_dir=out_dir,
        selection=selection,
        max_queries=val_max,
    )
    print(f"[phaseR-R2] val done: {val_meta['total_sam_calls']} SAM calls, "
          f"{val_meta['elapsed_seconds']}s")

    # --- Phase 4: gate conditions ---
    val_consensus_rows = _load_val_consensus_rows(
        out_dir / "val_consensus_metrics.csv"
    )
    val_per_ref_rows = _load_val_per_ref_rows(
        out_dir / "val_per_reference_metrics.csv"
    )
    val_cross_set_rows = _load_val_cross_set_rows(
        out_dir / "val_cross_set_stability.csv"
    )
    gate = compute_gate_r2(
        val_consensus_rows, val_per_ref_rows, val_cross_set_rows,
        selection, configs,
    )
    write_gate_r2_summary(gate, out_dir)
    write_run_metadata(out_dir, device=device, train_meta=train_meta,
                       val_meta=val_meta, gate=gate)

    print(f"[phaseR-R2] decision: {gate['decision']}")
    print(f"[phaseR-R2] artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
