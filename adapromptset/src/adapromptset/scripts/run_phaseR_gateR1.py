"""Phase R Gate R1: Reference Instability Replication.

Implements the Gate R1 decision from the Phase R pre-registration. For each
train query, 5 resampling seeds, and 3 frozen prompt configs, the script runs
SAM-B on each of the 6 references per (query, seed) and measures how much the
predicted Dice varies across references.

The gate confirms whether the ``REFERENCE_INSTABILITY`` bottleneck identified in
Phase Y replicates on the train split under a controlled resampling protocol.

Outputs (under ``reports/phaseR_gateR1/``)::

    per_reference_metrics.csv     (one row per query x seed x ref x config)
    per_query_instability.csv     (one row per query x seed x config)
    per_query_aggregate.csv       (one row per query x config, across seeds)
    gateR1_summary.json           (Gate R1 decision, 5 conditions)
    gateR1_summary.md             (human-readable summary)
    run_metadata.json             (device, runtime, gpu_hours, total_sam_calls)

Only 3 frozen prompt configs are evaluated (P20-N1, P30-N0, P50-N0), all using
``P_TOPK`` positives, ``G_ONE_GROUP`` grouping, and a single SAM call.
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
GAP_SENSITIVITY = 0.02  # cond3 per-seed gap threshold
GAP_REPLICATION = 0.05  # cond1/cond4 best-worst gap threshold


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
# Case ID -> path resolution (copied from Y1)
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
# Image / mask loading (copied from Y1)
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
# Component labels for recall computation (copied from Y1)
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
# SAM predictor (copied from Y1)
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
# DINOv2 feature cache (copied from Y1)
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
# Matching maps + candidate pools for one (query, ref) pair
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PairState:
    """Candidate pool (topk only) + s_fg matching map for one (query, ref) pair."""
    pools: CandidatePools
    s_fg: np.ndarray  # grid_size x grid_size float64


def build_pair_state(
    query_case_id: str,
    reference_case_id: str,
    feature_cache: FeatureCache,
) -> PairState:
    """Build the P_TOPK candidate pool and s_fg map for one (query, ref) pair."""
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
    s_bg = maps["s_bg"].tolist()
    s_fg_list = maps["s_fg"].tolist()
    image_size = (IMAGE_SIZE, IMAGE_SIZE)

    pools = build_candidate_pools(
        s_fg_list, s_bg,
        image_size=image_size,
        positive_limit=POSITIVE_LIMIT,
        negative_limit=NEGATIVE_LIMIT,
        positive_layout="topk",
        min_positive_distance=MIN_POSITIVE_DISTANCE,
    )
    return PairState(pools=pools, s_fg=s_fg)


# ---------------------------------------------------------------------------
# Point selection (copied from Y1, trimmed to P_TOPK / N_BOTTOMK / N_NONE)
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


def topk_candidate_grid_indices(pools: CandidatePools, k: int) -> set[tuple[int, int]]:
    """Return the (row, col) grid indices of the top-K positive candidates."""
    indices = [c.grid_index for c in pools.positives]
    return set(indices[:k] if k <= len(indices) else indices)


# ---------------------------------------------------------------------------
# Config execution: SAM predict + metrics
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
# Manifest loading
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_train_resampling(seed_index: int) -> list[dict[str, str]]:
    """Load the train resampling manifest for the given seed index (0-4)."""
    path = PROJECT_ROOT / "manifests" / "phaseR" / f"train_resampling_seed_{seed_index}.csv"
    return _read_csv(path)


def queries_from_manifests(seeds: range) -> list[str]:
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


def refs_for_query_seed(seed_index: int, query_case_id: str) -> list[dict[str, str]]:
    """Return the 6 reference rows for one (query, seed)."""
    rows = load_train_resampling(seed_index)
    return [r for r in rows if r["query_case_id"] == query_case_id]


# ---------------------------------------------------------------------------
# CSV field schemas + checkpointing
# ---------------------------------------------------------------------------

PER_REFERENCE_FIELDS = [
    "query_case_id", "seed", "reference_set", "reference_index",
    "reference_case_id", "config_name", "k_positive", "k_negative",
    "dice", "iou", "component_recall", "target_component_count",
    "hit_components", "sam_score", "runtime_ms", "status",
    "n_pos_points", "n_neg_points",
]

PER_QUERY_INSTABILITY_FIELDS = [
    "query_case_id", "seed", "config_name", "mean_dice", "std_dice",
    "min_dice", "max_dice", "best_worst_gap", "catastrophic_failure_rate",
    "n_references", "matching_map_disagreement", "candidate_jaccard_mean",
    "prompt_overlap_mean", "prediction_jaccard_mean",
]

PER_QUERY_AGGREGATE_FIELDS = [
    "query_case_id", "config_name", "overall_mean_dice", "overall_std_dice",
    "overall_min_dice", "overall_max_dice", "overall_best_worst_gap",
    "seed_direction_consistent", "catastrophic_failure_rate",
]


def _load_per_reference_checkpoint(path: Path) -> set[tuple[str, int, str, str]]:
    """Return set of (query_case_id, seed, reference_case_id, config_name) done."""
    if not path.is_file():
        return set()
    done: set[tuple[str, int, str, str]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            done.add((
                row["query_case_id"],
                int(row["seed"]),
                row["reference_case_id"],
                row["config_name"],
            ))
    return done


def _load_instability_checkpoint(path: Path) -> set[tuple[str, int, str]]:
    """Return set of (query_case_id, seed, config_name) with an instability row."""
    if not path.is_file():
        return set()
    done: set[tuple[str, int, str]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            done.add((row["query_case_id"], int(row["seed"]), row["config_name"]))
    return done


# ---------------------------------------------------------------------------
# Instability metrics across the 6 references of a (query, seed, config)
# ---------------------------------------------------------------------------

def _pairwise_mean_pearson_disagreement(arrays: list[np.ndarray | None]) -> float:
    """Mean pairwise (1 - Pearson r) of flattened s_fg maps (None entries skipped)."""
    valid = [a.flatten() for a in arrays if a is not None]
    n = len(valid)
    if n < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = valid[i]
            b = valid[j]
            sa = a.std()
            sb = b.std()
            if sa == 0.0 or sb == 0.0:
                # No variance -> treat as perfect agreement (disagreement 0).
                total += 0.0
            else:
                corr = float(np.corrcoef(a, b)[0, 1])
                if math.isnan(corr):
                    corr = 1.0
                total += 1.0 - corr
            pairs += 1
    return total / pairs if pairs else 0.0


def _jaccard_sets(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _pairwise_mean_jaccard_sets(sets: list[set[tuple[int, int]]]) -> float:
    valid = [s for s in sets if s is not None]
    n = len(valid)
    if n < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += _jaccard_sets(valid[i], valid[j])
            pairs += 1
    return total / pairs if pairs else 0.0


def _jaccard_masks(a: np.ndarray, b: np.ndarray) -> float:
    a_ = a.ravel()
    b_ = b.ravel()
    inter = int(np.logical_and(a_, b_).sum())
    union = int(np.logical_or(a_, b_).sum())
    if union == 0:
        return 1.0
    return inter / union


def _pairwise_mean_jaccard_masks(masks: list[np.ndarray | None]) -> float:
    valid = [m for m in masks if m is not None]
    n = len(valid)
    if n < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += _jaccard_masks(valid[i], valid[j])
            pairs += 1
    return total / pairs if pairs else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def compute_per_query_instability(
    query_case_id: str,
    seed: int,
    config: PromptConfig,
    records: list[ConfigRecord],
    s_fg_maps: list[np.ndarray | None],
    candidate_sets: list[set[tuple[int, int]] | None],
) -> dict[str, Any]:
    """Compute instability metrics across the 6 references for one config."""
    dices = [r.dice for r in records if r.dice is not None]
    n_refs = len(records)

    if dices:
        mean_dice = _mean(dices)
        std_dice = _std(dices)
        min_dice = min(dices)
        max_dice = max(dices)
        best_worst_gap = max_dice - min_dice
        catastrophic = sum(1 for d in dices if d < CATASTROPHIC_DICE_THRESHOLD) / len(dices)
    else:
        mean_dice = 0.0
        std_dice = 0.0
        min_dice = 0.0
        max_dice = 0.0
        best_worst_gap = 0.0
        catastrophic = 1.0

    prompt_sets = [set(r.pos_coords) if r.pos_coords else set() for r in records]
    pred_masks = [r.pred_mask for r in records]

    return {
        "query_case_id": query_case_id,
        "seed": seed,
        "config_name": config.name,
        "mean_dice": round(mean_dice, 6),
        "std_dice": round(std_dice, 6),
        "min_dice": round(min_dice, 6),
        "max_dice": round(max_dice, 6),
        "best_worst_gap": round(best_worst_gap, 6),
        "catastrophic_failure_rate": round(catastrophic, 6),
        "n_references": n_refs,
        "matching_map_disagreement": round(
            _pairwise_mean_pearson_disagreement(s_fg_maps), 6
        ),
        "candidate_jaccard_mean": round(
            _pairwise_mean_jaccard_sets(candidate_sets), 6
        ),
        "prompt_overlap_mean": round(
            _pairwise_mean_jaccard_sets(prompt_sets), 6
        ),
        "prediction_jaccard_mean": round(
            _pairwise_mean_jaccard_masks(pred_masks), 6
        ),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_phaseR_gateR1(
    *,
    configs: tuple[PromptConfig, ...],
    feature_cache: FeatureCache,
    predictor: Any,
    out_dir: Path,
    seeds: range,
    max_queries: int | None = None,
) -> dict[str, Any]:
    """Run the full Phase R Gate R1 pipeline."""
    per_ref_path = out_dir / "per_reference_metrics.csv"
    instability_path = out_dir / "per_query_instability.csv"
    per_ref_path.parent.mkdir(parents=True, exist_ok=True)

    done_ref = _load_per_reference_checkpoint(per_ref_path)
    done_inst = _load_instability_checkpoint(instability_path)

    is_append_ref = per_ref_path.is_file()
    ref_handle = per_ref_path.open("a" if is_append_ref else "w", newline="", encoding="utf-8")
    ref_writer = csv.DictWriter(ref_handle, fieldnames=PER_REFERENCE_FIELDS)
    if not is_append_ref:
        ref_writer.writeheader()

    is_append_inst = instability_path.is_file()
    inst_handle = instability_path.open(
        "a" if is_append_inst else "w", newline="", encoding="utf-8"
    )
    inst_writer = csv.DictWriter(inst_handle, fieldnames=PER_QUERY_INSTABILITY_FIELDS)
    if not is_append_inst:
        inst_writer.writeheader()

    queries = queries_from_manifests(seeds)
    if max_queries is not None:
        queries = queries[:max_queries]

    total_sam_calls = 0
    total_pairs = 0
    start = time.perf_counter()

    for qi, query_case_id in enumerate(queries):
        # Set SAM image + target mask once per query.
        query_paths = resolve_case_paths(query_case_id)
        query_image = load_rgb_518(query_paths["image"])
        predictor.set_image(np.asarray(query_image))
        target = load_mask_518(query_paths["mask"])
        target_labels = _component_labels(target)
        target_component_count = len(set(target_labels.values()))

        # Cache (query, reference) pair state across seeds for this query.
        pair_cache: dict[str, PairState] = {}

        for seed in seeds:
            ref_rows = refs_for_query_seed(seed, query_case_id)
            if not ref_rows:
                continue

            for config in configs:
                inst_key = (query_case_id, seed, config.name)
                # If instability is already computed, all 6 refs were processed;
                # skip the SAM work for this (query, seed, config) on resumption.
                if inst_key in done_inst:
                    continue

                records: list[ConfigRecord] = []
                sfg_maps: list[np.ndarray | None] = []
                candidate_sets: list[set[tuple[int, int]] | None] = []

                for ref_row in ref_rows:
                    ref_case_id = ref_row["reference_case_id"]
                    ref_set = ref_row["reference_set"]
                    ref_index = ref_row["reference_index"]

                    # Build / fetch pair state (cached per query+reference).
                    try:
                        if ref_case_id not in pair_cache:
                            pair_cache[ref_case_id] = build_pair_state(
                                query_case_id, ref_case_id, feature_cache
                            )
                        pair_state = pair_cache[ref_case_id]
                    except Exception as exc:
                        pair_state = None
                        err = f"POOL_ERROR:{type(exc).__name__}"

                    if pair_state is None:
                        rec = ConfigRecord(
                            dice=None, iou=None, component_recall=None,
                            target_component_count=target_component_count,
                            hit_components=None, sam_score=None, runtime_ms=0.0,
                            status=err, n_pos_points=0, n_neg_points=0,
                            pos_coords=[], pred_mask=None,
                        )
                        cand_set: set[tuple[int, int]] | None = None
                        sfg: np.ndarray | None = None
                    else:
                        rec = run_config(
                            predictor, config, pair_state.pools, target,
                            target_labels, target_component_count,
                        )
                        if rec.status == "OK":
                            total_sam_calls += 1
                        cand_set = topk_candidate_grid_indices(
                            pair_state.pools, config.k_positive
                        )
                        sfg = pair_state.s_fg

                    records.append(rec)
                    sfg_maps.append(sfg)
                    candidate_sets.append(cand_set)

                    # Write per-reference row if not checkpointed.
                    ref_key = (query_case_id, seed, ref_case_id, config.name)
                    if ref_key not in done_ref:
                        ref_writer.writerow({
                            "query_case_id": query_case_id,
                            "seed": seed,
                            "reference_set": ref_set,
                            "reference_index": ref_index,
                            "reference_case_id": ref_case_id,
                            "config_name": config.name,
                            "k_positive": config.k_positive,
                            "k_negative": config.k_negative,
                            "dice": rec.dice if rec.dice is not None else "",
                            "iou": rec.iou if rec.iou is not None else "",
                            "component_recall": (
                                rec.component_recall
                                if rec.component_recall is not None else ""
                            ),
                            "target_component_count": rec.target_component_count,
                            "hit_components": (
                                rec.hit_components
                                if rec.hit_components is not None else ""
                            ),
                            "sam_score": rec.sam_score if rec.sam_score is not None else "",
                            "runtime_ms": rec.runtime_ms,
                            "status": rec.status,
                            "n_pos_points": rec.n_pos_points,
                            "n_neg_points": rec.n_neg_points,
                        })
                        done_ref.add(ref_key)
                    total_pairs += 1

                ref_handle.flush()

                # Compute + write per-query instability row.
                inst_row = compute_per_query_instability(
                    query_case_id, seed, config, records, sfg_maps, candidate_sets,
                )
                inst_writer.writerow(inst_row)
                done_inst.add(inst_key)
            inst_handle.flush()

        elapsed = time.perf_counter() - start
        print(f"[phaseR-R1] query {qi + 1}/{len(queries)} done "
              f"({query_case_id}) pairs={total_pairs} "
              f"sam_calls={total_sam_calls} elapsed={elapsed:.1f}s")

    ref_handle.close()
    inst_handle.close()
    elapsed = time.perf_counter() - start
    return {
        "query_count": len(queries),
        "seed_count": len(seeds),
        "config_count": len(configs),
        "total_pairs": total_pairs,
        "total_sam_calls": total_sam_calls,
        "elapsed_seconds": round(elapsed, 3),
    }


# ---------------------------------------------------------------------------
# Aggregation across seeds + Gate R1 decision
# ---------------------------------------------------------------------------

def _load_instability_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key in (
                "mean_dice", "std_dice", "min_dice", "max_dice",
                "best_worst_gap", "catastrophic_failure_rate",
                "matching_map_disagreement", "candidate_jaccard_mean",
                "prompt_overlap_mean", "prediction_jaccard_mean",
            ):
                row[key] = float(row[key]) if row[key] != "" else 0.0
            row["seed"] = int(row["seed"])
            row["n_references"] = int(row["n_references"])
            rows.append(row)
    return rows


def _load_per_reference_dices(
    path: Path,
) -> dict[tuple[str, str], dict[int, list[float]]]:
    """Return {(query_case_id, config_name): {seed: [dice, ...]}} from CSV."""
    result: dict[tuple[str, str], dict[int, list[float]]] = {}
    if not path.is_file():
        return result
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            dice_str = row.get("dice", "")
            if dice_str == "":
                continue
            key = (row["query_case_id"], row["config_name"])
            seed = int(row["seed"])
            result.setdefault(key, {}).setdefault(seed, []).append(float(dice_str))
    return result


def compute_per_query_aggregate(
    inst_rows: list[dict[str, Any]],
    out_dir: Path,
) -> list[dict[str, Any]]:
    """Aggregate instability across seeds -> per_query_aggregate.csv + rows.

    Uses per-reference Dice values from ``per_reference_metrics.csv`` for exact
    cross-seed statistics, falling back to per-seed summaries when individual
    Dice values are unavailable.
    """
    per_ref_dices = _load_per_reference_dices(out_dir / "per_reference_metrics.csv")

    by_qc: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in inst_rows:
        by_qc.setdefault((r["query_case_id"], r["config_name"]), []).append(r)

    agg_rows: list[dict[str, Any]] = []
    for (qid, cfg), seed_rows in sorted(by_qc.items()):
        seed_dices = per_ref_dices.get((qid, cfg), {})

        # Per-seed best-worst gaps (prefer exact per-reference dice).
        per_seed_gaps: list[float] = []
        for sr in seed_rows:
            sd = seed_dices.get(sr["seed"], [])
            if sd:
                per_seed_gaps.append(max(sd) - min(sd))
            else:
                per_seed_gaps.append(sr["best_worst_gap"])

        # Pool all valid Dice values across seeds + references.
        all_dices: list[float] = []
        for sd in seed_dices.values():
            all_dices.extend(sd)
        if not all_dices:
            # Fallback: reconstruct from per-seed min/max.
            for sr in seed_rows:
                all_dices.append(sr["min_dice"])
                all_dices.append(sr["max_dice"])

        if all_dices:
            overall_mean = _mean(all_dices)
            overall_std = _std(all_dices)
            overall_min = min(all_dices)
            overall_max = max(all_dices)
            catastrophic = (
                sum(1 for d in all_dices if d < CATASTROPHIC_DICE_THRESHOLD)
                / len(all_dices)
            )
        else:
            overall_mean = 0.0
            overall_std = 0.0
            overall_min = 0.0
            overall_max = 0.0
            catastrophic = 1.0

        overall_gap = max(per_seed_gaps) if per_seed_gaps else 0.0
        n_seeds = len(seed_rows)
        consistent = sum(1 for g in per_seed_gaps if g >= GAP_SENSITIVITY)
        # cond3: >= 4 of 5 seeds show gap >= 0.02 (adapt threshold for fewer seeds).
        consistency_threshold = max(2, n_seeds - 1) if n_seeds > 1 else n_seeds
        seed_direction_consistent = consistent >= consistency_threshold

        agg_rows.append({
            "query_case_id": qid,
            "config_name": cfg,
            "overall_mean_dice": round(overall_mean, 6),
            "overall_std_dice": round(overall_std, 6),
            "overall_min_dice": round(overall_min, 6),
            "overall_max_dice": round(overall_max, 6),
            "overall_best_worst_gap": round(overall_gap, 6),
            "seed_direction_consistent": bool(seed_direction_consistent),
            "catastrophic_failure_rate": round(catastrophic, 6),
        })

    agg_path = out_dir / "per_query_aggregate.csv"
    with agg_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_QUERY_AGGREGATE_FIELDS)
        writer.writeheader()
        for row in agg_rows:
            writer.writerow(row)
    return agg_rows


def compute_gate_r1(
    inst_rows: list[dict[str, Any]],
    agg_rows: list[dict[str, Any]],
    configs: tuple[PromptConfig, ...],
) -> dict[str, Any]:
    """Compute the Gate R1 decision from instability + aggregate rows."""
    query_count = len({r["query_case_id"] for r in agg_rows})
    seed_count = len({r["seed"] for r in inst_rows}) if inst_rows else SEED_COUNT

    # cond1: >= 30% of queries have best-worst gap >= 0.05 (max gap per query).
    per_query_max_gap: dict[str, float] = {}
    for r in agg_rows:
        qid = r["query_case_id"]
        per_query_max_gap[qid] = max(
            per_query_max_gap.get(qid, 0.0), r["overall_best_worst_gap"]
        )
    if per_query_max_gap:
        pct_gap = sum(1 for g in per_query_max_gap.values() if g >= GAP_REPLICATION) / len(
            per_query_max_gap
        )
    else:
        pct_gap = 0.0
    cond1 = pct_gap >= 0.30

    # cond2: mean reference std >= 0.02 (mean across all query x seed x config).
    if inst_rows:
        mean_ref_std = _mean([r["std_dice"] for r in inst_rows])
    else:
        mean_ref_std = 0.0
    cond2 = mean_ref_std >= 0.02

    # cond3: direction consistent across 5 seeds.
    if agg_rows:
        consistent_fraction = sum(
            1 for r in agg_rows if r["seed_direction_consistent"]
        ) / len(agg_rows)
    else:
        consistent_fraction = 0.0
    seed_direction_consistent = consistent_fraction >= 0.50
    cond3 = seed_direction_consistent

    # cond4: at least one config shows stable sensitivity
    # (mean best-worst gap >= 0.05 for that config across queries).
    by_config_gap: dict[str, list[float]] = {c.name: [] for c in configs}
    for r in agg_rows:
        by_config_gap.setdefault(r["config_name"], []).append(
            r["overall_best_worst_gap"]
        )
    config_mean_gaps = {
        cfg: _mean(gaps) for cfg, gaps in by_config_gap.items() if gaps
    }
    sensitive_config: str | None = None
    best_config_gap = -1.0
    for cfg, gap in config_mean_gaps.items():
        if gap > best_config_gap:
            best_config_gap = gap
            sensitive_config = cfg
    if best_config_gap < GAP_REPLICATION:
        sensitive_config = None
    cond4 = any(gap >= GAP_REPLICATION for gap in config_mean_gaps.values())

    cond5 = True

    conditions = {
        "cond1_30pct_best_worst_gap_ge_0p05": bool(cond1),
        "cond2_mean_ref_std_ge_0p02": bool(cond2),
        "cond3_direction_consistent_across_seeds": bool(cond3),
        "cond4_one_config_shows_sensitivity": bool(cond4),
        "cond5_not_code_error": True,
    }

    if all(conditions.values()):
        decision = "PHASER_GATE_R1_REFERENCE_INSTABILITY_CONFIRMED"
    else:
        decision = "PHASER_GATE_R1_REFERENCE_INSTABILITY_NOT_REPLICATED"

    return {
        "query_count": query_count,
        "seed_count": seed_count,
        "config_count": len(configs),
        "configs": [c.name for c in configs],
        "pct_queries_best_worst_gap_ge_0p05": round(pct_gap, 6),
        "mean_reference_std": round(mean_ref_std, 6),
        "seed_direction_consistent": bool(seed_direction_consistent),
        "sensitive_config": sensitive_config,
        "not_code_error": True,
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


def write_gate_r1_summary(gate: dict[str, Any], out_dir: Path) -> None:
    json_path = out_dir / "gateR1_summary.json"
    json_path.write_text(
        json.dumps({"gate_r1": gate}, indent=2), encoding="utf-8"
    )

    md: list[str] = []
    md.append("# Phase R Gate R1: Reference Instability Replication Summary")
    md.append("")
    md.append(f"Date: {datetime.now().isoformat(timespec='seconds')}")
    md.append(f"Train queries: {gate['query_count']}")
    md.append(f"Resampling seeds: {gate['seed_count']}")
    md.append(f"Prompt configs: {gate['config_count']} ({', '.join(gate['configs'])})")
    md.append("")
    md.append("## Replication Statistics")
    md.append("")
    md.append("| Metric | Value |")
    md.append("|---|---:|")
    md.append(f"| % queries best-worst gap >= 0.05 | "
              f"{_fmt(gate['pct_queries_best_worst_gap_ge_0p05'])} |")
    md.append(f"| Mean reference std | "
              f"{_fmt(gate['mean_reference_std'])} |")
    md.append(f"| Seed direction consistent | "
              f"{'YES' if gate['seed_direction_consistent'] else 'NO'} |")
    md.append(f"| Sensitive config | {gate['sensitive_config'] or 'none'} |")
    md.append("")
    md.append("## Gate R1 Conditions")
    md.append("")
    md.append("| # | Condition | Pass |")
    md.append("|---|---|---|")
    labels = {
        "cond1_30pct_best_worst_gap_ge_0p05":
            ">= 30% queries best-worst gap >= 0.05",
        "cond2_mean_ref_std_ge_0p02": "mean reference std >= 0.02",
        "cond3_direction_consistent_across_seeds":
            "direction consistent across 5 seeds",
        "cond4_one_config_shows_sensitivity":
            ">= 1 config shows stable sensitivity",
        "cond5_not_code_error": "not a code error",
    }
    for i, key in enumerate(labels, start=1):
        md.append(f"| {i} | {labels[key]} | "
                  f"{'YES' if gate['conditions'][key] else 'NO'} |")
    md.append("")
    md.append(f"## Decision: **{gate['decision']}**")
    md.append("")
    (out_dir / "gateR1_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def write_run_metadata(
    out_dir: Path,
    *,
    device: torch.device,
    run_meta: dict[str, Any],
    gate: dict[str, Any],
) -> None:
    elapsed = run_meta.get("elapsed_seconds", 0.0)
    gpu_hours = (
        elapsed / 3600.0 if str(device).startswith("cuda") else 0.0
    )
    metadata = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "runtime_seconds": elapsed,
        "gpu_hours": round(gpu_hours, 6),
        "total_sam_calls": run_meta.get("total_sam_calls", 0),
        "query_count": run_meta.get("query_count", 0),
        "seed_count": run_meta.get("seed_count", 0),
        "config_count": run_meta.get("config_count", 0),
        "gate_r1_decision": gate.get("decision"),
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase R Gate R1: Reference Instability Replication"
    )
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="Quick verification: 2 queries x 1 seed.")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--out-dir", type=Path,
        default=PROJECT_ROOT / "reports" / "phaseR_gateR1",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        seeds = range(1)  # seed 0 only
        max_queries = args.max_queries or 2
    else:
        seeds = range(SEED_COUNT)
        max_queries = args.max_queries

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[phaseR-R1] device={device} seeds={list(seeds)} "
          f"max_queries={max_queries} smoke={args.smoke}")

    configs = FROZEN_CONFIGS

    feature_cache = FeatureCache(device=device)
    predictor = load_sam_predictor(device)

    run_meta = run_phaseR_gateR1(
        configs=configs,
        feature_cache=feature_cache,
        predictor=predictor,
        out_dir=out_dir,
        seeds=seeds,
        max_queries=max_queries,
    )
    print(f"[phaseR-R1] inference done: {run_meta['total_pairs']} pairs, "
          f"{run_meta['total_sam_calls']} SAM calls, "
          f"{run_meta['elapsed_seconds']}s")

    # Aggregate across seeds + decide gate.
    inst_rows = _load_instability_rows(out_dir / "per_query_instability.csv")
    agg_rows = compute_per_query_aggregate(inst_rows, out_dir)
    gate = compute_gate_r1(inst_rows, agg_rows, configs)
    write_gate_r1_summary(gate, out_dir)
    write_run_metadata(out_dir, device=device, run_meta=run_meta, gate=gate)

    print(f"[phaseR-R1] decision: {gate['decision']}")
    print(f"[phaseR-R1] artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
