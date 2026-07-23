"""Phase T WP1 MoNuSAC Holdout Confirmatory Benchmark.

Runs the reference-instability benchmark on the 40 MoNuSAC holdout queries that
have **never** been used in any prior phase (Phase Y ``confirmatory_holdout``
split, ``train_149`` .. ``train_033``). The holdout is the confirmatory test
for the WP1 reference-instability phenomenon and the consensus mitigation.

Pipeline (4 phases)::

    Phase 1 (CPU)  -- reference resampling. For each of the 40 holdout queries,
                      sample 6 patient-disjoint references from the Phase Y
                      ``reference_pool.csv`` under 10 fixed seeds (0..9).
                      Manifests -> manifests/phaseT/holdout_resampling_seed_{i}.csv

    Phase 2 (CPU/GPU) -- DINOv2-L matching + SAM-B segmentation.
                      For each (query, seed, config):
                        k=1 : run SAM-B on each of the 6 references  -> 6 Dice
                        k=3 : mean consensus of the first 3 refs    -> 1 Dice
                        k=6 : mean consensus of all 6 refs          -> 1 Dice
                      Total SAM calls = 40 * 10 * 3 * 8 = 9600.
                      Checkpointed to reports/phaseT_wp1_holdout/phase2_raw_per_call.csv.

    Phase 3 (CPU)  -- metric computation. Reads the raw checkpoint and emits the
                      8 WP1 metric CSVs (per_query_per_subset, within_query
                      reference variance, pairwise deviation, worst subset,
                      cvar, prediction consistency, reference count scaling,
                      cost scaling).

    Phase 4 (CPU)  -- WP1 gate evaluation (§5.6, 10 conditions).
                      -> gateWP1_summary.json, gateWP1_summary.md, run_metadata.json

The script reuses the proven DINOv2-L + SAM-B infrastructure from Phase R Gate
R1 and Phase S Track B (``FeatureCache``, ``build_pair_state_r2``,
``build_consensus_pool``, ``run_config``, ``PromptConfig``, ``FROZEN_CONFIGS``,
``resolve_case_paths``, mask loading, component recall). Those helpers are
copied verbatim from the reference scripts to keep this file self-contained and
runnable as ``python src/adapromptset/scripts/run_phaseT_wp1_holdout.py``.

CLI::

    python src/adapromptset/scripts/run_phaseT_wp1_holdout.py \
        --phase {1,2,3,4,all} --smoke --device {cpu,cuda} \
        --out-dir reports/phaseT_wp1_holdout

``--smoke`` restricts to 2 queries (and 2 seeds) for fast verification.
"""

from __future__ import annotations

import argparse
import base64
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
from itertools import combinations
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
from scipy.stats import spearmanr  # noqa: E402

# ---------------------------------------------------------------------------
# Paths + constants
# ---------------------------------------------------------------------------

SAM_CHECKPOINT = WORKSPACE_ROOT / "assets" / "checkpoints" / "sam_vit_b.pth"
DINOV2_CHECKPOINT = WORKSPACE_ROOT / "assets" / "checkpoints" / "dinov2_vitl14_pretrain.pth"
MONUSAC_ROOT = WORKSPACE_ROOT / "code" / "datasets" / "MoNuSAC" / "processed"

HOLDOUT_QUERY_MANIFEST = PROJECT_ROOT / "manifests" / "phaseY" / "confirmatory_holdout_queries.csv"
REFERENCE_POOL_MANIFEST = PROJECT_ROOT / "manifests" / "phaseY" / "reference_pool.csv"
PHASET_MANIFEST_DIR = PROJECT_ROOT / "manifests" / "phaseT"

POSITIVE_LIMIT = 100
NEGATIVE_LIMIT = 5
MIN_POSITIVE_DISTANCE = 4.0

CATASTROPHIC_DICE_THRESHOLD = 0.1
SEED_COUNT = 10                # 10 fixed resampling seeds (0..9)
REFS_PER_QUERY = 6
K_VALUES = (1, 3, 6)
CONSENSUS_METHOD = "B2_mean"   # mean aggregation of contrast maps
PRED_MASK_DOWN = 128           # prediction masks downsampled to 128x128 for J_pred

# Gate thresholds (§5.6)
GAP_THRESHOLD = 0.05           # cond1: best-worst gap threshold
MEAN_REF_STD_THRESHOLD = 0.02  # cond2
V_REF_FRAC_THRESHOLD = 0.10    # cond3
CONSENSUS_GAIN_THRESHOLD = 0.010  # cond4
CVAR_TOLERANCE = 0.005         # cond7
DISAGREEMENT_RHO_THRESHOLD = 0.25  # cond9
BOOTSTRAP_ITERS = 10000        # cond5

GATE_PASS = "PHASET_WP1_MONUSAC_HOLDOUT_CONFIRMED"
GATE_FAIL = "PHASET_WP1_MONUSAC_HOLDOUT_FAIL"


# ---------------------------------------------------------------------------
# Frozen prompt configs (copied from gateR1 / TrackB)
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
# Case ID -> path resolution (copied from gateR1)
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
# Image / mask loading (copied from gateR1)
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
# Component labels for recall computation (copied from gateR1)
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
# SAM predictor (copied from gateR1)
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
# DINOv2 feature cache (copied from gateR1)
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
# Contrast maps + per-reference pair state (copied from TrackB)
# ---------------------------------------------------------------------------

def _zscore_np(grid: np.ndarray) -> np.ndarray:
    """Per-grid standardization (population std, ddof=0)."""
    mean = float(grid.mean())
    std = float(grid.std())
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


def _b1_index(query_case_id: str, seed: int, n_refs: int) -> int:
    """Deterministic index of the random single reference (B1 baseline)."""
    digest = int(hashlib.sha256(query_case_id.encode("utf-8")).hexdigest(), 16)
    rng = random.Random(20260722 + (digest % (2**31)) + seed)
    return rng.randrange(n_refs)


# ---------------------------------------------------------------------------
# Point selection (copied from gateR1)
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


def topk_candidate_grid_indices(pools: CandidatePools, k: int) -> set[tuple[int, int]]:
    """Return the (row, col) grid indices of the top-K positive candidates."""
    indices = [c.grid_index for c in pools.positives]
    return set(indices[:k] if k <= len(indices) else indices)


# ---------------------------------------------------------------------------
# Config execution: SAM predict + metrics (copied from gateR1)
# ---------------------------------------------------------------------------

@dataclass
class ConfigRecord:
    """Result of running one config on one (query, seed, reference/subset)."""
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
    """Run one prompt config on one reference (or consensus pool) and return metrics."""
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
# Instability helpers across the 6 references (copied from gateR1)
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


# ---------------------------------------------------------------------------
# Prediction mask encoding (downsampled 128x128, packbits, base64)
# ---------------------------------------------------------------------------

def _encode_pred_mask(mask_np: np.ndarray | None) -> str:
    """Encode a 518x518 bool mask as a 128x128 packbits base64 string."""
    if mask_np is None:
        return ""
    arr = np.asarray(mask_np, dtype=np.uint8)
    img = Image.fromarray(arr).resize(
        (PRED_MASK_DOWN, PRED_MASK_DOWN), Image.Resampling.NEAREST
    )
    small = np.asarray(img, dtype=bool)
    packed = np.packbits(small.ravel())
    return base64.b64encode(packed.tobytes()).decode("ascii")


def _decode_pred_mask(b64: str) -> np.ndarray | None:
    if not b64:
        return None
    packed = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    bits = np.unpackbits(packed)[: PRED_MASK_DOWN * PRED_MASK_DOWN]
    return bits.reshape(PRED_MASK_DOWN, PRED_MASK_DOWN).astype(bool)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _row_float(row: dict[str, str], key: str) -> float | None:
    s = row.get(key, "")
    return float(s) if s not in ("", None) else None


def _fmt(x: float | None, n: int = 6) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{x:.{n}f}"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _open_csv(path: Path, fields: list[str]) -> tuple[Any, Any]:
    is_append = path.is_file()
    handle = path.open("a" if is_append else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fields)
    if not is_append:
        writer.writeheader()
    return handle, writer


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def load_holdout_queries() -> list[dict[str, str]]:
    """Load the 40 holdout query rows (with patient column)."""
    rows = _read_csv(HOLDOUT_QUERY_MANIFEST)
    # Sort by case_id for deterministic ordering (train_009..train_197 range).
    rows.sort(key=lambda r: r["case_id"])
    return rows


def load_reference_pool() -> list[dict[str, str]]:
    """Load the reference pool rows (with patient column)."""
    return _read_csv(REFERENCE_POOL_MANIFEST)


# ---------------------------------------------------------------------------
# Phase 1: Reference resampling
# ---------------------------------------------------------------------------

def _sample_refs_for_seed(
    query_case_id: str,
    query_patient: str,
    eligible: list[dict[str, str]],
    seed_index: int,
    k: int = REFS_PER_QUERY,
) -> list[dict[str, str]]:
    """Deterministic patient-disjoint reference sample for one (query, seed)."""
    digest = int(
        hashlib.sha256(f"{query_case_id}|{seed_index}".encode("utf-8")).hexdigest(), 16
    )
    rng = random.Random(20260723 + (digest % (2**31)))
    pool = list(eligible)
    rng.shuffle(pool)
    if len(pool) < k:
        raise RuntimeError(
            f"Not enough patient-disjoint references for {query_case_id} "
            f"(have {len(pool)}, need {k})"
        )
    return pool[:k]


def run_phase1(*, seeds: range, max_queries: int | None = None) -> dict[str, Any]:
    """Generate the 10 resampling manifests under manifests/phaseT/."""
    PHASET_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    queries = load_holdout_queries()
    if max_queries is not None:
        queries = queries[:max_queries]
    ref_pool = load_reference_pool()

    fields = [
        "query_case_id", "seed", "reference_index",
        "reference_case_id", "reference_patient",
    ]
    total_rows = 0
    leakage_violations = 0
    for seed_index in seeds:
        out_path = PHASET_MANIFEST_DIR / f"holdout_resampling_seed_{seed_index}.csv"
        rows: list[dict[str, Any]] = []
        for q in queries:
            qid = q["case_id"]
            qpatient = q["patient"]
            eligible = [r for r in ref_pool if r["patient"] != qpatient]
            sample = _sample_refs_for_seed(qid, qpatient, eligible, seed_index)
            for idx, ref in enumerate(sample):
                if ref["patient"] == qpatient:
                    leakage_violations += 1
                rows.append({
                    "query_case_id": qid,
                    "seed": seed_index,
                    "reference_index": idx,
                    "reference_case_id": ref["case_id"],
                    "reference_patient": ref["patient"],
                })
        _write_csv(out_path, fields, rows)
        total_rows += len(rows)
        print(f"[phaseT-WP1] phase1 seed {seed_index}: wrote {len(rows)} rows -> {out_path}")

    print(f"[phaseT-WP1] phase1 done: {len(queries)} queries x {len(seeds)} seeds, "
          f"{total_rows} total ref assignments, leakage_violations={leakage_violations}")
    return {
        "query_count": len(queries),
        "seed_count": len(seeds),
        "total_rows": total_rows,
        "leakage_violations": leakage_violations,
    }


def load_holdout_resampling(seed_index: int) -> list[dict[str, str]]:
    path = PHASET_MANIFEST_DIR / f"holdout_resampling_seed_{seed_index}.csv"
    return _read_csv(path)


def refs_for_query_seed(seed_index: int, query_case_id: str) -> list[dict[str, str]]:
    rows = load_holdout_resampling(seed_index)
    return [r for r in rows if r["query_case_id"] == query_case_id]


# ---------------------------------------------------------------------------
# Phase 2: DINOv2-L matching + SAM-B segmentation
# ---------------------------------------------------------------------------

RAW_PER_CALL_FIELDS = [
    "query_case_id", "seed", "config_name", "k", "subset_id",
    "reference_case_id", "reference_index",
    "dice", "iou", "component_recall", "target_component_count", "hit_components",
    "sam_score", "runtime_ms", "status", "n_pos_points", "n_neg_points",
    "pred_mask_b64_128",
]

EPISODE_INSTABILITY_FIELDS = [
    "query_case_id", "seed", "config_name",
    "matching_map_disagreement", "candidate_jaccard_mean_k1",
    "prompt_overlap_mean_k1", "prediction_jaccard_k1_within",
    "mean_dice_k1", "std_dice_k1", "min_dice_k1", "max_dice_k1",
    "best_worst_gap_k1", "catastrophic_failure_rate_k1", "n_references_k1",
]


def _load_raw_checkpoint(path: Path) -> dict[tuple, dict[str, str]]:
    """Return {(query, seed, config, k, subset_id) -> row} for the raw per-call CSV."""
    done: dict[tuple, dict[str, str]] = {}
    if not path.is_file():
        return done
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["query_case_id"],
                int(row["seed"]),
                row["config_name"],
                int(row["k"]),
                row["subset_id"],
            )
            done[key] = row
    return done


def _load_episode_checkpoint(path: Path) -> set[tuple[str, int, str]]:
    if not path.is_file():
        return set()
    done: set[tuple[str, int, str]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            done.add((row["query_case_id"], int(row["seed"]), row["config_name"]))
    return done


def _build_episode_metrics(
    query_case_id: str,
    seed: int,
    config: PromptConfig,
    k1_records: list[ConfigRecord],
    k1_masks_128: list[np.ndarray | None],
    sfg_maps: list[np.ndarray | None],
    candidate_sets: list[set[tuple[int, int]] | None],
) -> dict[str, Any]:
    """Compute per-(query, seed, config) episode instability (k=1 reference set)."""
    dices = [r.dice for r in k1_records if r.dice is not None]
    n_refs = len(k1_records)
    if dices:
        mean_dice = _mean(dices)
        std_dice = _std(dices)
        min_dice = min(dices)
        max_dice = max(dices)
        best_worst_gap = max_dice - min_dice
        cat = sum(1 for d in dices if d < CATASTROPHIC_DICE_THRESHOLD) / len(dices)
    else:
        mean_dice = std_dice = min_dice = max_dice = best_worst_gap = 0.0
        cat = 1.0

    prompt_sets = [set(r.pos_coords) if r.pos_coords else set() for r in k1_records]
    return {
        "query_case_id": query_case_id,
        "seed": seed,
        "config_name": config.name,
        "matching_map_disagreement": round(_pairwise_mean_pearson_disagreement(sfg_maps), 6),
        "candidate_jaccard_mean_k1": round(_pairwise_mean_jaccard_sets(candidate_sets), 6),
        "prompt_overlap_mean_k1": round(_pairwise_mean_jaccard_sets(prompt_sets), 6),
        "prediction_jaccard_k1_within": round(
            _pairwise_mean_jaccard_masks(k1_masks_128), 6
        ),
        "mean_dice_k1": round(mean_dice, 6),
        "std_dice_k1": round(std_dice, 6),
        "min_dice_k1": round(min_dice, 6),
        "max_dice_k1": round(max_dice, 6),
        "best_worst_gap_k1": round(best_worst_gap, 6),
        "catastrophic_failure_rate_k1": round(cat, 6),
        "n_references_k1": n_refs,
    }


def _record_to_raw_row(
    *,
    query_case_id: str,
    seed: int,
    config: PromptConfig,
    k: int,
    subset_id: str,
    reference_case_id: str,
    reference_index: int | str,
    rec: ConfigRecord,
) -> dict[str, Any]:
    return {
        "query_case_id": query_case_id,
        "seed": seed,
        "config_name": config.name,
        "k": k,
        "subset_id": subset_id,
        "reference_case_id": reference_case_id,
        "reference_index": reference_index,
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
        "pred_mask_b64_128": _encode_pred_mask(rec.pred_mask),
    }


def _raw_row_to_record(row: dict[str, str], target_component_count: int) -> ConfigRecord:
    """Reconstruct a ConfigRecord from a checkpointed raw row (for masks/dice)."""
    dice = _row_float(row, "dice")
    iou = _row_float(row, "iou")
    recall = _row_float(row, "component_recall")
    hit = row.get("hit_components", "")
    hit_components = int(hit) if hit not in ("", None) else None
    sam_score = _row_float(row, "sam_score")
    rt = _row_float(row, "runtime_ms")
    status = row.get("status", "OK")
    npos = int(row.get("n_pos_points", "0") or 0)
    nneg = int(row.get("n_neg_points", "0") or 0)
    mask = _decode_pred_mask(row.get("pred_mask_b64_128", ""))
    return ConfigRecord(
        dice=dice, iou=iou, component_recall=recall,
        target_component_count=target_component_count, hit_components=hit_components,
        sam_score=sam_score, runtime_ms=float(rt) if rt is not None else 0.0,
        status=status, n_pos_points=npos, n_neg_points=nneg,
        pos_coords=[], pred_mask=mask,
    )


def run_phase2(
    *,
    feature_cache: FeatureCache,
    predictor: Any,
    out_dir: Path,
    seeds: range,
    max_queries: int | None = None,
) -> dict[str, Any]:
    """Run DINOv2-L matching + SAM-B segmentation for all (query, seed, config)."""
    raw_path = out_dir / "phase2_raw_per_call.csv"
    episode_path = out_dir / "phase2_episode_instability.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_done = _load_raw_checkpoint(raw_path)
    episode_done = _load_episode_checkpoint(episode_path)

    raw_handle, raw_writer = _open_csv(raw_path, RAW_PER_CALL_FIELDS)
    ep_handle, ep_writer = _open_csv(episode_path, EPISODE_INSTABILITY_FIELDS)

    queries = load_holdout_queries()
    if max_queries is not None:
        queries = queries[:max_queries]

    total_sam_calls = 0
    total_episodes = 0
    start = time.perf_counter()

    for qi, qrow in enumerate(queries):
        query_case_id = qrow["case_id"]
        query_paths = resolve_case_paths(query_case_id)
        query_image = load_rgb_518(query_paths["image"])
        predictor.set_image(np.asarray(query_image))
        target = load_mask_518(query_paths["mask"])
        target_labels = _component_labels(target)
        target_component_count = len(set(target_labels.values()))

        # Pair-state cache scoped to this query (refs differ across seeds).
        pair_cache: dict[str, PairStateR2] = {}

        for seed in seeds:
            ref_rows = refs_for_query_seed(seed, query_case_id)
            if not ref_rows:
                continue
            # Ensure pair states exist for all 6 references of this seed.
            ref_case_ids = [rr["reference_case_id"] for rr in ref_rows]
            for ref_id in ref_case_ids:
                if ref_id not in pair_cache:
                    try:
                        pair_cache[ref_id] = build_pair_state_r2(
                            query_case_id, ref_id, feature_cache
                        )
                    except Exception as exc:
                        print(f"[phaseT-WP1] pair error {query_case_id}/{ref_id}: "
                              f"{type(exc).__name__}: {exc}")
                        pair_cache[ref_id] = None  # type: ignore[assignment]

            for config in FROZEN_CONFIGS:
                ep_key = (query_case_id, seed, config.name)
                # Build the 6 pair states (or None) in manifest order.
                pair_states: list[PairStateR2 | None] = [
                    pair_cache.get(rr["reference_case_id"]) for rr in ref_rows
                ]

                # ----- k=1: 6 individual references -------------------------
                k1_records: list[ConfigRecord] = []
                k1_masks_128: list[np.ndarray | None] = []
                sfg_maps: list[np.ndarray | None] = []
                candidate_sets: list[set[tuple[int, int]] | None] = []

                for ri, rr in enumerate(ref_rows):
                    ref_id = rr["reference_case_id"]
                    ref_index = rr["reference_index"]
                    subset_id = str(ref_index)
                    key = (query_case_id, seed, config.name, 1, subset_id)

                    if key in raw_done:
                        rec = _raw_row_to_record(
                            raw_done[key], target_component_count
                        )
                    else:
                        ps = pair_states[ri]
                        if ps is None:
                            rec = ConfigRecord(
                                dice=None, iou=None, component_recall=None,
                                target_component_count=target_component_count,
                                hit_components=None, sam_score=None, runtime_ms=0.0,
                                status="POOL_ERROR", n_pos_points=0, n_neg_points=0,
                                pos_coords=[], pred_mask=None,
                            )
                        else:
                            rec = run_config(
                                predictor, config, ps.pools, target,
                                target_labels, target_component_count,
                            )
                            if rec.status == "OK":
                                total_sam_calls += 1
                        row_dict = _record_to_raw_row(
                            query_case_id=query_case_id, seed=seed, config=config,
                            k=1, subset_id=subset_id, reference_case_id=ref_id,
                            reference_index=ref_index, rec=rec,
                        )
                        raw_writer.writerow(row_dict)
                        raw_done[key] = {k: str(v) for k, v in row_dict.items()}

                    k1_records.append(rec)
                    k1_masks_128.append(rec.pred_mask)
                    ps = pair_states[ri]
                    sfg_maps.append(ps.s_fg if ps is not None else None)
                    candidate_sets.append(
                        topk_candidate_grid_indices(ps.pools, config.k_positive)
                        if ps is not None else None
                    )

                raw_handle.flush()

                # ----- k=3: mean consensus of the first 3 references --------
                first3 = [0, 1, 2]
                key3 = (query_case_id, seed, config.name, 3, "first3")
                if key3 in raw_done:
                    rec3 = _raw_row_to_record(raw_done[key3], target_component_count)
                else:
                    first3_states = [pair_states[i] for i in first3
                                     if pair_states[i] is not None]
                    if not first3_states:
                        rec3 = ConfigRecord(
                            dice=None, iou=None, component_recall=None,
                            target_component_count=target_component_count,
                            hit_components=None, sam_score=None, runtime_ms=0.0,
                            status="POOL_ERROR", n_pos_points=0, n_neg_points=0,
                            pos_coords=[], pred_mask=None,
                        )
                    else:
                        contrast3 = np.mean(
                            np.stack([ps.contrast for ps in first3_states], axis=0),
                            axis=0,
                        )
                        pool3 = build_consensus_pool(contrast3, config)
                        rec3 = run_config(
                            predictor, config, pool3, target,
                            target_labels, target_component_count,
                        )
                        if rec3.status == "OK":
                            total_sam_calls += 1
                    row3 = _record_to_raw_row(
                        query_case_id=query_case_id, seed=seed, config=config,
                        k=3, subset_id="first3", reference_case_id=",".join(
                            ref_rows[i]["reference_case_id"] for i in first3
                        ),
                        reference_index="0-2", rec=rec3,
                    )
                    raw_writer.writerow(row3)
                    raw_done[key3] = {k: str(v) for k, v in row3.items()}

                # ----- k=6: mean consensus of all 6 references --------------
                key6 = (query_case_id, seed, config.name, 6, "all6")
                if key6 in raw_done:
                    rec6 = _raw_row_to_record(raw_done[key6], target_component_count)
                else:
                    valid_states = [ps for ps in pair_states if ps is not None]
                    if not valid_states:
                        rec6 = ConfigRecord(
                            dice=None, iou=None, component_recall=None,
                            target_component_count=target_component_count,
                            hit_components=None, sam_score=None, runtime_ms=0.0,
                            status="POOL_ERROR", n_pos_points=0, n_neg_points=0,
                            pos_coords=[], pred_mask=None,
                        )
                    else:
                        contrast6 = np.mean(
                            np.stack([ps.contrast for ps in valid_states], axis=0),
                            axis=0,
                        )
                        pool6 = build_consensus_pool(contrast6, config)
                        rec6 = run_config(
                            predictor, config, pool6, target,
                            target_labels, target_component_count,
                        )
                        if rec6.status == "OK":
                            total_sam_calls += 1
                    row6 = _record_to_raw_row(
                        query_case_id=query_case_id, seed=seed, config=config,
                        k=6, subset_id="all6",
                        reference_case_id=",".join(ref_rows[i]["reference_case_id"]
                                                   for i in range(len(ref_rows))),
                        reference_index="0-5", rec=rec6,
                    )
                    raw_writer.writerow(row6)
                    raw_done[key6] = {k: str(v) for k, v in row6.items()}
                raw_handle.flush()

                # ----- Episode instability (cheap, recomputable) ------------
                if ep_key not in episode_done:
                    ep_row = _build_episode_metrics(
                        query_case_id, seed, config, k1_records, k1_masks_128,
                        sfg_maps, candidate_sets,
                    )
                    ep_writer.writerow(ep_row)
                    ep_handle.flush()
                    episode_done.add(ep_key)
                total_episodes += 1

        elapsed = time.perf_counter() - start
        print(f"[phaseT-WP1] phase2 query {qi + 1}/{len(queries)} done "
              f"({query_case_id}) episodes={total_episodes} "
              f"sam_calls={total_sam_calls} elapsed={elapsed:.1f}s")

    raw_handle.close()
    ep_handle.close()
    elapsed = time.perf_counter() - start
    return {
        "query_count": len(queries),
        "seed_count": len(seeds),
        "config_count": len(FROZEN_CONFIGS),
        "total_episodes": total_episodes,
        "total_sam_calls": total_sam_calls,
        "elapsed_seconds": round(elapsed, 3),
    }


# ---------------------------------------------------------------------------
# Phase 3: metric computation
# ---------------------------------------------------------------------------

def _load_raw_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["seed"] = int(row["seed"])
            row["k"] = int(row["k"])
            row["dice"] = _row_float(row, "dice")
            row["runtime_ms"] = _row_float(row, "runtime_ms")
            rows.append(row)
    return rows


def _load_episode_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["seed"] = int(row["seed"])
            for key in (
                "matching_map_disagreement", "candidate_jaccard_mean_k1",
                "prompt_overlap_mean_k1", "prediction_jaccard_k1_within",
                "mean_dice_k1", "std_dice_k1", "min_dice_k1", "max_dice_k1",
                "best_worst_gap_k1", "catastrophic_failure_rate_k1",
            ):
                row[key] = _row_float(row, key) or 0.0
            row["n_references_k1"] = int(row.get("n_references_k1", "0") or 0)
            rows.append(row)
    return rows


def compute_cfg_k_metrics(
    raw_rows: list[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    """Compute all per-(config, k) metrics from the raw per-call rows.

    For k=1, within-episode reference variance (Var over 6 refs) is averaged
    over seeds then over queries. For k=3 / k=6, the per-query variance is the
    Var over the 10 resampling seeds (one consensus Dice per seed).
    """
    # Index rows by (config, k, query, seed) -> list of dice.
    by_cs: dict[tuple[str, int, str, int], list[float]] = {}
    for r in raw_rows:
        d = r["dice"]
        if d is None:
            continue
        key = (r["config_name"], r["k"], r["query_case_id"], r["seed"])
        by_cs.setdefault(key, []).append(d)

    metrics: dict[tuple[str, int], dict[str, Any]] = {}
    for cfg in [c.name for c in FROZEN_CONFIGS]:
        for k in K_VALUES:
            # collect per (query, seed) episode dices
            episodes: dict[tuple[str, int], list[float]] = {}
            for (c, kk, q, s), ds in by_cs.items():
                if c == cfg and kk == k:
                    episodes[(q, s)] = ds
            queries = sorted({q for (q, _s) in episodes})

            # For k=1 the unit of reference variation is the individual ref
            # (6 per episode); std/gap/mad/var are computed within each
            # (query, seed) episode then averaged over (query, seed).
            # For k=3 / k=6 each episode has a single consensus Dice, so the
            # unit of variation is the resampling seed: std/gap/mad/var are
            # computed per query across the 10 seed values then averaged over
            # queries.
            per_unit_std: list[float] = []
            per_unit_gap: list[float] = []
            per_unit_mad: list[float] = []
            per_query_var: list[float] = []   # within-query reference variance
            per_query_means: list[float] = []
            per_query_worst: list[float] = []

            for q in queries:
                seed_lists = {
                    s: episodes[(q, s)]
                    for (qq, s) in episodes
                    if qq == q and episodes.get((q, s))
                }
                if not seed_lists:
                    continue
                q_ds = [d for ds in seed_lists.values() for d in ds]
                per_query_means.append(float(np.mean(q_ds)))
                per_query_worst.append(min(q_ds))

                if k == 1:
                    # within-episode (over 6 refs) stats, averaged over seeds
                    ep_vars = [float(np.var(ds)) for ds in seed_lists.values() if ds]
                    if ep_vars:
                        per_query_var.append(float(np.mean(ep_vars)))
                    for ds in seed_lists.values():
                        if not ds:
                            continue
                        per_unit_std.append(float(np.std(ds)))
                        per_unit_gap.append(max(ds) - min(ds))
                        if len(ds) >= 2:
                            per_unit_mad.append(
                                float(np.mean([abs(a - b) for a, b in combinations(ds, 2)]))
                            )
                        else:
                            per_unit_mad.append(0.0)
                else:
                    # k=3 / k=6: one consensus Dice per seed -> stats across seeds
                    per_seed_vals = [float(np.mean(ds)) for ds in seed_lists.values() if ds]
                    if per_seed_vals:
                        per_query_var.append(float(np.var(per_seed_vals)))
                        per_unit_std.append(float(np.std(per_seed_vals)))
                        per_unit_gap.append(max(per_seed_vals) - min(per_seed_vals))
                        if len(per_seed_vals) >= 2:
                            per_unit_mad.append(float(np.mean(
                                [abs(a - b) for a, b in combinations(per_seed_vals, 2)]
                            )))
                        else:
                            per_unit_mad.append(0.0)

            all_flat = [d for ds in episodes.values() for d in ds]

            v_ref = float(np.mean(per_query_var)) if per_query_var else 0.0
            v_query = float(np.var(per_query_means)) if len(per_query_means) > 1 else 0.0
            v_ref_frac = v_ref / (v_ref + v_query) if (v_ref + v_query) > 0 else 0.0
            mean_dice = float(np.mean(all_flat)) if all_flat else 0.0
            mean_within_query_std = float(np.mean(per_unit_std)) if per_unit_std else 0.0
            mean_best_worst_gap = float(np.mean(per_unit_gap)) if per_unit_gap else 0.0
            mean_mad = float(np.mean(per_unit_mad)) if per_unit_mad else 0.0
            mean_worst_dice = float(np.mean(per_query_worst)) if per_query_worst else 0.0
            min_worst_dice = float(np.min(per_query_worst)) if per_query_worst else 0.0
            p10_worst = (
                float(np.percentile(per_query_worst, 10)) if per_query_worst else 0.0
            )
            if all_flat:
                sorted_flat = sorted(all_flat)
                n_tail = max(1, int(math.ceil(0.1 * len(sorted_flat))))
                cvar_0_1 = float(np.mean(sorted_flat[:n_tail]))
            else:
                cvar_0_1 = 0.0
            cat_rate = (
                float(np.mean([1.0 if d < CATASTROPHIC_DICE_THRESHOLD else 0.0
                               for d in all_flat]))
                if all_flat else 0.0
            )

            metrics[(cfg, k)] = {
                "n_queries": len(queries),
                "mean_dice": mean_dice,
                "v_ref": v_ref,
                "v_query": v_query,
                "v_ref_frac": v_ref_frac,
                "mean_within_query_std": mean_within_query_std,
                "mean_best_worst_gap": mean_best_worst_gap,
                "mean_mad": mean_mad,
                "mean_worst_dice": mean_worst_dice,
                "min_worst_dice": min_worst_dice,
                "p10_worst_dice": p10_worst,
                "cvar_0_1": cvar_0_1,
                "catastrophic_rate": cat_rate,
                "n_total": len(all_flat),
            }
    return metrics


def _build_b1_consensus_pairs(
    raw_rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, list[float]]]:
    """For each (query, config), collect B1-single k=1 Dice and k=6 consensus Dice
    across the 10 seeds (paired by seed)."""
    by_qcs: dict[tuple[str, str, int], list[tuple[str, float]]] = {}
    for r in raw_rows:
        d = r["dice"]
        if d is None:
            continue
        by_qcs.setdefault(
            (r["query_case_id"], r["config_name"], r["seed"]), []
        ).append((str(r["k"]) + "|" + r["subset_id"], d))

    pairs: dict[tuple[str, str], dict[str, list[float]]] = {}
    for (q, cfg, seed), entries in by_qcs.items():
        # B1 single index for this (query, seed).
        b1 = _b1_index(q, seed, REFS_PER_QUERY)
        b1_key = f"1|{b1}"
        k6_entries = [d for (k, d) in entries if k == "6|all6"]
        b1_entries = [d for (k, d) in entries if k == b1_key]
        if not b1_entries or not k6_entries:
            continue
        d_b1 = b1_entries[0]
        d_k6 = k6_entries[0]
        slot = pairs.setdefault((q, cfg), {"b1_single": [], "consensus_k6": []})
        slot["b1_single"].append(d_b1)
        slot["consensus_k6"].append(d_k6)
    return pairs


def _compute_prediction_consistency(
    raw_rows: list[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    """Cross-seed prediction Jaccard for B1-single (k=1) and consensus (k=6).

    For each (query, config): collect the 10 B1-single prediction masks (one
    per seed) and the 10 k=6 consensus masks, then compute mean pairwise
    Jaccard at 128x128. Returns per-(config, k) mean J_pred over queries.
    """
    by_qcs: dict[tuple[str, str, int], dict[str, np.ndarray | None]] = {}
    for r in raw_rows:
        q = r["query_case_id"]
        cfg = r["config_name"]
        seed = r["seed"]
        k = r["k"]
        sid = r["subset_id"]
        mask = _decode_pred_mask(r.get("pred_mask_b64_128", ""))
        if k == 6 and sid == "all6":
            by_qcs.setdefault((q, cfg, seed), {})["consensus_k6"] = mask
        elif k == 1:
            b1 = _b1_index(q, seed, REFS_PER_QUERY)
            if str(b1) == sid:
                by_qcs.setdefault((q, cfg, seed), {})["b1_single"] = mask

    # group by (query, config)
    by_qc: dict[tuple[str, str], dict[str, list[np.ndarray]]] = {}
    for (q, cfg, _s), masks in by_qcs.items():
        slot = by_qc.setdefault((q, cfg), {"b1_single": [], "consensus_k6": []})
        if masks.get("b1_single") is not None:
            slot["b1_single"].append(masks["b1_single"])
        if masks.get("consensus_k6") is not None:
            slot["consensus_k6"].append(masks["consensus_k6"])

    per_cfg: dict[str, dict[str, list[float]]] = {
        c.name: {"b1_single": [], "consensus_k6": []} for c in FROZEN_CONFIGS
    }
    for (q, cfg), slot in by_qc.items():
        if len(slot["b1_single"]) >= 2:
            per_cfg[cfg]["b1_single"].append(
                _pairwise_mean_jaccard_masks(slot["b1_single"])
            )
        if len(slot["consensus_k6"]) >= 2:
            per_cfg[cfg]["consensus_k6"].append(
                _pairwise_mean_jaccard_masks(slot["consensus_k6"])
            )

    out: dict[tuple[str, int], dict[str, Any]] = {}
    for cfg in [c.name for c in FROZEN_CONFIGS]:
        b1_vals = per_cfg[cfg]["b1_single"]
        k6_vals = per_cfg[cfg]["consensus_k6"]
        out[(cfg, 1)] = {
            "j_pred_type": "cross_seed_b1_single",
            "mean_j_pred": float(np.mean(b1_vals)) if b1_vals else 0.0,
            "n_queries": len(b1_vals),
        }
        out[(cfg, 6)] = {
            "j_pred_type": "cross_seed_consensus_k6",
            "mean_j_pred": float(np.mean(k6_vals)) if k6_vals else 0.0,
            "n_queries": len(k6_vals),
        }
    return out


def _paired_bootstrap_ci_lower(
    diffs_per_query: list[float], n_iter: int = BOOTSTRAP_ITERS, seed: int = 20260723,
) -> float:
    """Paired bootstrap (by query) 2.5th percentile of the mean difference."""
    if not diffs_per_query:
        return 0.0
    arr = np.asarray(diffs_per_query, dtype=np.float64)
    n = len(arr)
    rng = np.random.default_rng(seed)
    means = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        means[i] = float(arr[idx].mean())
    return float(np.percentile(means, 2.5))


def _safe_spearman(a: list[float], b: list[float]) -> tuple[float, float]:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if len(a_arr) < 3 or np.std(a_arr) == 0.0 or np.std(b_arr) == 0.0:
        return float("nan"), float("nan")
    rho, p = spearmanr(a_arr, b_arr)
    if rho is None or (isinstance(rho, float) and math.isnan(rho)):
        return float("nan"), float("nan")
    return float(rho), float(p)


def run_phase3(out_dir: Path) -> dict[str, Any]:
    """Read the Phase 2 checkpoint and write the 8 metric CSVs."""
    raw_rows = _load_raw_rows(out_dir / "phase2_raw_per_call.csv")
    episode_rows = _load_episode_rows(out_dir / "phase2_episode_instability.csv")
    print(f"[phaseT-WP1] phase3 loaded {len(raw_rows)} raw rows, "
          f"{len(episode_rows)} episode rows")

    metrics = compute_cfg_k_metrics(raw_rows)

    # 1. per_query_per_subset_metrics.csv
    pq_rows: list[dict[str, Any]] = []
    for r in raw_rows:
        if r["dice"] is None:
            continue
        pq_rows.append({
            "query_case_id": r["query_case_id"],
            "seed": r["seed"],
            "config_name": r["config_name"],
            "k": r["k"],
            "subset_id": r["subset_id"],
            "subset_dice": _fmt(r["dice"]),
        })
    pq_rows.sort(key=lambda x: (x["config_name"], x["k"], x["query_case_id"], x["seed"], str(x["subset_id"])))
    _write_csv(out_dir / "per_query_per_subset_metrics.csv",
               ["query_case_id", "seed", "config_name", "k", "subset_id", "subset_dice"], pq_rows)

    # 2. within_query_reference_variance.csv
    rows = []
    for cfg in [c.name for c in FROZEN_CONFIGS]:
        for k in K_VALUES:
            m = metrics[(cfg, k)]
            rows.append({
                "config_name": cfg, "k": k, "n_queries": m["n_queries"],
                "V_ref": _fmt(m["v_ref"]),
                "V_query": _fmt(m["v_query"]),
                "V_ref_frac": _fmt(m["v_ref_frac"]),
                "mean_within_query_std": _fmt(m["mean_within_query_std"]),
                "mean_best_worst_gap": _fmt(m["mean_best_worst_gap"]),
            })
    _write_csv(out_dir / "within_query_reference_variance.csv",
               ["config_name", "k", "n_queries", "V_ref", "V_query", "V_ref_frac",
                "mean_within_query_std", "mean_best_worst_gap"], rows)

    # 3. pairwise_reference_deviation.csv
    rows = []
    for cfg in [c.name for c in FROZEN_CONFIGS]:
        for k in K_VALUES:
            m = metrics[(cfg, k)]
            rows.append({"config_name": cfg, "k": k,
                         "mean_mad": _fmt(m["mean_mad"]), "n_queries": m["n_queries"]})
    _write_csv(out_dir / "pairwise_reference_deviation.csv",
               ["config_name", "k", "mean_mad", "n_queries"], rows)

    # 4. worst_subset_metrics.csv
    rows = []
    for cfg in [c.name for c in FROZEN_CONFIGS]:
        for k in K_VALUES:
            m = metrics[(cfg, k)]
            rows.append({
                "config_name": cfg, "k": k,
                "mean_worst_dice": _fmt(m["mean_worst_dice"]),
                "min_worst_dice": _fmt(m["min_worst_dice"]),
                "p10_worst_dice": _fmt(m["p10_worst_dice"]),
            })
    _write_csv(out_dir / "worst_subset_metrics.csv",
               ["config_name", "k", "mean_worst_dice", "min_worst_dice", "p10_worst_dice"], rows)

    # 5. cvar_metrics.csv
    rows = []
    for cfg in [c.name for c in FROZEN_CONFIGS]:
        for k in K_VALUES:
            m = metrics[(cfg, k)]
            rows.append({
                "config_name": cfg, "k": k,
                "cvar_0_1": _fmt(m["cvar_0_1"]),
                "catastrophic_rate": _fmt(m["catastrophic_rate"]),
                "n_total": m["n_total"],
            })
    _write_csv(out_dir / "cvar_metrics.csv",
               ["config_name", "k", "cvar_0_1", "catastrophic_rate", "n_total"], rows)

    # 6. prediction_consistency.csv
    jpred = _compute_prediction_consistency(raw_rows)
    rows = []
    for cfg in [c.name for c in FROZEN_CONFIGS]:
        for k in (1, 6):
            j = jpred[(cfg, k)]
            rows.append({
                "config_name": cfg, "k": k,
                "j_pred_type": j["j_pred_type"],
                "mean_j_pred": _fmt(j["mean_j_pred"]),
                "n_queries": j["n_queries"],
            })
    # also include within-episode k=1 from episode CSV
    for cfg in [c.name for c in FROZEN_CONFIGS]:
        vals = [e["prediction_jaccard_k1_within"] for e in episode_rows
                if e["config_name"] == cfg]
        rows.append({
            "config_name": cfg, "k": 1,
            "j_pred_type": "within_episode_k1",
            "mean_j_pred": _fmt(float(np.mean(vals)) if vals else 0.0),
            "n_queries": len(vals),
        })
    _write_csv(out_dir / "prediction_consistency.csv",
               ["config_name", "k", "j_pred_type", "mean_j_pred", "n_queries"], rows)

    # 7. reference_count_scaling.csv
    rows = []
    for cfg in [c.name for c in FROZEN_CONFIGS]:
        for k in K_VALUES:
            m = metrics[(cfg, k)]
            rows.append({
                "config_name": cfg, "k": k,
                "mean_dice": _fmt(m["mean_dice"]),
                "V_ref": _fmt(m["v_ref"]),
                "worst_dice": _fmt(m["mean_worst_dice"]),
                "CVaR": _fmt(m["cvar_0_1"]),
                "CFR": _fmt(m["catastrophic_rate"]),
            })
    _write_csv(out_dir / "reference_count_scaling.csv",
               ["config_name", "k", "mean_dice", "V_ref", "worst_dice", "CVaR", "CFR"], rows)

    # 8. cost_scaling.csv
    runtime_by: dict[tuple[str, int], list[float]] = {}
    sam_calls_by: dict[tuple[str, int], int] = {}
    for r in raw_rows:
        rt = r["runtime_ms"]
        if rt is None:
            continue
        runtime_by.setdefault((r["config_name"], r["k"]), []).append(rt)
        if r.get("status") == "OK":
            sam_calls_by[(r["config_name"], r["k"])] = (
                sam_calls_by.get((r["config_name"], r["k"]), 0) + 1
            )
    rows = []
    for cfg in [c.name for c in FROZEN_CONFIGS]:
        for k in K_VALUES:
            rts = runtime_by.get((cfg, k), [])
            mean_rt = float(np.mean(rts)) if rts else 0.0
            total_rt = float(np.sum(rts)) if rts else 0.0
            calls = sam_calls_by.get((cfg, k), 0)
            gpu_hours = total_rt / 3.6e9
            rows.append({
                "config_name": cfg, "k": k,
                "sam_calls": calls,
                "mean_runtime_ms": _fmt(mean_rt),
                "total_runtime_ms": _fmt(total_rt),
                "gpu_hours_estimate": _fmt(gpu_hours),
            })
    _write_csv(out_dir / "cost_scaling.csv",
               ["config_name", "k", "sam_calls", "mean_runtime_ms",
                "total_runtime_ms", "gpu_hours_estimate"], rows)

    print("[phaseT-WP1] phase3 wrote 8 metric CSVs")
    return {
        "metrics": metrics,
        "raw_rows": raw_rows,
        "episode_rows": episode_rows,
        "jpred": jpred,
    }


# ---------------------------------------------------------------------------
# Phase 4: WP1 Gate evaluation (§5.6, 10 conditions)
# ---------------------------------------------------------------------------

def _check_leakage() -> tuple[bool, int, int]:
    """Verify reference patients != query patients across all seed manifests."""
    queries = {q["case_id"]: q["patient"] for q in load_holdout_queries()}
    violations = 0
    total = 0
    for s in range(SEED_COUNT):
        rows = load_holdout_resampling(s)
        for r in rows:
            total += 1
            qpatient = queries.get(r["query_case_id"])
            if qpatient is None or r["reference_patient"] == qpatient:
                violations += 1
    return violations == 0, violations, total


def evaluate_gate_wp1(
    metrics: dict[tuple[str, int], dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
    jpred: dict[tuple[str, int], dict[str, Any]],
) -> tuple[dict[str, Any], list[tuple[str, bool, str]]]:
    """Evaluate the 10 WP1 gate conditions."""

    cfg_names = [c.name for c in FROZEN_CONFIGS]
    cond_list: list[tuple[str, bool, str]] = []

    # ----- cond1: >= 50% holdout queries best-worst gap >= 0.05 ---------------
    # Per query: mean over (seed, config) of the k=1 best-worst gap.
    by_q: dict[str, list[float]] = {}
    for e in episode_rows:
        by_q.setdefault(e["query_case_id"], []).append(e["best_worst_gap_k1"])
    per_q_gap = {q: float(np.mean(gs)) for q, gs in by_q.items() if gs}
    if per_q_gap:
        frac1 = float(np.mean([1.0 if g >= GAP_THRESHOLD else 0.0
                               for g in per_q_gap.values()]))
    else:
        frac1 = 0.0
    cond1_pass = frac1 >= 0.50
    cond_list.append((
        "cond1_50pct_queries_best_worst_gap_ge_0p05",
        cond1_pass,
        f"fraction of holdout queries with mean best-worst gap>=0.05 = {frac1:.4f} "
        f"(n={len(per_q_gap)}); threshold 0.50",
    ))

    # ----- cond2: mean within-query reference std >= 0.02 ---------------------
    stds = [e["std_dice_k1"] for e in episode_rows]
    mean_ref_std = float(np.mean(stds)) if stds else 0.0
    cond2_pass = mean_ref_std >= MEAN_REF_STD_THRESHOLD
    cond_list.append((
        "cond2_mean_within_query_ref_std_ge_0p02",
        cond2_pass,
        f"mean within-query reference std (k=1) = {mean_ref_std:.6f}; "
        f"threshold {MEAN_REF_STD_THRESHOLD}",
    ))

    # ----- cond3: V_ref_frac >= 0.10 -----------------------------------------
    frac3 = {cfg: metrics[(cfg, 1)]["v_ref_frac"] for cfg in cfg_names}
    cond3_pass = any(v >= V_REF_FRAC_THRESHOLD for v in frac3.values())
    cond_list.append((
        "cond3_v_ref_frac_ge_0p10",
        cond3_pass,
        f"V_ref_frac (k=1) per config = "
        f"{ {c: round(frac3[c], 4) for c in cfg_names} }; threshold 0.10 (any config)",
    ))

    # ----- cond4: mean consensus Dice - random single Dice >= 0.010 ----------
    pairs = _build_b1_consensus_pairs(raw_rows)
    all_b1 = [d for slot in pairs.values() for d in slot["b1_single"]]
    all_k6 = [d for slot in pairs.values() for d in slot["consensus_k6"]]
    mean_b1 = float(np.mean(all_b1)) if all_b1 else 0.0
    mean_k6 = float(np.mean(all_k6)) if all_k6 else 0.0
    gain = mean_k6 - mean_b1
    cond4_pass = gain >= CONSENSUS_GAIN_THRESHOLD
    cond_list.append((
        "cond4_consensus_gain_ge_0p010",
        cond4_pass,
        f"mean consensus(k=6) Dice = {mean_k6:.6f}, mean B1-single(k=1) Dice = "
        f"{mean_b1:.6f}, gain = {gain:.6f}; threshold {CONSENSUS_GAIN_THRESHOLD}",
    ))

    # ----- cond5: paired bootstrap 95% CI lower bound > 0 --------------------
    # Paired by query: diff_q = mean over (seed, config) of (consensus_k6 - b1).
    diff_by_q: dict[str, list[float]] = {}
    for (q, _cfg), slot in pairs.items():
        if slot["b1_single"] and slot["consensus_k6"]:
            diff_by_q.setdefault(q, []).append(
                float(np.mean(slot["consensus_k6"]))
                - float(np.mean(slot["b1_single"]))
            )
    diffs_per_q = [float(np.mean(ds)) for ds in diff_by_q.values() if ds]
    ci_lower = _paired_bootstrap_ci_lower(diffs_per_q)
    cond5_pass = ci_lower > 0.0
    cond_list.append((
        "cond5_paired_bootstrap_ci_lower_gt_0",
        cond5_pass,
        f"paired bootstrap (by query, {BOOTSTRAP_ITERS} iters) 95% CI lower bound = "
        f"{ci_lower:.6f}; n_queries={len(diffs_per_q)}",
    ))

    # ----- cond6: catastrophic failure not increase --------------------------
    cfr_b1 = float(np.mean([1.0 if d < CATASTROPHIC_DICE_THRESHOLD else 0.0
                            for d in all_b1])) if all_b1 else 0.0
    cfr_k6 = float(np.mean([1.0 if d < CATASTROPHIC_DICE_THRESHOLD else 0.0
                            for d in all_k6])) if all_k6 else 0.0
    cond6_pass = cfr_k6 <= cfr_b1
    cond_list.append((
        "cond6_consensus_cfr_le_single_cfr",
        cond6_pass,
        f"consensus CFR(k=6) = {cfr_k6:.6f}, single CFR(B1) = {cfr_b1:.6f}",
    ))

    # ----- cond7: CVaR_0.1 not decrease --------------------------------------
    def _cvar(vals: list[float]) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        n_tail = max(1, int(math.ceil(0.1 * len(s))))
        return float(np.mean(s[:n_tail]))

    cvar_b1 = _cvar(all_b1)
    cvar_k6 = _cvar(all_k6)
    cond7_pass = cvar_k6 >= (cvar_b1 - CVAR_TOLERANCE)
    cond_list.append((
        "cond7_consensus_cvar_ge_single_cvar_minus_tol",
        cond7_pass,
        f"consensus CVaR_0.1(k=6) = {cvar_k6:.6f}, single CVaR_0.1(B1) = "
        f"{cvar_b1:.6f}, tolerance {CVAR_TOLERANCE}",
    ))

    # ----- cond8: prediction consistency improve -----------------------------
    # mean over configs of cross-seed J_pred(consensus_k6) vs J_pred(b1_single).
    j_single = [jpred[(cfg, 1)]["mean_j_pred"] for cfg in cfg_names]
    j_cons = [jpred[(cfg, 6)]["mean_j_pred"] for cfg in cfg_names]
    mean_j_single = float(np.mean(j_single)) if j_single else 0.0
    mean_j_cons = float(np.mean(j_cons)) if j_cons else 0.0
    cond8_pass = mean_j_cons >= mean_j_single
    cond_list.append((
        "cond8_consensus_j_pred_ge_single_j_pred",
        cond8_pass,
        f"mean cross-seed J_pred: consensus(k=6) = {mean_j_cons:.6f}, "
        f"B1-single(k=1) = {mean_j_single:.6f}",
    ))

    # ----- cond9: disagreement feature Spearman |rho| >= 0.25 ----------------
    # Per config: Spearman(matching_map_disagreement, best_worst_gap_k1).
    cond9_details = {}
    cond9_pass = False
    best_rho = 0.0
    for cfg in cfg_names:
        rs = [e for e in episode_rows if e["config_name"] == cfg]
        feats = [e["matching_map_disagreement"] for e in rs]
        gaps = [e["best_worst_gap_k1"] for e in rs]
        rho, p = _safe_spearman(feats, gaps)
        cond9_details[cfg] = {"rho": rho, "p": p, "n": len(rs)}
        if not math.isnan(rho) and abs(rho) >= best_rho:
            best_rho = abs(rho)
        if not math.isnan(rho) and abs(rho) >= DISAGREEMENT_RHO_THRESHOLD:
            cond9_pass = True
    cond_list.append((
        "cond9_disagreement_feature_spearman_rho_ge_0p25",
        cond9_pass,
        f"best |Spearman(matching_map_disagreement, best_worst_gap_k1)| = "
        f"{best_rho:.4f}; per config "
        f"{ {c: (round(cond9_details[c]['rho'], 4) if not math.isnan(cond9_details[c]['rho']) else 'nan') for c in cfg_names} }; "
        f"threshold {DISAGREEMENT_RHO_THRESHOLD}",
    ))

    # ----- cond10: no data leakage -------------------------------------------
    ok, violations, total = _check_leakage()
    cond10_pass = ok
    cond_list.append((
        "cond10_no_data_leakage",
        cond10_pass,
        f"patient-disjoint check: {violations} violations / {total} reference "
        f"assignments across {SEED_COUNT} seeds",
    ))

    n_pass = sum(1 for _, ok_flag, _ in cond_list if ok_flag)
    decision = GATE_PASS if n_pass == len(cond_list) else GATE_FAIL

    summary = {
        "gate": "Phase T WP1 MoNuSAC Holdout Confirmatory Benchmark (§5.6)",
        "n_conditions": len(cond_list),
        "n_pass": n_pass,
        "n_queries": len(per_q_gap),
        "configs": cfg_names,
        "k_values": list(K_VALUES),
        "seed_count": SEED_COUNT,
        "metrics_summary": {
            "frac_queries_best_worst_gap_ge_0p05": round(frac1, 6),
            "mean_within_query_ref_std": round(mean_ref_std, 6),
            "v_ref_frac_k1": {c: round(frac3[c], 6) for c in cfg_names},
            "mean_consensus_dice_k6": round(mean_k6, 6),
            "mean_b1_single_dice_k1": round(mean_b1, 6),
            "consensus_gain": round(gain, 6),
            "bootstrap_ci_lower": round(ci_lower, 6),
            "cfr_b1_single": round(cfr_b1, 6),
            "cfr_consensus_k6": round(cfr_k6, 6),
            "cvar_b1_single": round(cvar_b1, 6),
            "cvar_consensus_k6": round(cvar_k6, 6),
            "mean_j_pred_b1_single": round(mean_j_single, 6),
            "mean_j_pred_consensus_k6": round(mean_j_cons, 6),
            "best_disagreement_rho": round(best_rho, 6),
            "leakage_violations": violations,
            "leakage_total": total,
        },
        "conditions": {
            name: {"pass": ok_flag, "details": detail}
            for name, ok_flag, detail in cond_list
        },
        "decision": decision,
    }
    return summary, cond_list


def write_gate_summary(
    out_dir: Path,
    summary: dict[str, Any],
    cond_list: list[tuple[str, bool, str]],
    metrics: dict[tuple[str, int], dict[str, Any]],
) -> None:
    json_path = out_dir / "gateWP1_summary.json"
    json_path.write_text(json.dumps({"gate_wp1": summary}, indent=2), encoding="utf-8")

    labels = {
        "cond1_50pct_queries_best_worst_gap_ge_0p05":
            ">= 50% holdout queries best-worst gap >= 0.05",
        "cond2_mean_within_query_ref_std_ge_0p02":
            "mean within-query reference std >= 0.02",
        "cond3_v_ref_frac_ge_0p10": "V_ref_frac >= 0.10",
        "cond4_consensus_gain_ge_0p010":
            "mean consensus Dice - random single Dice >= 0.010",
        "cond5_paired_bootstrap_ci_lower_gt_0":
            "paired bootstrap 95% CI lower bound > 0",
        "cond6_consensus_cfr_le_single_cfr":
            "catastrophic failure not increase (consensus CFR <= single CFR)",
        "cond7_consensus_cvar_ge_single_cvar_minus_tol":
            "CVaR_0.1 not decrease (consensus CVaR >= single CVaR - 0.005)",
        "cond8_consensus_j_pred_ge_single_j_pred":
            "prediction consistency improve (consensus J_pred >= single J_pred)",
        "cond9_disagreement_feature_spearman_rho_ge_0p25":
            "disagreement feature Spearman |rho| >= 0.25",
        "cond10_no_data_leakage": "no data leakage (ref patients != query patients)",
    }

    md: list[str] = []
    md.append("# Phase T WP1 MoNuSAC Holdout Confirmatory Benchmark Summary")
    md.append("")
    md.append(f"Date: {datetime.now().isoformat(timespec='seconds')}")
    md.append(f"Holdout queries: {summary['n_queries']}")
    md.append(f"Resampling seeds: {summary['seed_count']}")
    md.append(f"Prompt configs: {', '.join(summary['configs'])}")
    md.append(f"Reference subset sizes k: {', '.join(str(k) for k in summary['k_values'])}")
    md.append("")
    md.append("## k-scaling design")
    md.append("")
    md.append("| k  | subsets per (query, seed) | source | subset_id |")
    md.append("|----|---------------------------|--------|-----------|")
    md.append("| 1  | 6 (individual refs)        | per-reference s_fg pool | 0..5 |")
    md.append("| 3  | 1 (mean consensus first 3) | B2_mean contrast | first3 |")
    md.append("| 6  | 1 (mean consensus all 6)   | B2_mean contrast | all6 |")
    md.append("")
    md.append("## Within-query reference variance V_ref(k)")
    md.append("")
    md.append("| config | k | n_queries | V_ref | V_query | V_ref_frac | mean_ref_std | best-worst gap |")
    md.append("|--------|---|-----------|-------|---------|------------|--------------|----------------|")
    for cfg in [c.name for c in FROZEN_CONFIGS]:
        for k in K_VALUES:
            m = metrics[(cfg, k)]
            md.append(f"| {cfg} | {k} | {m['n_queries']} | {m['v_ref']:.6f} | "
                      f"{m['v_query']:.6f} | {m['v_ref_frac']:.4f} | "
                      f"{m['mean_within_query_std']:.6f} | {m['mean_best_worst_gap']:.6f} |")
    md.append("")
    md.append("## Mean Dice & tail risk by (config, k)")
    md.append("")
    md.append("| config | k | mean_dice | mean_worst | min_worst | p10_worst | CVaR_0.1 | CFR |")
    md.append("|--------|---|-----------|------------|-----------|-----------|----------|-----|")
    for cfg in [c.name for c in FROZEN_CONFIGS]:
        for k in K_VALUES:
            m = metrics[(cfg, k)]
            md.append(f"| {cfg} | {k} | {m['mean_dice']:.6f} | {m['mean_worst_dice']:.6f} | "
                      f"{m['min_worst_dice']:.6f} | {m['p10_worst_dice']:.6f} | "
                      f"{m['cvar_0_1']:.6f} | {m['catastrophic_rate']:.4f} |")
    md.append("")
    md.append("## WP1 Gate — 10 conditions (§5.6)")
    md.append("")
    md.append("| # | condition | pass |")
    md.append("|---|-----------|------|")
    for i, (name, ok_flag, _detail) in enumerate(cond_list, 1):
        md.append(f"| {i} | {labels.get(name, name)} | {'PASS' if ok_flag else 'FAIL'} |")
    md.append("")
    for i, (name, ok_flag, detail) in enumerate(cond_list, 1):
        md.append(f"**cond{i}** (`{name}`): **{'PASS' if ok_flag else 'FAIL'}** — {detail}")
        md.append("")
    n_pass = sum(1 for _, ok_flag, _ in cond_list if ok_flag)
    md.append(f"## Decision: **{summary['decision']}**")
    md.append("")
    md.append(f"({n_pass}/{len(cond_list)} conditions passed)")
    (out_dir / "gateWP1_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def write_run_metadata(
    out_dir: Path,
    *,
    device: torch.device,
    run_meta: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    elapsed = run_meta.get("elapsed_seconds", 0.0)
    gpu_hours = elapsed / 3600.0 if str(device).startswith("cuda") else 0.0
    metadata = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "runtime_seconds": elapsed,
        "gpu_hours": round(gpu_hours, 6),
        "total_sam_calls": run_meta.get("total_sam_calls", 0),
        "total_episodes": run_meta.get("total_episodes", 0),
        "query_count": run_meta.get("query_count", 0),
        "seed_count": run_meta.get("seed_count", 0),
        "config_count": run_meta.get("config_count", 0),
        "gate_wp1_decision": summary.get("decision"),
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase T WP1 MoNuSAC Holdout Confirmatory Benchmark"
    )
    parser.add_argument("--phase", default="all",
                        choices=["1", "2", "3", "4", "all"])
    parser.add_argument("--smoke", action="store_true",
                        help="Quick verification: 2 queries x 2 seeds.")
    parser.add_argument("--device", default="cpu",
                        choices=["cpu", "cuda", "auto"])
    parser.add_argument(
        "--out-dir", type=Path,
        default=PROJECT_ROOT / "reports" / "phaseT_wp1_holdout",
    )
    parser.add_argument("--max-queries", type=int, default=None)
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        seeds = range(2)
        max_queries = args.max_queries or 2
    else:
        seeds = range(SEED_COUNT)
        max_queries = args.max_queries

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[phaseT-WP1] device={device} seeds={list(seeds)} "
          f"max_queries={max_queries} smoke={args.smoke} phase={args.phase}")

    run_meta: dict[str, Any] = {
        "query_count": 0, "seed_count": len(seeds),
        "config_count": len(FROZEN_CONFIGS),
        "total_episodes": 0, "total_sam_calls": 0, "elapsed_seconds": 0.0,
    }

    # ---- Phase 1: resampling -----------------------------------------------
    if args.phase in ("1", "all"):
        print("=" * 72)
        print("[phaseT-WP1] Phase 1: reference resampling")
        print("=" * 72)
        p1 = run_phase1(seeds=seeds, max_queries=max_queries)
        # Update query count from phase 1 (authoritative for downstream).
        run_meta["query_count"] = p1["query_count"]

    # ---- Phase 2: DINOv2 + SAM-B -------------------------------------------
    if args.phase in ("2", "all"):
        print("=" * 72)
        print("[phaseT-WP1] Phase 2: DINOv2-L matching + SAM-B segmentation")
        print("=" * 72)
        feature_cache = FeatureCache(device=device)
        predictor = load_sam_predictor(device)
        run_meta = run_phase2(
            feature_cache=feature_cache,
            predictor=predictor,
            out_dir=out_dir,
            seeds=seeds,
            max_queries=max_queries,
        )
        print(f"[phaseT-WP1] phase2 done: {run_meta['total_episodes']} episodes, "
              f"{run_meta['total_sam_calls']} SAM calls, "
              f"{run_meta['elapsed_seconds']}s")

    # ---- Phase 3: metrics --------------------------------------------------
    summary: dict[str, Any] = {}
    metrics: dict[tuple[str, int], dict[str, Any]] = {}
    p3: dict[str, Any] | None = None
    if args.phase in ("3", "all"):
        print("=" * 72)
        print("[phaseT-WP1] Phase 3: metric computation")
        print("=" * 72)
        p3 = run_phase3(out_dir)
        metrics = p3["metrics"]
        # If phase 2 did not run this invocation, recover query/seed counts.
        if run_meta["query_count"] == 0:
            qs = {r["query_case_id"] for r in p3["raw_rows"]}
            run_meta["query_count"] = len(qs)
            run_meta["seed_count"] = len({r["seed"] for r in p3["raw_rows"]}) or len(seeds)

    # ---- Phase 4: gate evaluation -----------------------------------------
    if args.phase in ("4", "all"):
        print("=" * 72)
        print("[phaseT-WP1] Phase 4: WP1 gate evaluation (§5.6, 10 conditions)")
        print("=" * 72)
        # Reuse phase-3 results if computed this invocation; otherwise read the
        # on-disk checkpoint (run_phase3 is idempotent).
        if p3 is None:
            p3 = run_phase3(out_dir)
            metrics = p3["metrics"]
        summary, cond_list = evaluate_gate_wp1(
            metrics, p3["raw_rows"], p3["episode_rows"], p3["jpred"]
        )
        write_gate_summary(out_dir, summary, cond_list, metrics)
        write_run_metadata(out_dir, device=device, run_meta=run_meta, summary=summary)

        print()
        for i, (name, ok_flag, detail) in enumerate(cond_list, 1):
            print(f"  cond{i}: {'PASS' if ok_flag else 'FAIL'}  {name}")
            print(f"         {detail}")
        print()
        n_pass = sum(1 for _, ok_flag, _ in cond_list if ok_flag)
        print(f"  {n_pass}/{len(cond_list)} conditions passed")
        print(f"  FINAL STATE: {summary['decision']}")

    if args.phase in ("3", "all") and not summary:
        # Phase 3 only (no gate): still write run metadata.
        write_run_metadata(out_dir, device=device, run_meta=run_meta,
                           summary={"decision": None})

    print()
    print(f"[phaseT-WP1] artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
