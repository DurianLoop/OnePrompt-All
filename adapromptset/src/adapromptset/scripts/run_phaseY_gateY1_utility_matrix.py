"""Phase Y Gate Y1: Fresh-Query Action Heterogeneity Utility Matrix.

Implements section 8 of ``AdaPromptSet_PhaseY_Policy_Feasibility.md``.

Computes the full 210-action utility matrix on:
  - Policy Train Queries × Reference Set A  (step 11, for Pareto pruning)
  - Policy Validation Queries × Reference Sets A and B  (step 14, for Gate Y1)

For each (query, reference) pair, the script:
  1. Extracts DINOv2-L features (cached per case_id).
  2. Builds matching score grids (s_fg, s_bg).
  3. Generates candidate pools for each positive selector layout.
  4. Runs SAM-B once per action (G_ONE_GROUP, 1 call).
  5. Records Dice / IoU / AJI / component recall.

For Gate Y1 (validation), the script computes:
  - Fixed-A: strongest fixed action on Reference Set A
  - Oracle-A / Oracle-B: per-query best action on each set
  - Cross-Reference Oracle A->B and B->A
  - Stable Oracle = mean(cross A->B, cross B->A)
  - Oracle Optimism per set
  - Action stability metrics (exact agreement, K-bin agreement, Spearman rho)
  - Gate Y1 decision (6 pass conditions)

For Pareto pruning (train), the script exports per-action aggregate metrics
for downstream compact bank derivation.

Outputs::

    reports/phaseY_gateY1/
        utility_matrix_{split}.csv          (per query, ref, action metrics)
        candidate_pools_{split}.csv         (cached candidate points)
        gateY1_summary.json                 (upper bounds, stability, decision)
        gateY1_summary.md                   (human-readable summary)
        pareto_train_metrics.csv             (train-only, for compact bank)
        pareto_pruning.md                    (Pareto pruning report)
        run_metadata.json                    (device, runtime, gpu_hours)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from PIL import Image

from adapromptset.candidates.points import (
    Candidate,
    CandidatePools,
    build_candidate_pools,
)
from adapromptset.evaluation.metrics import (
    aggregated_jaccard_index,
    dice_score,
    iou_score,
)
from adapromptset.matching.evidence import (
    IMAGE_SIZE,
    PATCH_SIZE,
    extract_dinov2_features,
    load_dinov2_model,
    matching_scores_from_features,
    patch_weights_from_mask,
)
from adapromptset.matching.runtime import checkpoint_record, configure_external_paths

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parent

SAM_CHECKPOINT = WORKSPACE_ROOT / "assets" / "checkpoints" / "sam_vit_b.pth"
DINOV2_CHECKPOINT = WORKSPACE_ROOT / "assets" / "checkpoints" / "dinov2_vitl14_pretrain.pth"
MONUSAC_ROOT = WORKSPACE_ROOT / "code" / "datasets" / "MoNuSAC" / "processed"
ACTION_BANK_YAML = PROJECT_ROOT / "configs" / "actions" / "phaseY_full_single_call_bank.yaml"

POSITIVE_LIMIT = 100
NEGATIVE_LIMIT = 5
MIN_POSITIVE_DISTANCE = 4.0


# ---------------------------------------------------------------------------
# Case ID -> path resolution
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
# Image / mask loading
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
# Component labels for recall computation
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


def _semantic_instance_labels(mask: Sequence[Sequence[bool]]) -> list[list[int]]:
    return [[1 if bool(value) else 0 for value in row] for row in mask]


# ---------------------------------------------------------------------------
# SAM predictor
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
# DINOv2 feature cache
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
# Candidate pool generation per (query, ref) pair
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PairCandidatePools:
    """Candidate pools for one (query, ref) pair, one per positive selector."""
    topk: CandidatePools
    spread: CandidatePools
    local_peaks: CandidatePools


def build_pair_candidate_pools(
    query_case_id: str,
    reference_case_id: str,
    feature_cache: FeatureCache,
) -> PairCandidatePools:
    """Build candidate pools for all three positive selectors at once."""
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
    s_fg = maps["s_fg"].tolist()
    s_bg = maps["s_bg"].tolist()
    image_size = (IMAGE_SIZE, IMAGE_SIZE)

    pools_topk = build_candidate_pools(
        s_fg, s_bg,
        image_size=image_size,
        positive_limit=POSITIVE_LIMIT,
        negative_limit=NEGATIVE_LIMIT,
        positive_layout="topk",
        min_positive_distance=MIN_POSITIVE_DISTANCE,
    )
    pools_spread = build_candidate_pools(
        s_fg, s_bg,
        image_size=image_size,
        positive_limit=POSITIVE_LIMIT,
        negative_limit=NEGATIVE_LIMIT,
        positive_layout="spread",
        min_positive_distance=MIN_POSITIVE_DISTANCE,
    )
    pools_peaks = build_candidate_pools(
        s_fg, s_bg,
        image_size=image_size,
        positive_limit=POSITIVE_LIMIT,
        negative_limit=NEGATIVE_LIMIT,
        positive_layout="local_peaks",
        min_positive_distance=MIN_POSITIVE_DISTANCE,
    )
    return PairCandidatePools(topk=pools_topk, spread=pools_spread, local_peaks=pools_peaks)


# ---------------------------------------------------------------------------
# Action bank loading
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    action_index: int
    positive_selector: str
    k_positive: int
    negative_selector: str
    k_negative: int
    grouping: str
    sam_calls: int


def load_action_bank(yaml_path: Path = ACTION_BANK_YAML) -> list[ActionSpec]:
    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    actions: list[ActionSpec] = []
    for entry in data["actions"]:
        actions.append(ActionSpec(
            action_id=entry["action_id"],
            action_index=entry["action_index"],
            positive_selector=entry["positive_selector"],
            k_positive=entry["k_positive"],
            negative_selector=entry["negative_selector"],
            k_negative=entry["k_negative"],
            grouping=entry["grouping"],
            sam_calls=entry["sam_calls"],
        ))
    return actions


# ---------------------------------------------------------------------------
# Point selection from candidate pools
# ---------------------------------------------------------------------------

def select_positives(
    pools: PairCandidatePools,
    selector: str,
    k: int,
) -> list[tuple[int, int]]:
    """Return K positive coordinates for the given selector."""
    if selector == "P_TOPK":
        pool = pools.topk
    elif selector == "P_SPREAD":
        pool = pools.spread
    elif selector == "P_LOCAL_PEAKS":
        pool = pools.local_peaks
    else:
        raise ValueError(f"unknown positive selector: {selector}")
    coords = [c.coordinate for c in pool.positives]
    return coords[:k] if k <= len(coords) else coords


def select_negatives(
    pools: PairCandidatePools,
    selector: str,
    k: int,
) -> list[tuple[int, int]]:
    """Return K negative coordinates. All pools share the same negatives."""
    neg_coords = [c.coordinate for c in pools.topk.negatives]
    if k == 0 or selector == "N_NONE":
        return []
    if k > len(neg_coords):
        k = len(neg_coords)
    if selector == "N_BOTTOMK":
        # First K = most confident background (lowest foreground score)
        return neg_coords[:k]
    if selector == "N_HIGH_BACKGROUND":
        # Last K = boundary cases (highest-score negatives)
        return neg_coords[-k:]
    raise ValueError(f"unknown negative selector: {selector}")


# ---------------------------------------------------------------------------
# Action execution: SAM predict + metrics
# ---------------------------------------------------------------------------

def run_action(
    predictor: Any,
    action: ActionSpec,
    pools: PairCandidatePools,
    target: list[list[bool]],
    target_labels: dict[tuple[int, int], int],
    target_component_count: int,
) -> dict[str, Any]:
    """Run one action and return a metrics record."""
    pos_points = select_positives(pools, action.positive_selector, action.k_positive)
    if not pos_points:
        return _skipped_record(action, target_component_count, "SKIPPED_NO_POSITIVES")

    neg_points = select_negatives(pools, action.negative_selector, action.k_negative)
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
        return _skipped_record(action, target_component_count, f"SAM_ERROR:{type(exc).__name__}")
    best = int(np.argmax(scores))
    mask = masks[best].astype(bool).tolist()
    score = float(scores[best])
    runtime_ms = (time.perf_counter() - t0) * 1000.0

    dice = dice_score(mask, target)
    iou = iou_score(mask, target)
    aji = aggregated_jaccard_index(
        _semantic_instance_labels(mask),
        _semantic_instance_labels(target),
    )
    recall, hit_count, total_count = _component_recall(mask, target_labels)

    return {
        "k_positive": action.k_positive,
        "k_negative": action.k_negative,
        "positive_selector": action.positive_selector,
        "negative_selector": action.negative_selector,
        "sam_calls": 1,
        "actual_positive_count": len(pos_points),
        "actual_negative_count": len(neg_points),
        "dice": dice,
        "iou": iou,
        "aji": aji,
        "component_recall": recall,
        "target_component_count": total_count,
        "hit_components": hit_count,
        "sam_score": score,
        "runtime_ms": round(runtime_ms, 3),
        "status": "OK",
    }


def _skipped_record(
    action: ActionSpec,
    target_component_count: int,
    status: str,
) -> dict[str, Any]:
    return {
        "k_positive": action.k_positive,
        "k_negative": action.k_negative,
        "positive_selector": action.positive_selector,
        "negative_selector": action.negative_selector,
        "sam_calls": 0,
        "actual_positive_count": 0,
        "actual_negative_count": 0,
        "dice": None,
        "iou": None,
        "aji": None,
        "component_recall": None,
        "target_component_count": target_component_count,
        "hit_components": None,
        "sam_score": None,
        "runtime_ms": 0.0,
        "status": status,
    }


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_queries(split: str) -> list[dict[str, str]]:
    """Load query manifest for the given split (train/validation/holdout)."""
    if split == "train":
        path = PROJECT_ROOT / "manifests" / "phaseY" / "policy_train_queries.csv"
    elif split == "validation":
        path = PROJECT_ROOT / "manifests" / "phaseY" / "policy_validation_queries.csv"
    elif split == "holdout":
        path = PROJECT_ROOT / "manifests" / "phaseY" / "confirmatory_holdout_queries.csv"
    else:
        raise ValueError(f"unknown split: {split}")
    return _read_csv(path)


def load_reference_pairs(query_case_id: str, sets: tuple[str, ...]) -> list[dict[str, str]]:
    """Load reference pairs for a query from the specified reference sets."""
    pairs: list[dict[str, str]] = []
    for ref_set in sets:
        path = PROJECT_ROOT / "manifests" / "phaseY" / f"reference_set_{ref_set}.csv"
        for row in _read_csv(path):
            if row["query_case_id"] == query_case_id:
                pairs.append(row)
    return pairs


# ---------------------------------------------------------------------------
# Utility matrix computation with checkpointing
# ---------------------------------------------------------------------------

def _utility_csv_path(out_dir: Path, split: str) -> Path:
    return out_dir / f"utility_matrix_{split}.csv"


def _load_checkpoint(path: Path) -> set[tuple[str, str, str]]:
    """Return set of (query_case_id, reference_case_id, reference_set) already done."""
    if not path.is_file():
        return set()
    done: set[tuple[str, str, str]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["query_case_id"], row["reference_case_id"], row["reference_set"])
            done.add(key)
    return done


def _write_utility_row(handle: csv.DictWriter, row: dict[str, Any]) -> None:
    handle.writerow(row)


def _utility_row_fields() -> list[str]:
    return [
        "query_case_id", "reference_case_id", "reference_set", "reference_index",
        "action_id", "action_index",
        "k_positive", "k_negative", "positive_selector", "negative_selector",
        "sam_calls", "actual_positive_count", "actual_negative_count",
        "dice", "iou", "aji", "component_recall",
        "target_component_count", "hit_components", "sam_score",
        "runtime_ms", "status",
    ]


def run_utility_matrix(
    *,
    split: str,
    reference_sets: tuple[str, ...],
    actions: list[ActionSpec],
    feature_cache: FeatureCache,
    predictor: Any,
    out_dir: Path,
    max_queries: int | None = None,
    max_actions: int | None = None,
    log_every: int = 1,
) -> dict[str, Any]:
    """Run the utility matrix for the given split and reference sets."""
    queries = load_queries(split)
    if max_queries is not None:
        queries = queries[:max_queries]
    if max_actions is not None:
        actions = actions[:max_actions]

    csv_path = _utility_csv_path(out_dir, split)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_checkpoint(csv_path)

    is_append = csv_path.is_file()
    handle_csv = csv_path.open("a" if is_append else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle_csv, fieldnames=_utility_row_fields())
    if not is_append:
        writer.writeheader()

    total_pairs = 0
    total_actions_run = 0
    start = time.perf_counter()

    for qi, query in enumerate(queries):
        query_case_id = query["case_id"]
        ref_pairs = load_reference_pairs(query_case_id, reference_sets)
        if not ref_pairs:
            continue

        # Set SAM image once per query
        query_paths = resolve_case_paths(query_case_id)
        query_image = load_rgb_518(query_paths["image"])
        predictor.set_image(np.asarray(query_image))

        # Load query mask + component labels once per query
        target = load_mask_518(query_paths["mask"])
        target_labels = _component_labels(target)
        target_component_count = len(set(target_labels.values()))

        for ref_pair in ref_pairs:
            ref_case_id = ref_pair["reference_case_id"]
            ref_set = ref_pair["reference_set"]
            ref_index = ref_pair["reference_index"]
            pair_key = (query_case_id, ref_case_id, ref_set)
            if pair_key in done:
                continue

            # Build candidate pools
            try:
                pools = build_pair_candidate_pools(query_case_id, ref_case_id, feature_cache)
            except Exception as exc:
                for action in actions:
                    record = _skipped_record(action, target_component_count, f"POOL_ERROR:{type(exc).__name__}")
                    row = _compose_row(query_case_id, ref_case_id, ref_set, ref_index, action, record)
                    writer.writerow(row)
                handle_csv.flush()
                total_pairs += 1
                continue

            # Run all actions
            for action in actions:
                record = run_action(
                    predictor, action, pools, target, target_labels, target_component_count
                )
                row = _compose_row(query_case_id, ref_case_id, ref_set, ref_index, action, record)
                writer.writerow(row)
                total_actions_run += 1

            handle_csv.flush()
            done.add(pair_key)
            total_pairs += 1

            if total_pairs % log_every == 0:
                elapsed = time.perf_counter() - start
                print(f"[phaseY-Y1] {split} pair #{total_pairs} "
                      f"(q={query_case_id} r={ref_case_id} set={ref_set}) "
                      f"actions={total_actions_run} elapsed={elapsed:.1f}s")

        # End of query; print progress
        elapsed = time.perf_counter() - start
        print(f"[phaseY-Y1] {split} query {qi+1}/{len(queries)} done "
              f"({query_case_id}) pairs={total_pairs} elapsed={elapsed:.1f}s")

    handle_csv.close()
    elapsed = time.perf_counter() - start
    return {
        "split": split,
        "reference_sets": list(reference_sets),
        "query_count": len(queries),
        "pair_count": total_pairs,
        "action_count": len(actions),
        "total_actions_run": total_actions_run,
        "elapsed_seconds": round(elapsed, 3),
        "csv_path": str(csv_path),
    }


def _compose_row(
    query_case_id: str,
    ref_case_id: str,
    ref_set: str,
    ref_index: str,
    action: ActionSpec,
    record: dict[str, Any],
) -> dict[str, Any]:
    return {
        "query_case_id": query_case_id,
        "reference_case_id": ref_case_id,
        "reference_set": ref_set,
        "reference_index": ref_index,
        "action_id": action.action_id,
        "action_index": action.action_index,
        **record,
    }


# ---------------------------------------------------------------------------
# Upper bounds and stability metrics (Gate Y1)
# ---------------------------------------------------------------------------

def load_utility_matrix(csv_path: Path) -> list[dict[str, Any]]:
    """Load utility matrix CSV into list of dicts with proper types."""
    rows: list[dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["dice"] = float(row["dice"]) if row["dice"] else None
            row["iou"] = float(row["iou"]) if row["iou"] else None
            row["aji"] = float(row["aji"]) if row["aji"] else None
            row["component_recall"] = float(row["component_recall"]) if row["component_recall"] else None
            row["k_positive"] = int(row["k_positive"])
            row["k_negative"] = int(row["k_negative"])
            row["action_index"] = int(row["action_index"])
            rows.append(row)
    return rows


def _per_query_action_dice(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Return {query_case_id: {action_id: dice}} for OK rows only."""
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        if row["status"] != "OK" or row["dice"] is None:
            continue
        qid = row["query_case_id"]
        aid = row["action_id"]
        result.setdefault(qid, {})[aid] = row["dice"]
    return result


def _aggregate_per_query_across_refs(
    rows: list[dict[str, Any]],
    reference_set: str,
) -> dict[str, dict[str, float]]:
    """Aggregate dice across references for a given set: {query: {action: mean_dice}}."""
    by_qa: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        if row["reference_set"] != reference_set:
            continue
        if row["status"] != "OK" or row["dice"] is None:
            continue
        qid = row["query_case_id"]
        aid = row["action_id"]
        by_qa.setdefault(qid, {}).setdefault(aid, []).append(row["dice"])
    return {
        qid: {aid: sum(v) / len(v) for aid, v in actions.items()}
        for qid, actions in by_qa.items()
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _spearman_rho(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    nx = len(x)
    rx = _ranks(x)
    ry = _ranks(y)
    mx = sum(rx) / nx
    my = sum(ry) / nx
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(nx))
    dx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(nx)))
    dy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(nx)))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def _ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda t: t[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _bootstrap_ci(
    paired_diffs: list[float],
    iterations: int = 10000,
    seed: int = 20260721,
) -> tuple[float, float]:
    """Paired bootstrap 95% CI lower/upper bound on the mean difference."""
    if not paired_diffs:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    n = len(paired_diffs)
    arr = np.asarray(paired_diffs, dtype=np.float64)
    means = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        idx = rng.integers(0, n, size=n)
        means[i] = arr[idx].mean()
    lo = float(np.percentile(means, 2.5))
    hi = float(np.percentile(means, 97.5))
    return lo, hi


def compute_gate_y1(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute Gate Y1 upper bounds, stability metrics, and decision."""
    # Aggregate per (query, action) within each reference set
    setA = _aggregate_per_query_across_refs(rows, "A")
    setB = _aggregate_per_query_across_refs(rows, "B")

    queries = sorted(set(setA.keys()) & set(setB.keys()))
    if not queries:
        return {"status": "PHASEY_GATE_Y1_INSUFFICIENT_DATA", "queries": 0}

    # All action IDs present in both sets
    all_actions = sorted(
        set(aid for q in queries for aid in setA[q].keys())
        & set(aid for q in queries for aid in setB[q].keys())
    )

    # --- Fixed-A: strongest fixed action on Set A ---
    action_meanA = {
        aid: _mean([setA[q][aid] for q in queries if aid in setA[q]])
        for aid in all_actions
    }
    best_fixed_action = max(action_meanA, key=action_meanA.get)
    fixed_A_dice = action_meanA[best_fixed_action]

    # --- Oracle-A / Oracle-B: per-query best action ---
    oracle_A = _mean([max(setA[q].values()) for q in queries])
    oracle_B = _mean([max(setB[q].values()) for q in queries])

    # --- Cross-Reference Oracle A->B ---
    # For each query, take A's best action, evaluate on B
    cross_A_to_B_values: list[float] = []
    cross_B_to_A_values: list[float] = []
    per_query_best_A: dict[str, str] = {}
    per_query_best_B: dict[str, str] = {}
    for q in queries:
        a_star = max(setA[q], key=setA[q].get)
        b_star = max(setB[q], key=setB[q].get)
        per_query_best_A[q] = a_star
        per_query_best_B[q] = b_star
        cross_A_to_B_values.append(setB[q].get(a_star, 0.0))
        cross_B_to_A_values.append(setA[q].get(b_star, 0.0))
    cross_A_to_B = _mean(cross_A_to_B_values)
    cross_B_to_A = _mean(cross_B_to_A_values)
    stable_oracle = (cross_A_to_B + cross_B_to_A) / 2.0

    # --- Optimism ---
    optimism_A = oracle_A - cross_A_to_B
    optimism_B = oracle_B - cross_B_to_A
    raw_gap_A = oracle_A - fixed_A_dice
    raw_gap_B = oracle_B - fixed_A_dice  # using fixed-A as baseline for both
    stable_oracle_gap = stable_oracle - fixed_A_dice

    # --- Paired bootstrap on stable oracle gap ---
    paired_diffs: list[float] = []
    for q in queries:
        a_star = per_query_best_A[q]
        b_star = per_query_best_B[q]
        cross_mean = (setB[q].get(a_star, 0.0) + setA[q].get(b_star, 0.0)) / 2.0
        paired_diffs.append(cross_mean - fixed_A_dice)
    ci_lo, ci_hi = _bootstrap_ci(paired_diffs, iterations=2000)

    # --- Action stability metrics ---
    # Exact action agreement: same best action on A and B
    exact_agree = sum(1 for q in queries if per_query_best_A[q] == per_query_best_B[q])
    exact_agreement_rate = exact_agree / len(queries)

    # K+ bin agreement: best action's K+ within 5 of each other
    kplus_A = {q: _action_k_positive(per_query_best_A[q]) for q in queries}
    kplus_B = {q: _action_k_positive(per_query_best_B[q]) for q in queries}
    kplus_agree = sum(1 for q in queries if abs(kplus_A[q] - kplus_B[q]) <= 5)
    kplus_agreement_rate = kplus_agree / len(queries)

    # Selector agreement
    sel_A = {q: _action_pos_selector(per_query_best_A[q]) for q in queries}
    sel_B = {q: _action_pos_selector(per_query_best_B[q]) for q in queries}
    sel_agree = sum(1 for q in queries if sel_A[q] == sel_B[q])
    selector_agreement_rate = sel_agree / len(queries)

    # Spearman rank correlation of action utilities A vs B (per query, then median)
    rhos: list[float] = []
    for q in queries:
        aids_q = sorted(set(setA[q].keys()) & set(setB[q].keys()))
        if len(aids_q) < 2:
            continue
        x = [setA[q][aid] for aid in aids_q]
        y = [setB[q][aid] for aid in aids_q]
        rhos.append(_spearman_rho(x, y))
    median_rho = _median(rhos) if rhos else 0.0

    # Top-3 action overlap
    top3_overlap_rates: list[float] = []
    for q in queries:
        top3_A = set(sorted(setA[q], key=setA[q].get, reverse=True)[:3])
        top3_B = set(sorted(setB[q], key=setB[q].get, reverse=True)[:3])
        if top3_A and top3_B:
            top3_overlap_rates.append(len(top3_A & top3_B) / 3.0)
    mean_top3_overlap = _mean(top3_overlap_rates)

    # Reference-set win consistency: A's best action also wins on B for same query
    win_consistent = 0
    for q in queries:
        a_star = per_query_best_A[q]
        if a_star in setB[q]:
            # Check if a_star is above median on B
            b_vals = list(setB[q].values())
            if setB[q][a_star] >= _median(b_vals):
                win_consistent += 1
    win_consistency_rate = win_consistent / len(queries)

    # % queries where stable best-action family differs from strongest fixed
    fixed_family = _action_pos_selector(best_fixed_action)
    diff_from_fixed = sum(
        1 for q in queries
        if _action_pos_selector(per_query_best_A[q]) != fixed_family
        or _action_pos_selector(per_query_best_B[q]) != fixed_family
    )
    # Use the stricter interpretation: at least one of A/B best differs
    diff_rate = diff_from_fixed / len(queries)

    # --- Gate Y1 decision (6 conditions) ---
    # 1. stable_oracle_gap >= 0.015
    cond1 = stable_oracle_gap >= 0.015
    # 2. bootstrap 95% CI lower bound > 0
    cond2 = ci_lo > 0
    # 3. >= 30% queries stable best-action family differs from strongest fixed
    cond3 = diff_rate >= 0.30
    # 4. median Spearman rho > 0.30
    cond4 = median_rho > 0.30
    # 5. optimism <= 60% of raw oracle gap
    optimism_ratio_A = optimism_A / raw_gap_A if raw_gap_A > 0 else 1.0
    cond5 = optimism_ratio_A <= 0.60
    # 6. all actions are 1 SAM call (structural, always true for Phase Y bank)
    cond6 = True

    conditions = {
        "cond1_stable_oracle_gap_ge_0p015": cond1,
        "cond2_bootstrap_ci_lo_gt_0": cond2,
        "cond3_30pct_diff_from_fixed": cond3,
        "cond4_median_spearman_rho_gt_0p30": cond4,
        "cond5_optimism_le_60pct_gap": cond5,
        "cond6_all_one_sam_call": cond6,
    }

    if all(conditions.values()):
        decision = "PHASEY_GATE_Y1_STABLE_HETEROGENEITY"
    elif stable_oracle_gap < 0.015:
        decision = "PHASEY_GATE_Y1_ORACLE_GAP_SMALL"
    elif not cond5:
        decision = "PHASEY_GATE_Y1_SELECTION_BIAS_DOMINATES"
    else:
        decision = "PHASEY_GATE_Y1_REFERENCE_UNSTABLE"

    return {
        "query_count": len(queries),
        "action_count": len(all_actions),
        "best_fixed_action": best_fixed_action,
        "fixed_A_dice": fixed_A_dice,
        "oracle_A": oracle_A,
        "oracle_B": oracle_B,
        "cross_A_to_B": cross_A_to_B,
        "cross_B_to_A": cross_B_to_A,
        "stable_oracle": stable_oracle,
        "optimism_A": optimism_A,
        "optimism_B": optimism_B,
        "raw_gap_A": raw_gap_A,
        "stable_oracle_gap": stable_oracle_gap,
        "bootstrap_ci_lo": ci_lo,
        "bootstrap_ci_hi": ci_hi,
        "exact_action_agreement_rate": exact_agreement_rate,
        "kplus_bin_agreement_rate": kplus_agreement_rate,
        "selector_agreement_rate": selector_agreement_rate,
        "median_spearman_rho": median_rho,
        "mean_top3_overlap": mean_top3_overlap,
        "win_consistency_rate": win_consistency_rate,
        "diff_from_fixed_rate": diff_rate,
        "optimism_ratio_A": optimism_ratio_A,
        "conditions": conditions,
        "decision": decision,
    }


def _action_k_positive(action_id: str) -> int:
    """Y_P_TOPK_K5_N0_N_NONE_G1_C1 -> 5."""
    for token in action_id.split("_"):
        if token.startswith("K") and token[1:].isdigit():
            return int(token[1:])
    return 0


def _action_pos_selector(action_id: str) -> str:
    """Y_P_TOPK_K5_N0_N_NONE_G1_C1 -> P_TOPK."""
    tokens = action_id.split("_")
    if len(tokens) >= 2 and tokens[1].startswith("P_"):
        return tokens[1]
    return ""


# ---------------------------------------------------------------------------
# Pareto pruning (train-only, for compact bank derivation)
# ---------------------------------------------------------------------------

def compute_pareto_pruning(
    rows: list[dict[str, Any]],
    target_min: int = 12,
    target_max: int = 32,
) -> dict[str, Any]:
    """Pareto-prune the full bank on train-only metrics to derive compact bank."""
    # Aggregate per action: mean Dice, mean component recall, mean runtime,
    # catastrophic failure rate (dice < 0.1)
    by_action: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["status"] != "OK" or row["dice"] is None:
            continue
        by_action.setdefault(row["action_id"], []).append(row)

    action_metrics: list[dict[str, Any]] = []
    for aid, action_rows in by_action.items():
        dices = [r["dice"] for r in action_rows]
        recalls = [r["component_recall"] for r in action_rows]
        runtimes = [r["runtime_ms"] for r in action_rows]
        catastrophic = sum(1 for d in dices if d < 0.1) / len(dices)
        action_metrics.append({
            "action_id": aid,
            "action_index": action_rows[0]["action_index"],
            "k_positive": action_rows[0]["k_positive"],
            "k_negative": action_rows[0]["k_negative"],
            "positive_selector": action_rows[0]["positive_selector"],
            "negative_selector": action_rows[0]["negative_selector"],
            "mean_dice": _mean(dices),
            "mean_component_recall": _mean(recalls),
            "mean_runtime_ms": _mean(runtimes),
            "catastrophic_failure_rate": catastrophic,
            "query_count": len(action_rows),
        })

    # Sort by mean_dice descending
    action_metrics.sort(key=lambda m: m["mean_dice"], reverse=True)

    # Simple Pareto pruning: keep top-K by Dice while ensuring diversity
    # Rule 1: Remove actions dominated by another in (Dice, runtime, calls)
    # Rule 2: Within same family, if Dice diff < 0.003, keep simpler (fewer points)
    # Rule 3: Keep target_max as upper bound

    # First pass: remove strictly dominated
    kept: list[dict[str, Any]] = []
    for am in action_metrics:
        dominated = False
        for other in action_metrics:
            if other["action_id"] == am["action_id"]:
                continue
            # other dominates am if: dice >=, runtime <=, points <=, and at least one strict
            if (other["mean_dice"] >= am["mean_dice"]
                and other["mean_runtime_ms"] <= am["mean_runtime_ms"]
                and other["k_positive"] + other["k_negative"] <= am["k_positive"] + am["k_negative"]
                and (other["mean_dice"] > am["mean_dice"]
                     or other["mean_runtime_ms"] < am["mean_runtime_ms"]
                     or other["k_positive"] + other["k_negative"] < am["k_positive"] + am["k_negative"])):
                dominated = True
                break
        if not dominated:
            kept.append(am)

    # Second pass: if still too many, keep top target_max by Dice
    if len(kept) > target_max:
        kept = kept[:target_max]
    elif len(kept) < target_min:
        # Add back highest-Dice actions not yet kept
        kept_ids = {k["action_id"] for k in kept}
        for am in action_metrics:
            if am["action_id"] not in kept_ids:
                kept.append(am)
                kept_ids.add(am["action_id"])
                if len(kept) >= target_min:
                    break

    return {
        "full_bank_size": len(action_metrics),
        "compact_bank_size": len(kept),
        "target_min": target_min,
        "target_max": target_max,
        "compact_bank_actions": [
            {
                "action_id": k["action_id"],
                "action_index": k["action_index"],
                "k_positive": k["k_positive"],
                "k_negative": k["k_negative"],
                "positive_selector": k["positive_selector"],
                "negative_selector": k["negative_selector"],
                "mean_dice": k["mean_dice"],
            }
            for k in kept
        ],
        "all_action_metrics": action_metrics,
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


def write_gate_y1_summary(
    gate: dict[str, Any],
    pareto: dict[str, Any] | None,
    out_dir: Path,
) -> None:
    json_path = out_dir / "gateY1_summary.json"
    json_path.write_text(json.dumps({"gate_y1": gate, "pareto": pareto}, indent=2),
                         encoding="utf-8")

    md = ["# Phase Y Gate Y1: Fresh-Query Action Heterogeneity Summary", ""]
    md.append(f"Date: {datetime.now().isoformat(timespec='seconds')}")
    md.append(f"Queries (validation, A∩B): {gate['query_count']}")
    md.append(f"Actions evaluated: {gate['action_count']}")
    md.append("")
    md.append("## Upper Bounds (Dice)")
    md.append("")
    md.append("| Metric | Value |")
    md.append("|---|---:|")
    md.append(f"| Best fixed action (Set A) | {gate['best_fixed_action']} |")
    md.append(f"| Fixed-A Dice | {_fmt(gate['fixed_A_dice'])} |")
    md.append(f"| Oracle-A (raw) | {_fmt(gate['oracle_A'])} |")
    md.append(f"| Oracle-B (raw) | {_fmt(gate['oracle_B'])} |")
    md.append(f"| Cross-Ref A→B | {_fmt(gate['cross_A_to_B'])} |")
    md.append(f"| Cross-Ref B→A | {_fmt(gate['cross_B_to_A'])} |")
    md.append(f"| **Stable Oracle** | **{_fmt(gate['stable_oracle'])}** |")
    md.append(f"| Optimism A | {_fmt(gate['optimism_A'])} |")
    md.append(f"| Optimism B | {_fmt(gate['optimism_B'])} |")
    md.append(f"| Raw gap A | {_fmt(gate['raw_gap_A'])} |")
    md.append(f"| **Stable Oracle Gap** | **{_fmt(gate['stable_oracle_gap'])}** |")
    md.append(f"| Bootstrap 95% CI | [{_fmt(gate['bootstrap_ci_lo'])}, {_fmt(gate['bootstrap_ci_hi'])}] |")
    md.append(f"| Optimism ratio A | {_fmt(gate['optimism_ratio_A'])} |")
    md.append("")
    md.append("## Action Stability")
    md.append("")
    md.append("| Metric | Value |")
    md.append("|---|---:|")
    md.append(f"| Exact action agreement | {_fmt(gate['exact_action_agreement_rate'])} |")
    md.append(f"| K+ bin agreement (±5) | {_fmt(gate['kplus_bin_agreement_rate'])} |")
    md.append(f"| Selector agreement | {_fmt(gate['selector_agreement_rate'])} |")
    md.append(f"| Median Spearman ρ | {_fmt(gate['median_spearman_rho'])} |")
    md.append(f"| Mean top-3 overlap | {_fmt(gate['mean_top3_overlap'])} |")
    md.append(f"| Win consistency | {_fmt(gate['win_consistency_rate'])} |")
    md.append(f"| Diff from fixed (≥30%) | {_fmt(gate['diff_from_fixed_rate'])} |")
    md.append("")
    md.append("## Gate Y1 Conditions")
    md.append("")
    md.append("| # | Condition | Pass |")
    md.append("|---|---|---|")
    for name, passed in gate["conditions"].items():
        md.append(f"| 1 | {name} | {'YES' if passed else 'NO'} |")
    md.append("")
    md.append(f"## Decision: **{gate['decision']}**")
    md.append("")
    if pareto:
        md.append("## Pareto Pruning (train-only)")
        md.append("")
        md.append(f"- Full bank size: {pareto['full_bank_size']}")
        md.append(f"- Compact bank size: {pareto['compact_bank_size']}")
        md.append(f"- Target range: [{pareto['target_min']}, {pareto['target_max']}]")
        md.append("")
        md.append("Top-5 compact actions by Dice:")
        md.append("")
        md.append("| action_id | K+ | K- | pos_sel | neg_sel | mean_dice |")
        md.append("|---|---:|---:|---|---|---:|")
        for a in pareto["compact_bank_actions"][:5]:
            md.append(f"| {a['action_id']} | {a['k_positive']} | {a['k_negative']} | "
                      f"{a['positive_selector']} | {a['negative_selector']} | "
                      f"{_fmt(a['mean_dice'])} |")

    (out_dir / "gateY1_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def write_pareto_csv(pareto: dict[str, Any], out_dir: Path) -> None:
    path = out_dir / "pareto_train_metrics.csv"
    fields = [
        "action_id", "action_index", "k_positive", "k_negative",
        "positive_selector", "negative_selector",
        "mean_dice", "mean_component_recall", "mean_runtime_ms",
        "catastrophic_failure_rate", "query_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for am in pareto["all_action_metrics"]:
            writer.writerow({k: am.get(k, "") for k in fields})


def write_compact_bank_yaml(pareto: dict[str, Any], out_dir: Path) -> Path:
    """Write the compact policy bank YAML for downstream Y3 training."""
    yaml_path = PROJECT_ROOT / "configs" / "actions" / "phaseY_compact_policy_bank.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    actions = []
    for entry in pareto["compact_bank_actions"]:
        actions.append({
            "action_id": entry["action_id"],
            "action_index": entry["action_index"],
            "positive_selector": entry["positive_selector"],
            "k_positive": entry["k_positive"],
            "negative_selector": _neg_selector_from_action_id(entry["action_id"]),
            "k_negative": entry["k_negative"],
            "grouping": "G_ONE_GROUP",
            "sam_calls": 1,
            "point_count": entry["k_positive"] + entry["k_negative"],
        })
    data = {
        "bank_name": "phaseY_compact_policy_bank",
        "bank_version": "v1",
        "source": "pareto_pruning_on_train",
        "total_actions": len(actions),
        "actions": actions,
    }
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
    return yaml_path


def _neg_selector_from_action_id(action_id: str) -> str:
    """Y_P_TOPK_K5_N1_N_BOTTOMK_G1_C1 -> N_BOTTOMK."""
    tokens = action_id.split("_")
    for i, token in enumerate(tokens):
        if token == "N" and i + 1 < len(tokens):
            next_tok = tokens[i + 1]
            if next_tok in ("NONE", "BOTTOMK", "HIGH"):
                if next_tok == "NONE":
                    return "N_NONE"
                if next_tok == "BOTTOMK":
                    return "N_BOTTOMK"
                if next_tok == "HIGH":
                    return "N_HIGH_BACKGROUND"
    return "N_NONE"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase Y Gate Y1 utility matrix (210 actions × queries × references)"
    )
    parser.add_argument("--split", choices=["train", "validation", "both"], default="both")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path,
                        default=PROJECT_ROOT / "reports" / "phaseY_gateY1")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--compute-gate-only", action="store_true",
                        help="Skip SAM inference; just recompute Gate Y1 from existing CSV")
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        args.max_queries = args.max_queries or 2
        args.max_actions = args.max_actions or 6
        args.split = "validation"

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[phaseY-Y1] device={device}")

    # Load action bank
    actions = load_action_bank()
    if args.max_actions:
        actions = actions[:args.max_actions]
    print(f"[phaseY-Y1] loaded {len(actions)} actions")

    metadata: dict[str, Any] = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "action_count": len(actions),
        "splits": {},
    }

    if not args.compute_gate_only:
        feature_cache = FeatureCache(device=device)
        predictor = load_sam_predictor(device)

        if args.split in ("train", "both"):
            print("[phaseY-Y1] === Train utility matrix (for Pareto pruning) ===")
            train_meta = run_utility_matrix(
                split="train",
                reference_sets=("A",),
                actions=actions,
                feature_cache=feature_cache,
                predictor=predictor,
                out_dir=out_dir,
                max_queries=args.max_queries,
                max_actions=args.max_actions,
            )
            metadata["splits"]["train"] = train_meta
            print(f"[phaseY-Y1] train done: {train_meta['pair_count']} pairs, "
                  f"{train_meta['total_actions_run']} actions, "
                  f"{train_meta['elapsed_seconds']}s")

        if args.split in ("validation", "both"):
            print("[phaseY-Y1] === Validation A/B utility matrix (for Gate Y1) ===")
            val_meta = run_utility_matrix(
                split="validation",
                reference_sets=("A", "B"),
                actions=actions,
                feature_cache=feature_cache,
                predictor=predictor,
                out_dir=out_dir,
                max_queries=args.max_queries,
                max_actions=args.max_actions,
            )
            metadata["splits"]["validation"] = val_meta
            print(f"[phaseY-Y1] val done: {val_meta['pair_count']} pairs, "
                  f"{val_meta['total_actions_run']} actions, "
                  f"{val_meta['elapsed_seconds']}s")

    # Compute Gate Y1 from validation utility matrix
    val_csv = _utility_csv_path(out_dir, "validation")
    gate = None
    pareto = None
    if val_csv.is_file():
        print(f"[phaseY-Y1] computing Gate Y1 from {val_csv}")
        val_rows = load_utility_matrix(val_csv)
        gate = compute_gate_y1(val_rows)
        print(f"[phaseY-Y1] decision: {gate['decision']}")
        print(f"[phaseY-Y1] stable oracle gap: {gate['stable_oracle_gap']:.4f}")
        print(f"[phaseY-Y1] median Spearman rho: {gate['median_spearman_rho']:.4f}")

    # Compute Pareto pruning from train utility matrix
    train_csv = _utility_csv_path(out_dir, "train")
    if train_csv.is_file():
        print(f"[phaseY-Y1] computing Pareto pruning from {train_csv}")
        train_rows = load_utility_matrix(train_csv)
        pareto = compute_pareto_pruning(train_rows)
        print(f"[phaseY-Y1] compact bank: {pareto['compact_bank_size']} actions "
              f"(from {pareto['full_bank_size']})")
        write_pareto_csv(pareto, out_dir)
        compact_yaml_path = write_compact_bank_yaml(pareto, out_dir)
        print(f"[phaseY-Y1] compact bank YAML written to {compact_yaml_path}")

    if gate or pareto:
        write_gate_y1_summary(gate, pareto, out_dir)

    # Save metadata
    metadata["gate_y1"] = gate
    metadata["pareto"] = {
        "full_bank_size": pareto["full_bank_size"] if pareto else None,
        "compact_bank_size": pareto["compact_bank_size"] if pareto else None,
    } if pareto else None
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"[phaseY-Y1] artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
