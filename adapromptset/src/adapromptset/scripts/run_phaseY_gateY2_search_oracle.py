"""Phase Y Gate Y2: Search Oracle Stability (section 9).

Implements section 9 of ``AdaPromptSet_PhaseY_Policy_Feasibility.md``.

GT-guided search is run on policy train + validation queries using dual
reference sets A and B. The script measures whether the search result
(prompt subset) is stable across references, not just whether search Dice
is high.

Search protocol (section 9.2):
  For each validation query:
    1. Run GT-guided search with Reference Set A -> S_A (subset indices)
    2. Run GT-guided search with Reference Set B -> S_B (subset indices)
    3. Cross-evaluate:
       - S_A's indices applied to B's candidate pool -> cross Dice
       - S_B's indices applied to A's candidate pool -> cross Dice

Reported metrics (section 9.2):
  - raw search Dice (per-reference-set, then averaged)
  - cross-reference search Dice (A->B, B->A, then averaged)
  - K+ stability (median |K_A - K_B|)
  - K- stability
  - point-coordinate overlap (Jaccard of physical points)
  - spatial coverage overlap
  - candidate-rank overlap (Jaccard of pool indices)
  - search optimism = raw - cross
  - final prompt area bias

Gate Y2 decision (section 9.4):
  1. cross-reference search >= 0.03 Dice over strongest fixed
  2. paired bootstrap CI lower bound > 0
  3. search optimism <= 50% of raw search gap
  4. median |K_A - K_B| <= 8 OR cross Dice diff <= 0.01
  5. >= 70% queries: A-search prompt beats strongest fixed on B

Outputs::

    reports/phaseY_gateY2/
        search_oracle_per_query.csv     (per-query raw/cross Dice, K, overlap)
        search_oracle_per_pair.csv      (per (query, ref) search details)
        gateY2_summary.json             (aggregate metrics + decision)
        gateY2_summary.md               (human-readable summary)
        run_metadata.json               (device, runtime, gpu_hours)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Reuse Y1 infrastructure
from adapromptset.evaluation.metrics import (
    aggregated_jaccard_index,
    dice_score,
    iou_score,
)
from adapromptset.scripts.run_phaseY_gateY1_utility_matrix import (
    ACTION_BANK_YAML,
    DINOV2_CHECKPOINT,
    FeatureCache,
    MONUSAC_ROOT,
    PROJECT_ROOT,
    SAM_CHECKPOINT,
    WORKSPACE_ROOT,
    PairCandidatePools,
    _bootstrap_ci,
    _component_labels,
    _component_recall,
    _mean,
    _median,
    _semantic_instance_labels,
    _spearman_rho,
    build_pair_candidate_pools,
    load_action_bank,
    load_queries,
    load_reference_pairs,
    load_rgb_518,
    load_mask_518,
    load_sam_predictor,
    resolve_case_paths,
    select_negatives,
    select_positives,
)

OUT_DIR_DEFAULT = PROJECT_ROOT / "reports" / "phaseY_gateY2"

# Search configuration (frozen at preregistration, mirrors Phase X).
DEFAULT_K_POSITIVE_START = 20
DEFAULT_K_NEGATIVE_START = 1
DEFAULT_NEGATIVE_SELECTOR = "N_BOTTOMK"
DEFAULT_B_POINT = 100        # max total positives allowed in search
DEFAULT_B_NEGATIVE = 5       # max total negatives allowed in search
MIN_IMPROVEMENT = 0.001      # min Dice delta to accept a greedy move
GREEDY_ADD_CANDIDATE_LIMIT = 30  # only try top-N unselected per iteration
SKIP_GREEDY_SWAP = True      # O(K*(100-K)) too expensive


# ---------------------------------------------------------------------------
# Search state and primitives
# ---------------------------------------------------------------------------

@dataclass
class SearchState:
    """Mutable search state: selected positive/negative indices into pool."""
    positive_indices: list[int] = field(default_factory=list)
    negative_indices: list[int] = field(default_factory=list)
    dice: float = 0.0
    sam_calls: int = 0
    # Record the pool's positive/negative coordinate lists for convenience
    positive_coords: list[tuple[int, int]] = field(default_factory=list)
    negative_coords: list[tuple[int, int]] = field(default_factory=list)


def _predict_one(
    predictor: Any,
    positive_points: list[tuple[int, int]],
    negative_points: list[tuple[int, int]],
) -> tuple[list[list[bool]], float, int]:
    """Single SAM predict call. Returns (mask, score, sam_calls)."""
    if not positive_points:
        return [[False] * 518 for _ in range(518)], 0.0, 0
    points = positive_points + negative_points
    labels = [1] * len(positive_points) + [0] * len(negative_points)
    masks, scores, _ = predictor.predict(
        point_coords=np.asarray(points, dtype=np.float32),
        point_labels=np.asarray(labels, dtype=np.int32),
        multimask_output=True,
    )
    best = int(np.argmax(scores))
    return masks[best].astype(bool).tolist(), float(scores[best]), 1


def _evaluate(
    predictor: Any,
    positive_coords: list[tuple[int, int]],
    negative_coords: list[tuple[int, int]],
    target: list[list[bool]],
    target_labels: dict[tuple[int, int], int],
    target_component_count: int,
) -> tuple[float, dict[str, Any], int]:
    """Run SAM predict + metrics. Returns (dice, metrics, sam_calls)."""
    if not positive_coords:
        return 0.0, {
            "dice": 0.0, "iou": 0.0, "aji": 0.0,
            "component_recall": 0.0, "retained_masks": 0,
        }, 0
    mask, score, calls = _predict_one(predictor, positive_coords, negative_coords)
    dice = dice_score(mask, target)
    iou = iou_score(mask, target)
    aji = aggregated_jaccard_index(
        _semantic_instance_labels(mask),
        _semantic_instance_labels(target),
    )
    recall, hit_count, total_count = _component_recall(mask, target_labels)
    return dice, {
        "dice": dice, "iou": iou, "aji": aji,
        "component_recall": recall, "retained_masks": 1,
        "sam_score": score,
    }, calls


# ---------------------------------------------------------------------------
# Greedy search operators
# ---------------------------------------------------------------------------

def _greedy_add_positives(
    predictor: Any,
    state: SearchState,
    target: list[list[bool]],
    target_labels: dict[tuple[int, int], int],
    target_component_count: int,
    max_positives: int,
) -> SearchState:
    """Greedy add: try each unselected positive (top-N), keep best if improves."""
    improved = True
    while improved and len(state.positive_indices) < max_positives:
        improved = False
        best_idx = -1
        best_dice = state.dice
        selected_set = set(state.positive_indices)
        unselected = [
            i for i in range(len(state.positive_coords))
            if i not in selected_set
        ]
        unselected = unselected[:GREEDY_ADD_CANDIDATE_LIMIT]
        for idx in unselected:
            trial_pos = [state.positive_coords[i] for i in state.positive_indices] + [
                state.positive_coords[idx]
            ]
            trial_neg = [state.negative_coords[i] for i in state.negative_indices]
            trial_dice, _, calls = _evaluate(
                predictor, trial_pos, trial_neg, target, target_labels,
                target_component_count,
            )
            state.sam_calls += calls
            if trial_dice > best_dice + MIN_IMPROVEMENT:
                best_dice = trial_dice
                best_idx = idx
        if best_idx >= 0:
            state.positive_indices.append(best_idx)
            state.dice = best_dice
            improved = True
    return state


def _greedy_remove_positives(
    predictor: Any,
    state: SearchState,
    target: list[list[bool]],
    target_labels: dict[tuple[int, int], int],
    target_component_count: int,
    min_positives: int = 1,
) -> SearchState:
    """Greedy remove: try removing each selected positive, keep if improves."""
    if len(state.positive_indices) <= min_positives:
        return state
    improved = True
    while improved and len(state.positive_indices) > min_positives:
        improved = False
        best_remove = -1
        best_dice = state.dice
        for i, _ in enumerate(state.positive_indices):
            trial_pos_indices = state.positive_indices[:i] + state.positive_indices[i + 1:]
            trial_pos = [state.positive_coords[j] for j in trial_pos_indices]
            trial_neg = [state.negative_coords[j] for j in state.negative_indices]
            trial_dice, _, calls = _evaluate(
                predictor, trial_pos, trial_neg, target, target_labels,
                target_component_count,
            )
            state.sam_calls += calls
            if trial_dice > best_dice + MIN_IMPROVEMENT:
                best_dice = trial_dice
                best_remove = i
        if best_remove >= 0:
            state.positive_indices.pop(best_remove)
            state.dice = best_dice
            improved = True
    return state


def _greedy_add_negatives(
    predictor: Any,
    state: SearchState,
    target: list[list[bool]],
    target_labels: dict[tuple[int, int], int],
    target_component_count: int,
    max_negatives: int,
) -> SearchState:
    """Greedy add negatives: try each unselected negative (top-N), keep best."""
    improved = True
    while improved and len(state.negative_indices) < max_negatives:
        improved = False
        best_idx = -1
        best_dice = state.dice
        selected_set = set(state.negative_indices)
        unselected = [
            i for i in range(len(state.negative_coords))
            if i not in selected_set
        ]
        unselected = unselected[:GREEDY_ADD_CANDIDATE_LIMIT]
        for idx in unselected:
            trial_pos = [state.positive_coords[i] for i in state.positive_indices]
            trial_neg = [state.negative_coords[i] for i in state.negative_indices] + [
                state.negative_coords[idx]
            ]
            trial_dice, _, calls = _evaluate(
                predictor, trial_pos, trial_neg, target, target_labels,
                target_component_count,
            )
            state.sam_calls += calls
            if trial_dice > best_dice + MIN_IMPROVEMENT:
                best_dice = trial_dice
                best_idx = idx
        if best_idx >= 0:
            state.negative_indices.append(best_idx)
            state.dice = best_dice
            improved = True
    return state


# ---------------------------------------------------------------------------
# Run search oracle for one (query, reference) pair
# ---------------------------------------------------------------------------

def run_search_for_pair(
    predictor: Any,
    pools: PairCandidatePools,
    target: list[list[bool]],
    target_labels: dict[tuple[int, int], int],
    target_component_count: int,
    *,
    k_positive_start: int = DEFAULT_K_POSITIVE_START,
    k_negative_start: int = DEFAULT_K_NEGATIVE_START,
    negative_selector: str = DEFAULT_NEGATIVE_SELECTOR,
    b_point: int = DEFAULT_B_POINT,
    b_negative: int = DEFAULT_B_NEGATIVE,
    positive_pool_name: str = "topk",
) -> dict[str, Any]:
    """Run GT-guided search for one (query, ref) pair.

    Returns dict with:
      - positive_indices, negative_indices (into the chosen pool)
      - positive_coords, negative_coords (physical points)
      - search_dice, sam_calls
      - k_positive, k_negative (final counts)
    """
    # Pick the positive pool by selector name
    if positive_pool_name == "topk":
        pool = pools.topk
    elif positive_pool_name == "spread":
        pool = pools.spread
    elif positive_pool_name == "local_peaks":
        pool = pools.local_peaks
    else:
        raise ValueError(f"unknown positive_pool_name: {positive_pool_name}")

    positive_coords_all = [c.coordinate for c in pool.positives]
    # Negatives are shared across pools
    negative_coords_all = [c.coordinate for c in pools.topk.negatives]

    if not positive_coords_all:
        return {
            "positive_indices": [],
            "negative_indices": [],
            "positive_coords": [],
            "negative_coords": [],
            "search_dice": 0.0,
            "sam_calls": 0,
            "k_positive": 0,
            "k_negative": 0,
            "status": "NO_CANDIDATES",
        }

    # Build starting state: top-K positives + K_neg negatives
    k_pos = min(k_positive_start, len(positive_coords_all))
    k_neg = min(k_negative_start, len(negative_coords_all))
    # For negative_selector N_BOTTOMK: first K (lowest foreground score)
    # For N_HIGH_BACKGROUND: last K (highest-score negatives)
    if negative_selector == "N_BOTTOMK":
        start_neg_indices = list(range(k_neg))
    elif negative_selector == "N_HIGH_BACKGROUND":
        start_neg_indices = list(range(len(negative_coords_all) - k_neg, len(negative_coords_all)))
    else:
        start_neg_indices = list(range(k_neg))

    start_pos_indices = list(range(k_pos))
    start_pos_coords = [positive_coords_all[i] for i in start_pos_indices]
    start_neg_coords = [negative_coords_all[i] for i in start_neg_indices
                        if 0 <= i < len(negative_coords_all)]
    # Clip negative indices to valid range
    start_neg_indices = [i for i in start_neg_indices if 0 <= i < len(negative_coords_all)]

    start_dice, _, calls = _evaluate(
        predictor, start_pos_coords, start_neg_coords, target, target_labels,
        target_component_count,
    )

    state = SearchState(
        positive_indices=list(start_pos_indices),
        negative_indices=list(start_neg_indices),
        dice=start_dice,
        sam_calls=calls,
        positive_coords=positive_coords_all,
        negative_coords=negative_coords_all,
    )

    # Greedy add positives up to B_POINT
    state = _greedy_add_positives(
        predictor, state, target, target_labels, target_component_count,
        max_positives=min(b_point, len(positive_coords_all)),
    )
    # Greedy remove positives (try to simplify)
    state = _greedy_remove_positives(
        predictor, state, target, target_labels, target_component_count,
        min_positives=1,
    )
    # Greedy add negatives
    state = _greedy_add_negatives(
        predictor, state, target, target_labels, target_component_count,
        max_negatives=min(b_negative, len(negative_coords_all)),
    )

    final_pos_coords = [positive_coords_all[i] for i in state.positive_indices]
    final_neg_coords = [negative_coords_all[i] for i in state.negative_indices]

    return {
        "positive_indices": list(state.positive_indices),
        "negative_indices": list(state.negative_indices),
        "positive_coords": final_pos_coords,
        "negative_coords": final_neg_coords,
        "search_dice": state.dice,
        "sam_calls": state.sam_calls,
        "k_positive": len(state.positive_indices),
        "k_negative": len(state.negative_indices),
        "status": "OK",
    }


# ---------------------------------------------------------------------------
# Cross-evaluation: apply S_A's indices to B's pool
# ---------------------------------------------------------------------------

def cross_evaluate(
    predictor: Any,
    search_result: dict[str, Any],
    target_pools: PairCandidatePools,
    target: list[list[bool]],
    target_labels: dict[tuple[int, int], int],
    target_component_count: int,
    *,
    positive_pool_name: str = "topk",
) -> tuple[float, int]:
    """Apply search_result's indices to target_pools, return (dice, sam_calls).

    The search_result contains indices into the SOURCE pool. We use the SAME
    indices on the TARGET pool to fetch physical coordinates, then run SAM.
    This measures whether the search's RANK-based selection transfers.
    """
    if positive_pool_name == "topk":
        target_pool = target_pools.topk
    elif positive_pool_name == "spread":
        target_pool = target_pools.spread
    elif positive_pool_name == "local_peaks":
        target_pool = target_pools.local_peaks
    else:
        raise ValueError(f"unknown positive_pool_name: {positive_pool_name}")

    target_pos_coords_all = [c.coordinate for c in target_pool.positives]
    target_neg_coords_all = [c.coordinate for c in target_pools.topk.negatives]

    pos_indices = search_result["positive_indices"]
    neg_indices = search_result["negative_indices"]

    # Clip indices to target pool size (pool sizes may differ between refs)
    pos_coords = [
        target_pos_coords_all[i]
        for i in pos_indices
        if 0 <= i < len(target_pos_coords_all)
    ]
    neg_coords = [
        target_neg_coords_all[i]
        for i in neg_indices
        if 0 <= i < len(target_neg_coords_all)
    ]

    if not pos_coords:
        return 0.0, 0

    dice, _, calls = _evaluate(
        predictor, pos_coords, neg_coords, target, target_labels,
        target_component_count,
    )
    return dice, calls


# ---------------------------------------------------------------------------
# Per-query search oracle: run search on all 6 refs, cross-evaluate
# ---------------------------------------------------------------------------

def run_query_search_oracle(
    predictor: Any,
    query_case_id: str,
    feature_cache: FeatureCache,
    query_target: list[list[bool]],
    query_target_labels: dict[tuple[int, int], int],
    query_target_component_count: int,
    ref_pairs: list[dict[str, str]],
    *,
    positive_pool_name: str = "topk",
) -> dict[str, Any]:
    """Run search oracle for one query across all 6 references.

    Returns per-ref search results + cross-evaluation metrics.
    """
    # Build candidate pools for each ref
    ref_pools: dict[str, PairCandidatePools] = {}  # key = ref_case_id
    for ref_pair in ref_pairs:
        ref_case_id = ref_pair["reference_case_id"]
        if ref_case_id in ref_pools:
            continue
        try:
            pools = build_pair_candidate_pools(query_case_id, ref_case_id, feature_cache)
            ref_pools[ref_case_id] = pools
        except Exception as exc:
            ref_pools[ref_case_id] = None  # type: ignore[assignment]

    # Run search for each ref
    search_results: dict[str, dict[str, Any]] = {}  # key = ref_case_id
    for ref_pair in ref_pairs:
        ref_case_id = ref_pair["reference_case_id"]
        ref_set = ref_pair["reference_set"]
        if ref_case_id in search_results:
            continue
        pools = ref_pools.get(ref_case_id)
        if pools is None:
            search_results[ref_case_id] = {
                "ref_case_id": ref_case_id,
                "ref_set": ref_set,
                "search_dice": 0.0,
                "sam_calls": 0,
                "k_positive": 0,
                "k_negative": 0,
                "positive_indices": [],
                "negative_indices": [],
                "positive_coords": [],
                "negative_coords": [],
                "status": f"POOL_ERROR",
            }
            continue
        result = run_search_for_pair(
            predictor, pools, query_target, query_target_labels,
            query_target_component_count,
            positive_pool_name=positive_pool_name,
        )
        result["ref_case_id"] = ref_case_id
        result["ref_set"] = ref_set
        search_results[ref_case_id] = result

    # Separate by reference set
    a_refs = [r for r in ref_pairs if r["reference_set"] == "A"]
    b_refs = [r for r in ref_pairs if r["reference_set"] == "B"]

    # Raw search Dice on A = mean across A refs
    raw_A_values = [search_results[r["reference_case_id"]]["search_dice"] for r in a_refs]
    raw_B_values = [search_results[r["reference_case_id"]]["search_dice"] for r in b_refs]
    raw_A = _mean(raw_A_values)
    raw_B = _mean(raw_B_values)
    raw_search_dice = (raw_A + raw_B) / 2.0

    # Cross-evaluation: S_A_i's indices on B_j's pool, mean across (i, j)
    cross_A_to_B_values: list[float] = []
    cross_B_to_A_values: list[float] = []
    cross_A_to_B_calls = 0
    cross_B_to_A_calls = 0
    for a_ref in a_refs:
        a_result = search_results[a_ref["reference_case_id"]]
        if a_result.get("status") != "OK":
            continue
        for b_ref in b_refs:
            b_pools = ref_pools.get(b_ref["reference_case_id"])
            if b_pools is None:
                continue
            dice, calls = cross_evaluate(
                predictor, a_result, b_pools, query_target, query_target_labels,
                query_target_component_count,
                positive_pool_name=positive_pool_name,
            )
            cross_A_to_B_values.append(dice)
            cross_A_to_B_calls += calls

    for b_ref in b_refs:
        b_result = search_results[b_ref["reference_case_id"]]
        if b_result.get("status") != "OK":
            continue
        for a_ref in a_refs:
            a_pools = ref_pools.get(a_ref["reference_case_id"])
            if a_pools is None:
                continue
            dice, calls = cross_evaluate(
                predictor, b_result, a_pools, query_target, query_target_labels,
                query_target_component_count,
                positive_pool_name=positive_pool_name,
            )
            cross_B_to_A_values.append(dice)
            cross_B_to_A_calls += calls

    cross_A_to_B = _mean(cross_A_to_B_values)
    cross_B_to_A = _mean(cross_B_to_A_values)
    cross_search_dice = (cross_A_to_B + cross_B_to_A) / 2.0

    # K+ stability: median |K_A - K_B| across all (A, B) ref pairs
    kplus_diffs: list[int] = []
    for a_ref in a_refs:
        ka = search_results[a_ref["reference_case_id"]].get("k_positive", 0)
        for b_ref in b_refs:
            kb = search_results[b_ref["reference_case_id"]].get("k_positive", 0)
            kplus_diffs.append(abs(ka - kb))
    median_kplus_diff = _median([float(d) for d in kplus_diffs])

    # K- stability
    kneg_diffs: list[int] = []
    for a_ref in a_refs:
        ka = search_results[a_ref["reference_case_id"]].get("k_negative", 0)
        for b_ref in b_refs:
            kb = search_results[b_ref["reference_case_id"]].get("k_negative", 0)
            kneg_diffs.append(abs(ka - kb))
    median_kneg_diff = _median([float(d) for d in kneg_diffs])

    # Point-coordinate overlap (Jaccard of physical positive points)
    # Aggregate A's positive coords and B's positive coords, compute Jaccard
    a_pos_coords_all: set[tuple[int, int]] = set()
    for r in a_refs:
        result = search_results[r["reference_case_id"]]
        a_pos_coords_all.update(result.get("positive_coords", []))
    b_pos_coords_all: set[tuple[int, int]] = set()
    for r in b_refs:
        result = search_results[r["reference_case_id"]]
        b_pos_coords_all.update(result.get("positive_coords", []))
    if a_pos_coords_all or b_pos_coords_all:
        union = a_pos_coords_all | b_pos_coords_all
        inter = a_pos_coords_all & b_pos_coords_all
        point_overlap = len(inter) / len(union) if union else 0.0
    else:
        point_overlap = 0.0

    # Candidate-rank overlap (Jaccard of pool indices)
    a_pos_indices_all: set[int] = set()
    for r in a_refs:
        result = search_results[r["reference_case_id"]]
        a_pos_indices_all.update(result.get("positive_indices", []))
    b_pos_indices_all: set[int] = set()
    for r in b_refs:
        result = search_results[r["reference_case_id"]]
        b_pos_indices_all.update(result.get("positive_indices", []))
    if a_pos_indices_all or b_pos_indices_all:
        union = a_pos_indices_all | b_pos_indices_all
        inter = a_pos_indices_all & b_pos_indices_all
        rank_overlap = len(inter) / len(union) if union else 0.0
    else:
        rank_overlap = 0.0

    # Spatial coverage overlap: bounding-box IoU of A's and B's positive points
    def _bbox(coords: set[tuple[int, int]]) -> tuple[int, int, int, int] | None:
        if not coords:
            return None
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        return min(xs), min(ys), max(xs), max(ys)

    a_bbox = _bbox(a_pos_coords_all)
    b_bbox = _bbox(b_pos_coords_all)
    if a_bbox and b_bbox:
        x1 = max(a_bbox[0], b_bbox[0])
        y1 = max(a_bbox[1], b_bbox[1])
        x2 = min(a_bbox[2], b_bbox[2])
        y2 = min(a_bbox[3], b_bbox[3])
        if x2 > x1 and y2 > y1:
            inter_area = (x2 - x1) * (y2 - y1)
        else:
            inter_area = 0
        a_area = (a_bbox[2] - a_bbox[0]) * (a_bbox[3] - a_bbox[1])
        b_area = (b_bbox[2] - b_bbox[0]) * (b_bbox[3] - b_bbox[1])
        spatial_overlap = inter_area / (a_area + b_area - inter_area) if (a_area + b_area - inter_area) > 0 else 0.0
    else:
        spatial_overlap = 0.0

    # Final prompt area bias: |#pos_A - #pos_B| / max(#pos_A, #pos_B)
    a_count = sum(len(search_results[r["reference_case_id"]].get("positive_coords", [])) for r in a_refs)
    b_count = sum(len(search_results[r["reference_case_id"]].get("positive_coords", [])) for r in b_refs)
    area_bias = abs(a_count - b_count) / max(a_count, b_count) if max(a_count, b_count) > 0 else 0.0

    return {
        "query_case_id": query_case_id,
        "raw_search_dice_A": raw_A,
        "raw_search_dice_B": raw_B,
        "raw_search_dice": raw_search_dice,
        "cross_A_to_B": cross_A_to_B,
        "cross_B_to_A": cross_B_to_A,
        "cross_search_dice": cross_search_dice,
        "search_optimism": raw_search_dice - cross_search_dice,
        "median_kplus_diff": median_kplus_diff,
        "median_kneg_diff": median_kneg_diff,
        "point_overlap_jaccard": point_overlap,
        "rank_overlap_jaccard": rank_overlap,
        "spatial_overlap_iou": spatial_overlap,
        "prompt_area_bias": area_bias,
        "total_sam_calls": sum(r.get("sam_calls", 0) for r in search_results.values()) + cross_A_to_B_calls + cross_B_to_A_calls,
        "per_ref_results": list(search_results.values()),
    }


# ---------------------------------------------------------------------------
# Strongest fixed action baseline (from Y1 validation utility matrix)
# ---------------------------------------------------------------------------

def load_strongest_fixed_dice(y1_val_csv: Path) -> tuple[str, float, dict[str, float]]:
    """Load Y1 validation utility matrix, compute strongest fixed action on A.

    Returns (best_action_id, best_mean_dice_on_A, per_query_fixed_dice_on_B).
    The third element is {query: dice_of_best_A_action_on_B} for condition 5.
    """
    if not y1_val_csv.is_file():
        return "", 0.0, {}
    # Use Y1's loader
    from adapromptset.scripts.run_phaseY_gateY1_utility_matrix import (
        load_utility_matrix,
        _aggregate_per_query_across_refs,
    )
    rows = load_utility_matrix(y1_val_csv)
    setA = _aggregate_per_query_across_refs(rows, "A")
    setB = _aggregate_per_query_across_refs(rows, "B")
    queries = sorted(set(setA.keys()) & set(setB.keys()))
    if not queries:
        return "", 0.0, {}
    all_actions = sorted(
        set(aid for q in queries for aid in setA[q].keys())
        & set(aid for q in queries for aid in setB[q].keys())
    )
    action_meanA = {
        aid: _mean([setA[q][aid] for q in queries if aid in setA[q]])
        for aid in all_actions
    }
    best_fixed_action = max(action_meanA, key=action_meanA.get)
    fixed_A_dice = action_meanA[best_fixed_action]
    # Per-query: dice of best_fixed_action on B
    per_query_fixed_on_B = {q: setB[q].get(best_fixed_action, 0.0) for q in queries}
    return best_fixed_action, fixed_A_dice, per_query_fixed_on_B


# ---------------------------------------------------------------------------
# Gate Y2 decision
# ---------------------------------------------------------------------------

def compute_gate_y2(
    per_query: list[dict[str, Any]],
    fixed_dice: float,
    per_query_fixed_on_B: dict[str, float],
) -> dict[str, Any]:
    """Compute Gate Y2 aggregate metrics and decision (5 conditions)."""
    queries = [q for q in per_query if q.get("raw_search_dice") is not None]
    if not queries:
        return {"status": "PHASEY_GATE_Y2_INSUFFICIENT_DATA", "queries": 0}

    raw_search_mean = _mean([q["raw_search_dice"] for q in queries])
    cross_search_mean = _mean([q["cross_search_dice"] for q in queries])
    raw_gap = raw_search_mean - fixed_dice
    search_optimism = raw_search_mean - cross_search_mean
    optimism_ratio = search_optimism / raw_gap if raw_gap > 0 else 1.0

    # Paired bootstrap on (cross_search - fixed_dice) per query
    paired_diffs = [q["cross_search_dice"] - fixed_dice for q in queries]
    ci_lo, ci_hi = _bootstrap_ci(paired_diffs, iterations=2000)

    # Median |K_A - K_B|
    median_kplus_diff = _median([q["median_kplus_diff"] for q in queries])

    # Cross Dice diff per query: |cross_A_to_B - cross_B_to_A|
    cross_diffs = [abs(q["cross_A_to_B"] - q["cross_B_to_A"]) for q in queries]
    median_cross_diff = _median(cross_diffs)

    # % queries: A-search prompt beats strongest fixed on B
    # Use cross_A_to_B (A's search result evaluated on B) vs fixed on B
    beat_fixed_count = 0
    for q in queries:
        qid = q["query_case_id"]
        fixed_on_B = per_query_fixed_on_B.get(qid, fixed_dice)
        if q["cross_A_to_B"] > fixed_on_B:
            beat_fixed_count += 1
    beat_fixed_rate = beat_fixed_count / len(queries)

    # 5 conditions
    cond1 = cross_search_mean >= fixed_dice + 0.03
    cond2 = ci_lo > 0
    cond3 = optimism_ratio <= 0.50
    cond4 = (median_kplus_diff <= 8) or (median_cross_diff <= 0.01)
    cond5 = beat_fixed_rate >= 0.70

    conditions = {
        "cond1_cross_ge_fixed_plus_0p03": cond1,
        "cond2_bootstrap_ci_lo_gt_0": cond2,
        "cond3_optimism_le_50pct_gap": cond3,
        "cond4_k_stable_or_cross_dice_stable": cond4,
        "cond5_70pct_A_beats_fixed_on_B": cond5,
    }

    if all(conditions.values()):
        decision = "PHASEY_GATE_Y2_SEARCH_LABELS_STABLE"
    else:
        decision = "PHASEY_GATE_Y2_SEARCH_ORACLE_OVERFIT"

    return {
        "query_count": len(queries),
        "fixed_dice": fixed_dice,
        "raw_search_dice": raw_search_mean,
        "cross_search_dice": cross_search_mean,
        "raw_search_gap": raw_gap,
        "search_optimism": search_optimism,
        "optimism_ratio": optimism_ratio,
        "bootstrap_ci_lo": ci_lo,
        "bootstrap_ci_hi": ci_hi,
        "median_kplus_diff": median_kplus_diff,
        "median_cross_dice_diff": median_cross_diff,
        "beat_fixed_rate": beat_fixed_rate,
        "mean_point_overlap": _mean([q["point_overlap_jaccard"] for q in queries]),
        "mean_rank_overlap": _mean([q["rank_overlap_jaccard"] for q in queries]),
        "mean_spatial_overlap": _mean([q["spatial_overlap_iou"] for q in queries]),
        "mean_prompt_area_bias": _mean([q["prompt_area_bias"] for q in queries]),
        "conditions": conditions,
        "decision": decision,
    }


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def _per_query_fields() -> list[str]:
    return [
        "query_case_id",
        "raw_search_dice_A", "raw_search_dice_B", "raw_search_dice",
        "cross_A_to_B", "cross_B_to_A", "cross_search_dice",
        "search_optimism",
        "median_kplus_diff", "median_kneg_diff",
        "point_overlap_jaccard", "rank_overlap_jaccard",
        "spatial_overlap_iou", "prompt_area_bias",
        "total_sam_calls",
    ]


def _per_pair_fields() -> list[str]:
    return [
        "query_case_id", "ref_case_id", "ref_set",
        "search_dice", "sam_calls",
        "k_positive", "k_negative",
        "positive_indices_count", "negative_indices_count",
        "status",
    ]


def write_per_query_csv(per_query: list[dict[str, Any]], out_dir: Path) -> None:
    path = out_dir / "search_oracle_per_query.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_per_query_fields())
        writer.writeheader()
        for q in per_query:
            row = {k: q.get(k, "") for k in _per_query_fields()}
            writer.writerow(row)


def write_per_pair_csv(per_query: list[dict[str, Any]], out_dir: Path) -> None:
    path = out_dir / "search_oracle_per_pair.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_per_pair_fields())
        writer.writeheader()
        for q in per_query:
            for r in q.get("per_ref_results", []):
                row = {
                    "query_case_id": q["query_case_id"],
                    "ref_case_id": r.get("ref_case_id", ""),
                    "ref_set": r.get("ref_set", ""),
                    "search_dice": r.get("search_dice", 0.0),
                    "sam_calls": r.get("sam_calls", 0),
                    "k_positive": r.get("k_positive", 0),
                    "k_negative": r.get("k_negative", 0),
                    "positive_indices_count": len(r.get("positive_indices", [])),
                    "negative_indices_count": len(r.get("negative_indices", [])),
                    "status": r.get("status", ""),
                }
                writer.writerow(row)


def write_gate_y2_summary(
    gate: dict[str, Any],
    fixed_action: str,
    out_dir: Path,
) -> None:
    # JSON
    json_path = out_dir / "gateY2_summary.json"
    summary = {
        "fixed_action_baseline": fixed_action,
        **gate,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Markdown
    md_path = out_dir / "gateY2_summary.md"
    lines = [
        "# Phase Y Gate Y2: Search Oracle Stability",
        "",
        f"- **Queries evaluated**: {gate.get('query_count', 0)}",
        f"- **Fixed action baseline**: `{fixed_action}` (Dice={gate.get('fixed_dice', 0):.4f})",
        f"- **Raw search Dice**: {gate.get('raw_search_dice', 0):.4f}",
        f"- **Cross-reference search Dice**: {gate.get('cross_search_dice', 0):.4f}",
        f"- **Raw search gap**: {gate.get('raw_search_gap', 0):.4f}",
        f"- **Search optimism**: {gate.get('search_optimism', 0):.4f} (ratio={gate.get('optimism_ratio', 0):.2%})",
        f"- **Bootstrap 95% CI**: [{gate.get('bootstrap_ci_lo', 0):.4f}, {gate.get('bootstrap_ci_hi', 0):.4f}]",
        f"- **Median |K_A - K_B|**: {gate.get('median_kplus_diff', 0):.1f}",
        f"- **Median cross Dice diff**: {gate.get('median_cross_dice_diff', 0):.4f}",
        f"- **A-search beats fixed on B**: {gate.get('beat_fixed_rate', 0):.1%}",
        f"- **Mean point overlap (Jaccard)**: {gate.get('mean_point_overlap', 0):.4f}",
        f"- **Mean rank overlap (Jaccard)**: {gate.get('mean_rank_overlap', 0):.4f}",
        f"- **Mean spatial overlap (IoU)**: {gate.get('mean_spatial_overlap', 0):.4f}",
        f"- **Mean prompt area bias**: {gate.get('mean_prompt_area_bias', 0):.4f}",
        "",
        "## Gate conditions",
        "",
    ]
    for k, v in gate.get("conditions", {}).items():
        mark = "PASS" if v else "FAIL"
        lines.append(f"- [{mark}] `{k}`")
    lines.extend([
        "",
        f"## Decision: `{gate.get('decision', '')}`",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase Y Gate Y2: Search Oracle Stability"
    )
    parser.add_argument("--split", choices=["train", "validation"], default="validation",
                        help="Split to run search on (validation for Gate Y2)")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    parser.add_argument("--y1-val-csv", type=Path,
                        default=PROJECT_ROOT / "reports" / "phaseY_gateY1" / "utility_matrix_validation.csv",
                        help="Y1 validation utility matrix CSV for fixed-action baseline")
    parser.add_argument("--positive-pool-name", default="topk",
                        choices=["topk", "spread", "local_peaks"],
                        help="Which positive candidate pool to search over")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        args.max_queries = args.max_queries or 2

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[phaseY-Y2] device={device}")

    # Load Y1 fixed-action baseline
    fixed_action, fixed_dice, per_query_fixed_on_B = load_strongest_fixed_dice(args.y1_val_csv)
    print(f"[phaseY-Y2] fixed baseline: {fixed_action} (Dice={fixed_dice:.4f})")

    queries = load_queries(args.split)
    if args.max_queries:
        queries = queries[:args.max_queries]
    print(f"[phaseY-Y2] {len(queries)} queries to process")

    feature_cache = FeatureCache(device=device)
    predictor = load_sam_predictor(device)

    per_query_results: list[dict[str, Any]] = []
    total_sam_calls = 0
    start = time.perf_counter()

    for qi, query in enumerate(queries):
        query_case_id = query["case_id"]
        ref_pairs = load_reference_pairs(query_case_id, ("A", "B"))
        if not ref_pairs:
            print(f"[phaseY-Y2] skip {query_case_id}: no references")
            continue

        # Set SAM image once per query
        query_paths = resolve_case_paths(query_case_id)
        query_image = load_rgb_518(query_paths["image"])
        predictor.set_image(np.asarray(query_image))

        # Load query mask + labels once
        target = load_mask_518(query_paths["mask"])
        target_labels = _component_labels(target)
        target_component_count = len(set(target_labels.values()))

        result = run_query_search_oracle(
            predictor, query_case_id, feature_cache,
            target, target_labels, target_component_count,
            ref_pairs,
            positive_pool_name=args.positive_pool_name,
        )
        per_query_results.append(result)
        total_sam_calls += result["total_sam_calls"]

        elapsed = time.perf_counter() - start
        print(f"[phaseY-Y2] query {qi+1}/{len(queries)} {query_case_id} "
              f"raw={result['raw_search_dice']:.4f} "
              f"cross={result['cross_search_dice']:.4f} "
              f"K_diff={result['median_kplus_diff']:.0f} "
              f"sam_calls={result['total_sam_calls']} "
              f"elapsed={elapsed:.1f}s")

    elapsed = time.perf_counter() - start
    print(f"[phaseY-Y2] all queries done: {len(per_query_results)} queries, "
          f"{total_sam_calls} SAM calls, {elapsed:.1f}s")

    # Write per-query and per-pair CSVs
    write_per_query_csv(per_query_results, out_dir)
    write_per_pair_csv(per_query_results, out_dir)

    # Compute Gate Y2
    gate = compute_gate_y2(per_query_results, fixed_dice, per_query_fixed_on_B)
    print(f"[phaseY-Y2] decision: {gate['decision']}")
    print(f"[phaseY-Y2] raw search Dice: {gate['raw_search_dice']:.4f}")
    print(f"[phaseY-Y2] cross search Dice: {gate['cross_search_dice']:.4f}")
    print(f"[phaseY-Y2] median |K_A-K_B|: {gate['median_kplus_diff']:.1f}")
    print(f"[phaseY-Y2] A beats fixed on B: {gate['beat_fixed_rate']:.1%}")

    write_gate_y2_summary(gate, fixed_action, out_dir)

    # Metadata
    metadata = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "split": args.split,
        "query_count": len(per_query_results),
        "total_sam_calls": total_sam_calls,
        "elapsed_seconds": round(elapsed, 3),
        "gpu_hours": round(elapsed / 3600.0, 4),
        "fixed_action_baseline": fixed_action,
        "fixed_dice": fixed_dice,
        "positive_pool_name": args.positive_pool_name,
        "gate_y2": gate,
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"[phaseY-Y2] artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
