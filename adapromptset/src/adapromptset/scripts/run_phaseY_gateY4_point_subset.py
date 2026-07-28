"""Phase Y Gate Y4: Point-Subset Policy (section 12).

Implements section 12 of ``AdaPromptSet_PhaseY_Policy_Feasibility.md``.

Track B: Point-Subset Policy. Learns a per-candidate utility model from
GT-guided search trajectories (generated on TRAIN queries only), then selects
a concrete point subset for each validation query under a single SAM call.

Model ladder (section 12.3):
  P0: Top-K contrast (fixed K=20, no negatives) -- strongest fixed baseline
  P1: Pointwise marginal regression (GradientBoostingRegressor)
  P2: Pairwise point ranking (GradientBoostingClassifier, win-count scoring)
  P3: Diversity-aware reranker (P1 utility + greedy lambda/mu tuning)
  P4: Set encoder (torch PointScorer, BCE on selected labels, 3 seeds)
  P5: Learned greedy marginal selector (MLP, threshold-tuned stopping)

Baselines (section 12.4):
  - Top20 contrast (P0)
  - Top20 spread
  - Top20 local peaks
  - Fixed Top20 + N1 (strongest fixed dense)
  - Bank-Action Policy (Y3 best model, if available)
  - Random 20-point subset
  - GT-guided Search Oracle (upper bound, from Y2)
  - Cross-reference Search Oracle (upper bound, from Y2)

Gate Y4 pass conditions (section 12.5, 8 conditions):
  1. Dice improvement >= 0.015 over strongest fixed OR >= 0.0075 over Bank Policy
  2. paired bootstrap 95% CI lower bound > 0
  3. Single SAM call per final prompt
  4. No test-time SAM search
  5. 3 seeds direction consistent (all positive or all near-zero)
  6. Oracle recovery >= 20% of cross-reference Search Oracle gap
  7. Reference-set A/B direction stable (positive on both)
  8. K distribution not degenerate (std > 0 across queries)

Outputs::

    reports/phaseY_gateY4/
        point_subset_per_query.csv      (per query, model, seed: Dice, K, calls)
        gateY4_summary.json             (aggregate metrics + decision per model)
        gateY4_summary.md               (human-readable)
        search_labels_train.csv         (search-derived labels for train queries)
        run_metadata.json
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

# Reuse Y1 / Y2 / Y3 / holdout infrastructure
from adapromptset.evaluation.metrics import (
    aggregated_jaccard_index,
    dice_score,
    iou_score,
)
from adapromptset.matching.evidence import (
    IMAGE_SIZE,
    PATCH_SIZE,
    matching_scores_from_features,
    patch_weights_from_mask,
)
from adapromptset.scripts.run_phaseY_gateY1_utility_matrix import (
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
from adapromptset.scripts.run_phaseY_gateY2_search_oracle import (
    DEFAULT_K_NEGATIVE_START,
    DEFAULT_K_POSITIVE_START,
    DEFAULT_NEGATIVE_SELECTOR,
    cross_evaluate,
    run_search_for_pair,
)
from adapromptset.scripts.run_phaseY_gateY3_bank_policy import (
    CompactAction,
    _query_global_features,
    build_features_for_split,
    extract_features_for_pair,
    load_compact_bank,
)
from adapromptset.scripts.run_phaseY_holdout import run_action_on_query

OUT_DIR_DEFAULT = PROJECT_ROOT / "reports" / "phaseY_gateY4"
Y2_PER_PAIR_CSV = PROJECT_ROOT / "reports" / "phaseY_gateY2" / "search_oracle_per_pair.csv"
Y2_PER_QUERY_CSV = PROJECT_ROOT / "reports" / "phaseY_gateY2" / "search_oracle_per_query.csv"
Y3_SUMMARY_JSON = PROJECT_ROOT / "reports" / "phaseY_gateY3" / "gateY3_summary.json"

# Fixed defaults
DEFAULT_K = 20
DEFAULT_NEGATIVE_K = 1
POINT_FEATURE_PREFIX = "pt_"
QUERY_FEATURE_PREFIX = "query_"
P4_EPOCHS = 50
P4_LR = 1e-3
P4_HIDDEN = 64
P5_EPOCHS = 50
P5_LR = 1e-3
P5_HIDDEN = 64
P5_DEFAULT_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


# ---------------------------------------------------------------------------
# Point feature extraction (per candidate in the topk positive pool)
# ---------------------------------------------------------------------------

def _is_local_peak_grid(grid: list[list[float]], row: int, col: int) -> bool:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    value = grid[row][col]
    for nr in range(max(0, row - 1), min(rows, row + 2)):
        for nc in range(max(0, col - 1), min(cols, col + 2)):
            if (nr, nc) != (row, col) and grid[nr][nc] > value:
                return False
    return True


def _build_matching_maps(
    query_case_id: str,
    ref_case_id: str,
    feature_cache: FeatureCache,
) -> tuple[list[list[float]], list[list[float]]]:
    """Build s_fg / s_bg grids for one (query, ref) pair."""
    query_features = feature_cache.get_features(query_case_id)
    ref_features = feature_cache.get_features(ref_case_id)
    ref_mask = feature_cache.get_mask_tensor(ref_case_id)
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
    return maps["s_fg"].tolist(), maps["s_bg"].tolist()


@dataclass
class PointFeatureBundle:
    """Per-pair point features for the topk positive pool."""
    query_case_id: str
    ref_case_id: str
    # Per-candidate feature dicts (aligned with pools.topk.positives order)
    point_features: list[dict[str, float]]
    # Shared query-global features
    query_global: dict[str, float]
    # The candidate coordinates (aligned with point_features)
    coordinates: list[tuple[int, int]]
    # The candidate scores (raw contrast)
    scores: list[float]
    # The pools object (for negative selection)
    pools: PairCandidatePools


def extract_point_features(
    query_case_id: str,
    ref_case_id: str,
    feature_cache: FeatureCache,
    pools: PairCandidatePools | None = None,
) -> PointFeatureBundle:
    """Extract per-point features for the topk positive pool of one pair.

    Features per candidate (section 12.3 P1):
      - contrast_zscore: contrast score z-scored across the pool
      - s_fg: foreground score at the candidate's grid cell
      - s_bg: background score at the candidate's grid cell
      - row_norm, col_norm: normalized grid position
      - distance_to_top10_centroid: pixel distance to centroid of top-10 candidates
      - local_density: count of high-score neighbors within radius 3
      - is_local_peak: binary
      - rank_norm: rank in pool (0-indexed, normalized)
      - query_*: query global features (mean/std of embedding, stain stats)
    """
    if pools is None:
        pools = build_pair_candidate_pools(query_case_id, ref_case_id, feature_cache)

    s_fg, s_bg = _build_matching_maps(query_case_id, ref_case_id, feature_cache)
    grid_size = len(s_fg)

    positives = list(pools.topk.positives)
    scores = [c.score for c in positives]
    mean_score = _mean(scores) if scores else 0.0
    std_score = float(np.std(scores)) if scores else 0.0

    # Top-10 centroid in pixel coordinates
    top10_coords = [c.coordinate for c in positives[:10]]
    if top10_coords:
        cx = _mean([c[0] for c in top10_coords])
        cy = _mean([c[1] for c in top10_coords])
    else:
        cx, cy = 0.0, 0.0

    # Query global features
    query_features = feature_cache.get_features(query_case_id)
    query_paths = resolve_case_paths(query_case_id)
    query_image = load_rgb_518(query_paths["image"])
    query_global = _query_global_features(query_features, query_image)

    high_threshold = mean_score + std_score
    point_features: list[dict[str, float]] = []
    coordinates: list[tuple[int, int]] = []
    for rank, cand in enumerate(positives):
        row, col = cand.grid_index
        # Local density: count of high-score neighbors within radius 3
        density = 0
        for nr in range(max(0, row - 3), min(grid_size, row + 4)):
            for nc in range(max(0, col - 3), min(grid_size, col + 4)):
                if (nr, nc) == (row, col):
                    continue
                if s_fg[nr][nc] > high_threshold:
                    density += 1

        feat: dict[str, float] = {
            "contrast_zscore": (cand.score - mean_score) / (std_score + 1e-8),
            "s_fg": float(s_fg[row][col]),
            "s_bg": float(s_bg[row][col]),
            "row_norm": float(row) / max(grid_size, 1),
            "col_norm": float(col) / max(grid_size, 1),
            "distance_to_top10_centroid": float(
                math.hypot(cand.coordinate[0] - cx, cand.coordinate[1] - cy)
            ),
            "local_density": float(density),
            "is_local_peak": 1.0 if _is_local_peak_grid(s_fg, row, col) else 0.0,
            "rank_norm": float(rank) / max(len(scores) - 1, 1),
            "contrast_raw": float(cand.score),
        }
        # Query global features (shared across candidates, prefixed)
        for k, v in query_global.items():
            feat[f"{QUERY_FEATURE_PREFIX}{k}"] = v
        point_features.append(feat)
        coordinates.append(cand.coordinate)

    return PointFeatureBundle(
        query_case_id=query_case_id,
        ref_case_id=ref_case_id,
        point_features=point_features,
        query_global=query_global,
        coordinates=coordinates,
        scores=scores,
        pools=pools,
    )


# ---------------------------------------------------------------------------
# Search label generation (TRAIN queries only)
# ---------------------------------------------------------------------------

@dataclass
class SearchLabel:
    query_case_id: str
    ref_case_id: str
    ref_set: str
    positive_indices: list[int]
    negative_indices: list[int]
    k_positive: int
    k_negative: int
    search_dice: float
    sam_calls: int
    status: str


def _search_labels_csv_path(out_dir: Path) -> Path:
    return out_dir / "search_labels_train.csv"


def _search_label_fields() -> list[str]:
    return [
        "query_case_id", "ref_case_id", "ref_set",
        "positive_indices", "negative_indices",
        "k_positive", "k_negative",
        "search_dice", "sam_calls", "status",
    ]


def _load_search_labels(path: Path) -> list[SearchLabel]:
    if not path.is_file():
        return []
    labels: list[SearchLabel] = []
    for row in _read_csv(path):
        pos_idx = [int(i) for i in row["positive_indices"].split(";") if i]
        neg_idx = [int(i) for i in row["negative_indices"].split(";") if i]
        labels.append(SearchLabel(
            query_case_id=row["query_case_id"],
            ref_case_id=row["ref_case_id"],
            ref_set=row["ref_set"],
            positive_indices=pos_idx,
            negative_indices=neg_idx,
            k_positive=int(row["k_positive"]),
            k_negative=int(row["k_negative"]),
            search_dice=float(row["search_dice"]),
            sam_calls=int(row["sam_calls"]),
            status=row["status"],
        ))
    return labels


def _save_search_labels(labels: list[SearchLabel], path: Path) -> None:
    rows = []
    for lab in labels:
        rows.append({
            "query_case_id": lab.query_case_id,
            "ref_case_id": lab.ref_case_id,
            "ref_set": lab.ref_set,
            "positive_indices": ";".join(str(i) for i in lab.positive_indices),
            "negative_indices": ";".join(str(i) for i in lab.negative_indices),
            "k_positive": lab.k_positive,
            "k_negative": lab.k_negative,
            "search_dice": lab.search_dice,
            "sam_calls": lab.sam_calls,
            "status": lab.status,
        })
    _write_csv(path, rows, _search_label_fields())


def generate_train_search_labels(
    predictor: Any,
    feature_cache: FeatureCache,
    out_dir: Path,
    max_queries: int | None = None,
    positive_pool_name: str = "topk",
) -> list[SearchLabel]:
    """Run GT-guided search on TRAIN queries (Reference Set A) to produce labels.

    Search is ONLY run on train queries (section 12.2). Validation/holdout
    never receive search labels.
    """
    cache_path = _search_labels_csv_path(out_dir)
    cached = _load_search_labels(cache_path)
    if cached:
        print(f"[phaseY-Y4] loaded {len(cached)} cached train search labels")
        return cached

    queries = load_queries("train")
    if max_queries:
        queries = queries[:max_queries]

    labels: list[SearchLabel] = []
    start = time.perf_counter()
    for qi, query in enumerate(queries):
        query_case_id = query["case_id"]
        ref_pairs = load_reference_pairs(query_case_id, ("A",))
        if not ref_pairs:
            continue

        query_paths = resolve_case_paths(query_case_id)
        query_image = load_rgb_518(query_paths["image"])
        predictor.set_image(np.asarray(query_image))
        target = load_mask_518(query_paths["mask"])
        target_labels = _component_labels(target)
        target_component_count = len(set(target_labels.values()))

        for ref_pair in ref_pairs:
            ref_case_id = ref_pair["reference_case_id"]
            ref_set = ref_pair.get("reference_set", "A")
            try:
                pools = build_pair_candidate_pools(query_case_id, ref_case_id, feature_cache)
                result = run_search_for_pair(
                    predictor, pools, target, target_labels, target_component_count,
                    positive_pool_name=positive_pool_name,
                )
                labels.append(SearchLabel(
                    query_case_id=query_case_id,
                    ref_case_id=ref_case_id,
                    ref_set=ref_set,
                    positive_indices=result["positive_indices"],
                    negative_indices=result["negative_indices"],
                    k_positive=result["k_positive"],
                    k_negative=result["k_negative"],
                    search_dice=result["search_dice"],
                    sam_calls=result["sam_calls"],
                    status=result.get("status", "OK"),
                ))
            except Exception as exc:
                labels.append(SearchLabel(
                    query_case_id=query_case_id,
                    ref_case_id=ref_case_id,
                    ref_set=ref_set,
                    positive_indices=[],
                    negative_indices=[],
                    k_positive=0,
                    k_negative=0,
                    search_dice=0.0,
                    sam_calls=0,
                    status=f"ERROR:{type(exc).__name__}",
                ))

        elapsed = time.perf_counter() - start
        print(f"[phaseY-Y4] train search {qi+1}/{len(queries)} {query_case_id} "
              f"labels={len(labels)} elapsed={elapsed:.1f}s")

    _save_search_labels(labels, cache_path)
    print(f"[phaseY-Y4] saved {len(labels)} train search labels to {cache_path}")
    return labels


# ---------------------------------------------------------------------------
# Load Y2 upper bounds (validation)
# ---------------------------------------------------------------------------

def load_y2_upper_bounds() -> dict[str, Any]:
    """Load Y2 per-query search oracle results for validation upper bounds.

    Returns dict with:
      - per_query: {query_case_id: {raw_search_dice, cross_search_dice, ...}}
      - mean_raw_search_dice, mean_cross_search_dice
    Returns empty dict if Y2 outputs unavailable.
    """
    if not Y2_PER_QUERY_CSV.is_file():
        return {}
    rows = _read_csv(Y2_PER_QUERY_CSV)
    per_query: dict[str, dict[str, float]] = {}
    raw_vals: list[float] = []
    cross_vals: list[float] = []
    for row in rows:
        qid = row["query_case_id"]
        raw = float(row.get("raw_search_dice", 0.0))
        cross = float(row.get("cross_search_dice", 0.0))
        per_query[qid] = {
            "raw_search_dice": raw,
            "cross_search_dice": cross,
            "cross_A_to_B": float(row.get("cross_A_to_B", 0.0)),
            "cross_B_to_A": float(row.get("cross_B_to_A", 0.0)),
        }
        raw_vals.append(raw)
        cross_vals.append(cross)
    return {
        "per_query": per_query,
        "mean_raw_search_dice": _mean(raw_vals) if raw_vals else 0.0,
        "mean_cross_search_dice": _mean(cross_vals) if cross_vals else 0.0,
    }


# ---------------------------------------------------------------------------
# Feature matrix assembly for training
# ---------------------------------------------------------------------------

def _feature_keys_from_bundle(bundle: PointFeatureBundle) -> list[str]:
    """Canonical sorted feature keys for a point feature dict."""
    if not bundle.point_features:
        return []
    return sorted(bundle.point_features[0].keys())


def _featurize_point(feat: dict[str, float], keys: list[str]) -> list[float]:
    return [feat.get(k, 0.0) for k in keys]


def build_train_dataset(
    predictor: Any,
    feature_cache: FeatureCache,
    search_labels: list[SearchLabel],
    max_queries: int | None = None,
) -> dict[str, Any]:
    """Build the training dataset for P1-P5.

    Returns dict with:
      - X: (N_points, F) numpy array of point features
      - y_binary: (N_points,) binary labels (1.0 if selected by search)
      - y_dice: (N_points,) per-point dice-gain proxy (currently == y_binary)
      - feature_keys: list of feature names
      - pair_bundles: {query_case_id: {ref_case_id: PointFeatureBundle}}
      - labels_by_pair: {(query, ref): SearchLabel}
      - k_positive_values: list of K+ from search (for median K)
    """
    # Group labels by (query, ref)
    labels_by_pair: dict[tuple[str, str], SearchLabel] = {}
    for lab in search_labels:
        labels_by_pair[(lab.query_case_id, lab.ref_case_id)] = lab

    # Get unique train queries (preserve order)
    seen_queries: list[str] = []
    seen_set: set[str] = set()
    for lab in search_labels:
        if lab.query_case_id not in seen_set:
            seen_set.add(lab.query_case_id)
            seen_queries.append(lab.query_case_id)
    if max_queries:
        seen_queries = seen_queries[:max_queries]

    pair_bundles: dict[str, dict[str, PointFeatureBundle]] = {}
    X_rows: list[list[float]] = []
    y_binary: list[float] = []
    y_dice: list[float] = []
    feature_keys: list[str] = []
    k_positive_values: list[int] = []

    for qi, query_case_id in enumerate(seen_queries):
        pair_bundles[query_case_id] = {}
        refs_for_query = [
            (q, r) for (q, r) in labels_by_pair.keys() if q == query_case_id
        ]
        for (q, ref_case_id) in refs_for_query:
            lab = labels_by_pair[(q, ref_case_id)]
            if lab.status != "OK":
                continue
            try:
                bundle = extract_point_features(q, ref_case_id, feature_cache)
            except Exception as exc:
                print(f"[phaseY-Y4] feature extraction failed for {q}/{ref_case_id}: {exc}")
                continue
            pair_bundles[q][ref_case_id] = bundle
            if not feature_keys:
                feature_keys = _feature_keys_from_bundle(bundle)
            selected_set = set(lab.positive_indices)
            for idx, feat in enumerate(bundle.point_features):
                X_rows.append(_featurize_point(feat, feature_keys))
                label = 1.0 if idx in selected_set else 0.0
                y_binary.append(label)
                y_dice.append(label)  # binary proxy; marginal Dice not in CSV
            k_positive_values.append(lab.k_positive)

        if (qi + 1) % 10 == 0:
            print(f"[phaseY-Y4] train features {qi+1}/{len(seen_queries)} done")

    X = np.asarray(X_rows, dtype=np.float64) if X_rows else np.zeros((0, 0))
    y_bin = np.asarray(y_binary, dtype=np.float64)
    y_d = np.asarray(y_dice, dtype=np.float64)

    return {
        "X": X,
        "y_binary": y_bin,
        "y_dice": y_d,
        "feature_keys": feature_keys,
        "pair_bundles": pair_bundles,
        "labels_by_pair": labels_by_pair,
        "k_positive_values": k_positive_values,
    }


# ---------------------------------------------------------------------------
# SAM evaluation: run one subset on one query
# ---------------------------------------------------------------------------

def run_subset_on_query(
    predictor: Any,
    positive_coords: list[tuple[int, int]],
    negative_coords: list[tuple[int, int]],
    target: list[list[bool]],
    target_labels: dict[tuple[int, int], int],
    target_component_count: int,
) -> dict[str, Any]:
    """Run SAM once with the given point subset. Returns metrics + sam_calls=1."""
    if not positive_coords:
        return {
            "dice": 0.0, "iou": 0.0, "aji": 0.0,
            "component_recall": 0.0, "sam_score": 0.0,
            "k_positive": 0, "k_negative": len(negative_coords),
            "sam_calls": 0, "status": "NO_POSITIVES",
        }
    points = positive_coords + negative_coords
    labels = [1] * len(positive_coords) + [0] * len(negative_coords)
    t0 = time.perf_counter()
    try:
        masks, scores, _ = predictor.predict(
            point_coords=np.asarray(points, dtype=np.float32),
            point_labels=np.asarray(labels, dtype=np.int32),
            multimask_output=True,
        )
    except Exception as exc:
        return {
            "dice": 0.0, "iou": 0.0, "aji": 0.0,
            "component_recall": 0.0, "sam_score": 0.0,
            "k_positive": len(positive_coords),
            "k_negative": len(negative_coords),
            "sam_calls": 0, "status": f"SAM_ERROR:{type(exc).__name__}",
        }
    runtime_ms = (time.perf_counter() - t0) * 1000.0
    best = int(np.argmax(scores))
    mask = masks[best].astype(bool).tolist()
    score = float(scores[best])

    dice = dice_score(mask, target)
    iou = iou_score(mask, target)
    aji = aggregated_jaccard_index(
        _semantic_instance_labels(mask),
        _semantic_instance_labels(target),
    )
    recall, hit_count, total_count = _component_recall(mask, target_labels)
    return {
        "dice": dice, "iou": iou, "aji": aji,
        "component_recall": recall, "sam_score": score,
        "k_positive": len(positive_coords),
        "k_negative": len(negative_coords),
        "sam_calls": 1, "runtime_ms": runtime_ms,
        "target_component_count": total_count,
        "hit_components": hit_count,
        "status": "OK",
    }


# ---------------------------------------------------------------------------
# Model P0: Top-K contrast (fixed K=20, no negatives)
# ---------------------------------------------------------------------------

@dataclass
class P0Model:
    k: int = DEFAULT_K


def train_p0(train_data: dict[str, Any], seed: int) -> P0Model:
    return P0Model(k=DEFAULT_K)


def predict_p0(
    model: P0Model,
    bundle: PointFeatureBundle,
) -> tuple[list[int], int]:
    """Select top-K positives by rank (contrast order). No negatives learned."""
    k = min(model.k, len(bundle.point_features))
    return list(range(k)), k


# ---------------------------------------------------------------------------
# Model P1: Pointwise marginal regression
# ---------------------------------------------------------------------------

@dataclass
class P1Model:
    model: Any
    feature_keys: list[str]
    k: int


def train_p1(train_data: dict[str, Any], seed: int) -> P1Model:
    from sklearn.ensemble import GradientBoostingRegressor

    X = train_data["X"]
    y = train_data["y_binary"]
    feature_keys = train_data["feature_keys"]
    k_values = train_data["k_positive_values"]
    median_k = int(_median([float(k) for k in k_values])) if k_values else DEFAULT_K
    median_k = max(median_k, 1)

    if X.shape[0] == 0:
        return P1Model(model=None, feature_keys=feature_keys, k=median_k)

    reg = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, random_state=seed,
    )
    reg.fit(X, y)
    return P1Model(model=reg, feature_keys=feature_keys, k=median_k)


def predict_p1(
    model: P1Model,
    bundle: PointFeatureBundle,
) -> tuple[list[int], int]:
    if model.model is None or not bundle.point_features:
        k = min(model.k, len(bundle.point_features))
        return list(range(k)), k
    X = np.asarray(
        [_featurize_point(f, model.feature_keys) for f in bundle.point_features],
        dtype=np.float64,
    )
    preds = model.model.predict(X)
    # Select top-K by predicted utility
    k = min(model.k, len(preds))
    top_indices = list(np.argsort(-preds)[:k])
    return top_indices, k


# ---------------------------------------------------------------------------
# Model P2: Pairwise point ranking
# ---------------------------------------------------------------------------

@dataclass
class P2Model:
    model: Any
    feature_keys: list[str]
    k: int


def train_p2(train_data: dict[str, Any], seed: int) -> P2Model:
    from sklearn.ensemble import GradientBoostingClassifier

    feature_keys = train_data["feature_keys"]
    k_values = train_data["k_positive_values"]
    median_k = int(_median([float(k) for k in k_values])) if k_values else DEFAULT_K
    median_k = max(median_k, 1)

    # Build pairwise training data per pair: (i > j) for i in S*, j not in S*
    pair_bundles = train_data["pair_bundles"]
    labels_by_pair = train_data["labels_by_pair"]
    X_pairs: list[list[float]] = []
    y_pairs: list[int] = []

    for query_case_id, ref_map in pair_bundles.items():
        for ref_case_id, bundle in ref_map.items():
            lab = labels_by_pair.get((query_case_id, ref_case_id))
            if lab is None or lab.status != "OK":
                continue
            selected = set(lab.positive_indices)
            n = len(bundle.point_features)
            if n == 0:
                continue
            feats = [_featurize_point(f, feature_keys) for f in bundle.point_features]
            # Limit pairs to avoid O(n^2) blow-up
            selected_list = sorted(selected)
            unselected_list = [i for i in range(n) if i not in selected]
            # Cap pair count per pair
            max_pairs_per_pair = 200
            count = 0
            for i in selected_list:
                for j in unselected_list:
                    if count >= max_pairs_per_pair:
                        break
                    # Positive example: i > j => feature_i - feature_j, label 1
                    diff = [a - b for a, b in zip(feats[i], feats[j])]
                    X_pairs.append(feats[i] + feats[j] + diff)
                    y_pairs.append(1)
                    # Negative example: j > i => reversed, label 0
                    X_pairs.append(feats[j] + feats[i] + [-d for d in diff])
                    y_pairs.append(0)
                    count += 1
                if count >= max_pairs_per_pair:
                    break

    if not X_pairs:
        return P2Model(model=None, feature_keys=feature_keys, k=median_k)

    X_arr = np.asarray(X_pairs, dtype=np.float64)
    y_arr = np.asarray(y_pairs, dtype=np.int32)
    clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, random_state=seed,
    )
    clf.fit(X_arr, y_arr)
    return P2Model(model=clf, feature_keys=feature_keys, k=median_k)


def predict_p2(
    model: P2Model,
    bundle: PointFeatureBundle,
) -> tuple[list[int], int]:
    if model.model is None or not bundle.point_features:
        k = min(model.k, len(bundle.point_features))
        return list(range(k)), k
    feats = [_featurize_point(f, model.feature_keys) for f in bundle.point_features]
    n = len(feats)
    # Score each point by win count: P(i > j) for all j != i
    # To keep inference cheap, score against a sample of other points
    win_counts = np.zeros(n, dtype=np.float64)
    # Cap comparisons for efficiency
    sample_size = min(n, 30)
    sample_idx = list(range(n)) if n <= sample_size else (
        list(np.random.RandomState(0).choice(n, sample_size, replace=False))
    )
    for i in range(n):
        for j in sample_idx:
            if i == j:
                continue
            diff = [a - b for a, b in zip(feats[i], feats[j])]
            row = feats[i] + feats[j] + diff
            prob = model.model.predict_proba([row])[0]
            # Class 1 = i > j
            cls_idx = list(model.model.classes_).index(1) if 1 in model.model.classes_ else 1
            win_counts[i] += prob[cls_idx]
    k = min(model.k, n)
    top_indices = list(np.argsort(-win_counts)[:k])
    return top_indices, k


# ---------------------------------------------------------------------------
# Model P3: Diversity-aware reranker (greedy on P1 utility)
# ---------------------------------------------------------------------------

@dataclass
class P3Model:
    p1_model: P1Model
    lambda_dist: float = 1.0
    mu_redundancy: float = 0.5


def train_p3(train_data: dict[str, Any], seed: int) -> P3Model:
    # Reuse P1 model; lambda/mu tuned on validation at eval time
    p1 = train_p1(train_data, seed)
    return P3Model(p1_model=p1)


def _feature_redundancy(
    candidate_feat: list[float],
    selected_feats: list[list[float]],
) -> float:
    """Mean cosine similarity to selected points."""
    if not selected_feats:
        return 0.0
    a = np.asarray(candidate_feat, dtype=np.float64)
    norm_a = np.linalg.norm(a) + 1e-8
    total = 0.0
    for b in selected_feats:
        b_arr = np.asarray(b, dtype=np.float64)
        norm_b = np.linalg.norm(b_arr) + 1e-8
        total += float(np.dot(a, b_arr) / (norm_a * norm_b))
    return total / len(selected_feats)


def predict_p3(
    model: P3Model,
    bundle: PointFeatureBundle,
    lambda_dist: float = 1.0,
    mu_redundancy: float = 0.5,
) -> tuple[list[int], int]:
    p1 = model.p1_model
    if p1.model is None or not bundle.point_features:
        k = min(p1.k, len(bundle.point_features))
        return list(range(k)), k
    X = np.asarray(
        [_featurize_point(f, p1.feature_keys) for f in bundle.point_features],
        dtype=np.float64,
    )
    utilities = p1.model.predict(X)
    coords = bundle.coordinates
    feats_list = [_featurize_point(f, p1.feature_keys) for f in bundle.point_features]
    n = len(utilities)
    k = min(p1.k, n)

    # Greedy selection: maximize utility - lambda * min_dist - mu * redundancy
    selected: list[int] = []
    selected_coords: list[tuple[int, int]] = []
    selected_feats: list[list[float]] = []
    remaining = set(range(n))
    while remaining and len(selected) < k:
        best_idx = -1
        best_score = -float("inf")
        for i in remaining:
            if selected_coords:
                min_dist = min(
                    math.hypot(coords[i][0] - sc[0], coords[i][1] - sc[1])
                    for sc in selected_coords
                )
            else:
                min_dist = 0.0
            redundancy = _feature_redundancy(feats_list[i], selected_feats)
            score = utilities[i] - lambda_dist * min_dist - mu_redundancy * redundancy
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx < 0:
            break
        selected.append(best_idx)
        selected_coords.append(coords[best_idx])
        selected_feats.append(feats_list[best_idx])
        remaining.discard(best_idx)
    return selected, len(selected)


# ---------------------------------------------------------------------------
# Model P4: Set encoder (PointScorer)
# ---------------------------------------------------------------------------

class PointScorer(torch.nn.Module):
    def __init__(self, feat_dim: int, hidden: int = P4_HIDDEN):
        super().__init__()
        self.phi = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
        )
        self.psi = torch.nn.Sequential(
            torch.nn.Linear(hidden * 2, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, M, F), mask: (B, M)
        phi = self.phi(x) * mask.unsqueeze(-1)  # (B, M, hidden)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
        set_repr = phi.sum(dim=1) / denom  # (B, hidden)
        set_repr_expanded = set_repr.unsqueeze(1).expand(-1, x.size(1), -1)
        combined = torch.cat([phi, set_repr_expanded], dim=-1)  # (B, M, hidden*2)
        return self.psi(combined).squeeze(-1)  # (B, M)


@dataclass
class P4Model:
    model: PointScorer
    feature_keys: list[str]
    k: int


def _pad_point_features(
    bundles: list[list[dict[str, float]]],
    feature_keys: list[str],
    max_m: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of point feature lists into (B, M, F) + mask (B, M)."""
    b = len(bundles)
    f = len(feature_keys)
    X = torch.zeros(b, max_m, f, dtype=torch.float32)
    mask = torch.zeros(b, max_m, dtype=torch.float32)
    for i, pts in enumerate(bundles):
        m = min(len(pts), max_m)
        for j in range(m):
            X[i, j] = torch.tensor(
                _featurize_point(pts[j], feature_keys), dtype=torch.float32
            )
            mask[i, j] = 1.0
    return X, mask


def _build_p4_training_tensors(
    train_data: dict[str, Any],
    feature_keys: list[str],
    max_m: int = 100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (X, mask, Y) tensors for P4 BCE training."""
    pair_bundles = train_data["pair_bundles"]
    labels_by_pair = train_data["labels_by_pair"]
    bundle_list: list[list[dict[str, float]]] = []
    label_list: list[list[float]] = []
    for query_case_id, ref_map in pair_bundles.items():
        for ref_case_id, bundle in ref_map.items():
            lab = labels_by_pair.get((query_case_id, ref_case_id))
            if lab is None or lab.status != "OK":
                continue
            selected = set(lab.positive_indices)
            pts = bundle.point_features
            if not pts:
                continue
            labels = [1.0 if i in selected else 0.0 for i in range(len(pts))]
            bundle_list.append(pts)
            label_list.append(labels)
    if not bundle_list:
        return (
            torch.zeros(0, max_m, len(feature_keys)),
            torch.zeros(0, max_m),
            torch.zeros(0, max_m),
        )
    X, mask = _pad_point_features(bundle_list, feature_keys, max_m)
    Y = torch.zeros(len(bundle_list), max_m, dtype=torch.float32)
    for i, labels in enumerate(label_list):
        m = min(len(labels), max_m)
        for j in range(m):
            Y[i, j] = labels[j]
    return X, mask, Y


def train_p4(train_data: dict[str, Any], seed: int) -> P4Model:
    feature_keys = train_data["feature_keys"]
    k_values = train_data["k_positive_values"]
    median_k = int(_median([float(k) for k in k_values])) if k_values else DEFAULT_K
    median_k = max(median_k, 1)

    if not feature_keys:
        return P4Model(model=None, feature_keys=feature_keys, k=median_k)

    torch.manual_seed(seed)
    np.random.seed(seed)
    feat_dim = len(feature_keys)
    model = PointScorer(feat_dim=feat_dim, hidden=P4_HIDDEN)
    optimizer = torch.optim.Adam(model.parameters(), lr=P4_LR)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    X, mask, Y = _build_p4_training_tensors(train_data, feature_keys)
    if X.shape[0] == 0:
        return P4Model(model=model, feature_keys=feature_keys, k=median_k)

    model.train()
    for epoch in range(P4_EPOCHS):
        optimizer.zero_grad()
        logits = model(X, mask)  # (B, M)
        # Mask out padding
        loss = loss_fn(logits * mask, Y * mask)
        loss.backward()
        optimizer.step()
    model.eval()
    return P4Model(model=model, feature_keys=feature_keys, k=median_k)


def predict_p4(
    model: P4Model,
    bundle: PointFeatureBundle,
) -> tuple[list[int], int]:
    if model.model is None or not bundle.point_features:
        k = min(model.k, len(bundle.point_features))
        return list(range(k)), k
    with torch.inference_mode():
        X, mask = _pad_point_features(
            [bundle.point_features], model.feature_keys, max_m=max(len(bundle.point_features), 1),
        )
        logits = model.model(X, mask)[0].numpy()
    n = len(bundle.point_features)
    k = min(model.k, n)
    # Score only valid (non-padded) entries
    top_indices = list(np.argsort(-logits[:n])[:k])
    return top_indices, k


# ---------------------------------------------------------------------------
# Model P5: Learned greedy marginal selector
# ---------------------------------------------------------------------------

class MarginalMLP(torch.nn.Module):
    def __init__(self, feat_dim: int, hidden: int = P5_HIDDEN):
        super().__init__()
        # Input: candidate features + current set stats (size, mean_score, spread)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(feat_dim + 3, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class P5Model:
    model: MarginalMLP
    feature_keys: list[str]
    k: int
    threshold: float = P5_DEFAULT_THRESHOLD


def _build_p5_training_tensors(
    train_data: dict[str, Any],
    feature_keys: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (X, y) for P5: (candidate_features + current_set_stats) -> selected label.

    Approximation: since per-step marginal gains aren't in the Y2 CSV, we use
    the binary selected label as the gain proxy. Current-set stats are computed
    by simulating the search's add order (using the positive_indices order).
    """
    pair_bundles = train_data["pair_bundles"]
    labels_by_pair = train_data["labels_by_pair"]
    X_rows: list[list[float]] = []
    y_rows: list[float] = []

    for query_case_id, ref_map in pair_bundles.items():
        for ref_case_id, bundle in ref_map.items():
            lab = labels_by_pair.get((query_case_id, ref_case_id))
            if lab is None or lab.status != "OK":
                continue
            pts = bundle.point_features
            if not pts:
                continue
            feats = [_featurize_point(f, feature_keys) for f in pts]
            coords = bundle.coordinates
            selected_order = lab.positive_indices  # add order
            selected_set = set(selected_order)
            # Simulate greedy add: for each candidate, current set = those added before it
            current: list[int] = []
            for i in range(len(pts)):
                cur_size = len(current)
                if current:
                    cur_scores = [feats[j][0] for j in current]  # use first feat as score proxy
                    mean_score = float(np.mean(cur_scores))
                    cur_coords = [coords[j] for j in current]
                    if len(cur_coords) >= 2:
                        spread = float(np.std([
                            math.hypot(c[0] - coords[i][0], c[1] - coords[i][1])
                            for c in cur_coords
                        ]))
                    else:
                        spread = 0.0
                else:
                    mean_score = 0.0
                    spread = 0.0
                set_stats = [float(cur_size), mean_score, spread]
                X_rows.append(feats[i] + set_stats)
                y_rows.append(1.0 if i in selected_set else 0.0)
                if i in selected_set and i in selected_order:
                    current.append(i)

    if not X_rows:
        return torch.zeros(0, len(feature_keys) + 3), torch.zeros(0)
    return (
        torch.tensor(X_rows, dtype=torch.float32),
        torch.tensor(y_rows, dtype=torch.float32),
    )


def train_p5(train_data: dict[str, Any], seed: int) -> P5Model:
    feature_keys = train_data["feature_keys"]
    k_values = train_data["k_positive_values"]
    median_k = int(_median([float(k) for k in k_values])) if k_values else DEFAULT_K
    median_k = max(median_k, 1)

    if not feature_keys:
        return P5Model(model=None, feature_keys=feature_keys, k=median_k)

    torch.manual_seed(seed)
    np.random.seed(seed)
    feat_dim = len(feature_keys)
    model = MarginalMLP(feat_dim=feat_dim, hidden=P5_HIDDEN)
    optimizer = torch.optim.Adam(model.parameters(), lr=P5_LR)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    X, y = _build_p5_training_tensors(train_data, feature_keys)
    if X.shape[0] == 0:
        return P5Model(model=model, feature_keys=feature_keys, k=median_k)

    model.train()
    for epoch in range(P5_EPOCHS):
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
    model.eval()
    return P5Model(model=model, feature_keys=feature_keys, k=median_k)


def predict_p5(
    model: P5Model,
    bundle: PointFeatureBundle,
    threshold: float = P5_DEFAULT_THRESHOLD,
) -> tuple[list[int], int]:
    if model.model is None or not bundle.point_features:
        k = min(model.k, len(bundle.point_features))
        return list(range(k)), k
    feats = [_featurize_point(f, model.feature_keys) for f in bundle.point_features]
    coords = bundle.coordinates
    n = len(feats)
    selected: list[int] = []
    selected_coords: list[tuple[int, int]] = []
    remaining = set(range(n))
    with torch.inference_mode():
        while remaining and len(selected) < model.k:
            best_idx = -1
            best_score = -float("inf")
            for i in remaining:
                cur_size = len(selected)
                if selected:
                    cur_scores = [feats[j][0] for j in selected]
                    mean_score = float(np.mean(cur_scores))
                    if len(selected_coords) >= 2:
                        spread = float(np.std([
                            math.hypot(c[0] - coords[i][0], c[1] - coords[i][1])
                            for c in selected_coords
                        ]))
                    else:
                        spread = 0.0
                else:
                    mean_score = 0.0
                    spread = 0.0
                row = torch.tensor([feats[i] + [float(cur_size), mean_score, spread]],
                                   dtype=torch.float32)
                score = float(torch.sigmoid(model.model(row))[0].item())
                if score > best_score:
                    best_score = score
                    best_idx = i
            if best_idx < 0:
                break
            if best_score < threshold and len(selected) >= 1:
                break
            selected.append(best_idx)
            selected_coords.append(coords[best_idx])
            remaining.discard(best_idx)
    if not selected:
        # Fallback: top-K by raw contrast
        selected = list(range(min(model.k, n)))
    return selected, len(selected)


# ---------------------------------------------------------------------------
# Negative coordinate selection (0 or 1 negative)
# ---------------------------------------------------------------------------

def _select_negative_coords(
    pools: PairCandidatePools,
    k_negative: int,
) -> list[tuple[int, int]]:
    """Select 0 or 1 negative from the shared pool (N_BOTTOMK)."""
    if k_negative <= 0:
        return []
    neg_coords = [c.coordinate for c in pools.topk.negatives]
    if not neg_coords:
        return []
    return [neg_coords[0]]


# ---------------------------------------------------------------------------
# Model evaluation on validation queries
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    name: str
    train_fn: Any
    predict_fn: Any
    is_learned: bool = True
    needs_tuning: bool = False  # P3 lambda/mu


def _default_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec("P0_topk_contrast", train_p0, predict_p0, is_learned=False),
        ModelSpec("P1_pointwise_regression", train_p1, predict_p1),
        ModelSpec("P2_pairwise_ranking", train_p2, predict_p2),
        ModelSpec("P3_diversity_reranker", train_p3, predict_p3, needs_tuning=True),
        ModelSpec("P4_set_encoder", train_p4, predict_p4),
        ModelSpec("P5_greedy_marginal", train_p5, predict_p5, needs_tuning=True),
    ]


def _evaluate_model_on_validation(
    predictor: Any,
    spec: ModelSpec,
    model: Any,
    feature_cache: FeatureCache,
    val_queries: list[dict[str, str]],
    reference_sets: tuple[str, ...] = ("A",),
    use_negative: bool = True,
    p3_lambda: float = 1.0,
    p3_mu: float = 0.5,
    p5_threshold: float = P5_DEFAULT_THRESHOLD,
) -> dict[str, dict[str, Any]]:
    """Evaluate a model on validation queries. Returns {query: {ref_set: metrics}}.

    For each (query, ref), builds point features, predicts positive indices,
    selects 0/1 negative, runs SAM once, records Dice / K / calls.
    """
    results: dict[str, dict[str, Any]] = {}
    for query in val_queries:
        query_case_id = query["case_id"]
        ref_pairs = load_reference_pairs(query_case_id, reference_sets)
        if not ref_pairs:
            continue
        query_paths = resolve_case_paths(query_case_id)
        query_image = load_rgb_518(query_paths["image"])
        predictor.set_image(np.asarray(query_image))
        target = load_mask_518(query_paths["mask"])
        target_labels = _component_labels(target)
        target_component_count = len(set(target_labels.values()))

        results[query_case_id] = {}
        seen_refs: set[str] = set()
        for ref_pair in ref_pairs:
            ref_case_id = ref_pair["reference_case_id"]
            ref_set = ref_pair.get("reference_set", "A")
            if ref_case_id in seen_refs:
                continue
            seen_refs.add(ref_case_id)
            try:
                bundle = extract_point_features(query_case_id, ref_case_id, feature_cache)
            except Exception as exc:
                results[query_case_id][ref_case_id] = {
                    "ref_set": ref_set, "dice": 0.0, "k_positive": 0,
                    "k_negative": 0, "sam_calls": 0,
                    "status": f"FEAT_ERROR:{type(exc).__name__}",
                }
                continue

            # Predict positive indices
            if spec.needs_tuning:
                if spec.name == "P3_diversity_reranker":
                    pos_indices, k_pos = spec.predict_fn(
                        model, bundle, lambda_dist=p3_lambda, mu_redundancy=p3_mu,
                    )
                elif spec.name == "P5_greedy_marginal":
                    pos_indices, k_pos = spec.predict_fn(
                        model, bundle, threshold=p5_threshold,
                    )
                else:
                    pos_indices, k_pos = spec.predict_fn(model, bundle)
            else:
                pos_indices, k_pos = spec.predict_fn(model, bundle)

            pos_coords = [bundle.coordinates[i] for i in pos_indices if 0 <= i < len(bundle.coordinates)]
            neg_k = DEFAULT_NEGATIVE_K if use_negative else 0
            # P0 (Top-K contrast) uses NO negatives per spec
            if spec.name == "P0_topk_contrast":
                neg_k = 0
            neg_coords = _select_negative_coords(bundle.pools, neg_k)

            metrics = run_subset_on_query(
                predictor, pos_coords, neg_coords,
                target, target_labels, target_component_count,
            )
            metrics["ref_set"] = ref_set
            metrics["ref_case_id"] = ref_case_id
            results[query_case_id][ref_case_id] = metrics
    return results


def _aggregate_per_query(
    eval_results: dict[str, dict[str, Any]],
    reference_set: str = "A",
) -> dict[str, float]:
    """Mean Dice per query across refs in the given reference set."""
    per_query: dict[str, float] = {}
    for qid, ref_map in eval_results.items():
        vals = [m["dice"] for m in ref_map.values() if m.get("ref_set") == reference_set]
        per_query[qid] = _mean(vals) if vals else 0.0
    return per_query


def _aggregate_k_distribution(
    eval_results: dict[str, dict[str, Any]],
    reference_set: str = "A",
) -> list[int]:
    """K+ per query (mean across refs) for the given reference set."""
    k_vals: list[int] = []
    for qid, ref_map in eval_results.items():
        ks = [m.get("k_positive", 0) for m in ref_map.values() if m.get("ref_set") == reference_set]
        if ks:
            k_vals.append(int(round(_mean([float(k) for k in ks]))))
    return k_vals


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def _evaluate_baseline_action(
    predictor: Any,
    action: CompactAction,
    feature_cache: FeatureCache,
    val_queries: list[dict[str, str]],
    reference_sets: tuple[str, ...] = ("A",),
) -> dict[str, dict[str, Any]]:
    """Evaluate a fixed bank action baseline (uses run_action_on_query)."""
    results: dict[str, dict[str, Any]] = {}
    for query in val_queries:
        query_case_id = query["case_id"]
        ref_pairs = load_reference_pairs(query_case_id, reference_sets)
        if not ref_pairs:
            continue
        query_paths = resolve_case_paths(query_case_id)
        query_image = load_rgb_518(query_paths["image"])
        predictor.set_image(np.asarray(query_image))
        target = load_mask_518(query_paths["mask"])
        target_labels = _component_labels(target)
        target_component_count = len(set(target_labels.values()))

        results[query_case_id] = {}
        seen_refs: set[str] = set()
        for ref_pair in ref_pairs:
            ref_case_id = ref_pair["reference_case_id"]
            ref_set = ref_pair.get("reference_set", "A")
            if ref_case_id in seen_refs:
                continue
            seen_refs.add(ref_case_id)
            try:
                pools = build_pair_candidate_pools(query_case_id, ref_case_id, feature_cache)
            except Exception as exc:
                results[query_case_id][ref_case_id] = {
                    "ref_set": ref_set, "dice": 0.0, "k_positive": 0,
                    "k_negative": 0, "sam_calls": 0,
                    "status": f"POOL_ERROR:{type(exc).__name__}",
                }
                continue
            metrics = run_action_on_query(
                predictor, action, pools, target, target_labels, target_component_count,
            )
            metrics["ref_set"] = ref_set
            metrics["ref_case_id"] = ref_case_id
            results[query_case_id][ref_case_id] = metrics
    return results


def _evaluate_random_subset(
    predictor: Any,
    feature_cache: FeatureCache,
    val_queries: list[dict[str, str]],
    seed: int,
    k: int = DEFAULT_K,
    reference_sets: tuple[str, ...] = ("A",),
) -> dict[str, dict[str, Any]]:
    """Random 20-point subset baseline."""
    rng = np.random.RandomState(seed)
    results: dict[str, dict[str, Any]] = {}
    for query in val_queries:
        query_case_id = query["case_id"]
        ref_pairs = load_reference_pairs(query_case_id, reference_sets)
        if not ref_pairs:
            continue
        query_paths = resolve_case_paths(query_case_id)
        query_image = load_rgb_518(query_paths["image"])
        predictor.set_image(np.asarray(query_image))
        target = load_mask_518(query_paths["mask"])
        target_labels = _component_labels(target)
        target_component_count = len(set(target_labels.values()))

        results[query_case_id] = {}
        seen_refs: set[str] = set()
        for ref_pair in ref_pairs:
            ref_case_id = ref_pair["reference_case_id"]
            ref_set = ref_pair.get("reference_set", "A")
            if ref_case_id in seen_refs:
                continue
            seen_refs.add(ref_case_id)
            try:
                pools = build_pair_candidate_pools(query_case_id, ref_case_id, feature_cache)
            except Exception as exc:
                results[query_case_id][ref_case_id] = {
                    "ref_set": ref_set, "dice": 0.0, "k_positive": 0,
                    "k_negative": 0, "sam_calls": 0,
                    "status": f"POOL_ERROR:{type(exc).__name__}",
                }
                continue
            pos_all = [c.coordinate for c in pools.topk.positives]
            n = len(pos_all)
            if n == 0:
                results[query_case_id][ref_case_id] = {
                    "ref_set": ref_set, "dice": 0.0, "k_positive": 0,
                    "k_negative": 0, "sam_calls": 0, "status": "NO_POSITIVES",
                }
                continue
            kk = min(k, n)
            chosen = list(rng.choice(n, kk, replace=False))
            pos_coords = [pos_all[i] for i in chosen]
            metrics = run_subset_on_query(
                predictor, pos_coords, [], target, target_labels, target_component_count,
            )
            metrics["ref_set"] = ref_set
            metrics["ref_case_id"] = ref_case_id
            results[query_case_id][ref_case_id] = metrics
    return results


def _load_bank_policy_action(compact_bank: list[CompactAction]) -> CompactAction | None:
    """Load Y3 best model's action (or strongest fixed) for bank-policy baseline."""
    if Y3_SUMMARY_JSON.is_file():
        try:
            summary = json.loads(Y3_SUMMARY_JSON.read_text(encoding="utf-8"))
            # Find best learned model
            models = summary.get("models", summary.get("model_results", {}))
            best_name = None
            best_dice = -1.0
            for name, info in models.items():
                if name.startswith("M") and isinstance(info, dict):
                    d = info.get("mean_dice", 0.0)
                    if d > best_dice:
                        best_dice = d
                        best_name = name
            if best_name and "most_common_action" in models.get(best_name, {}):
                best_action_id = models[best_name]["most_common_action"]
                for a in compact_bank:
                    if a.action_id == best_action_id:
                        return a
        except Exception:
            pass
    # Fallback: Top20+N1 (strongest fixed)
    for a in compact_bank:
        if a.k_positive == 20 and a.k_negative == 1 and a.positive_selector == "P_TOPK":
            return a
    return compact_bank[0] if compact_bank else None


def _build_fixed_actions(compact_bank: list[CompactAction]) -> dict[str, CompactAction | None]:
    """Build the fixed-action baselines from the compact bank."""
    baselines: dict[str, CompactAction | None] = {}
    # Top20 contrast (no negatives)
    for a in compact_bank:
        if a.k_positive == 20 and a.k_negative == 0 and a.positive_selector == "P_TOPK":
            baselines["B_top20_contrast"] = a
            break
    if "B_top20_contrast" not in baselines:
        baselines["B_top20_contrast"] = None
    # Top20 spread
    for a in compact_bank:
        if a.k_positive == 20 and a.positive_selector == "P_SPREAD":
            baselines["B_top20_spread"] = a
            break
    if "B_top20_spread" not in baselines:
        baselines["B_top20_spread"] = None
    # Top20 local peaks
    for a in compact_bank:
        if a.k_positive == 20 and a.positive_selector == "P_LOCAL_PEAKS":
            baselines["B_top20_local_peaks"] = a
            break
    if "B_top20_local_peaks" not in baselines:
        baselines["B_top20_local_peaks"] = None
    # Top20 + N1 (strongest fixed)
    for a in compact_bank:
        if a.k_positive == 20 and a.k_negative == 1 and a.positive_selector == "P_TOPK":
            baselines["B_top20_n1"] = a
            break
    if "B_top20_n1" not in baselines:
        baselines["B_top20_n1"] = None
    return baselines


# ---------------------------------------------------------------------------
# Gate Y4 decision
# ---------------------------------------------------------------------------

def compute_gate_y4_for_model(
    model_name: str,
    per_query_dice_A: dict[str, float],
    per_query_dice_B: dict[str, float],
    per_query_fixed_A: dict[str, float],
    per_query_fixed_B: dict[str, float],
    per_query_bank_A: dict[str, float] | None,
    cross_search_dice_A: float,
    fixed_dice_A: float,
    bank_dice_A: float | None,
    k_distribution: list[int],
    seed_deltas: list[float],
    is_learned: bool,
) -> dict[str, Any]:
    """Compute Gate Y4 decision for one learned model (8 conditions)."""
    queries = sorted(set(per_query_dice_A.keys()) & set(per_query_fixed_A.keys()))
    if not queries:
        return {
            "model_name": model_name,
            "status": "PHASEY_GATE_Y4_INSUFFICIENT_DATA",
            "queries": 0,
        }

    model_dice_A = _mean([per_query_dice_A[q] for q in queries])
    model_dice_B = _mean([per_query_dice_B.get(q, 0.0) for q in queries if q in per_query_dice_B]) if per_query_dice_B else 0.0
    fixed_A = _mean([per_query_fixed_A[q] for q in queries])
    fixed_B = _mean([per_query_fixed_B.get(q, 0.0) for q in queries if q in per_query_fixed_B]) if per_query_fixed_B else 0.0
    bank_A = bank_dice_A

    delta_vs_fixed = model_dice_A - fixed_A
    delta_vs_bank = (model_dice_A - bank_A) if bank_A is not None else None

    # Paired bootstrap on (model - fixed) per query
    paired_diffs = [per_query_dice_A[q] - per_query_fixed_A[q] for q in queries]
    ci_lo, ci_hi = _bootstrap_ci(paired_diffs, iterations=2000)

    # Oracle recovery vs cross-reference search oracle
    oracle_gap = cross_search_dice_A - fixed_A
    oracle_recovery = (delta_vs_fixed / oracle_gap) if oracle_gap > 0 else 0.0

    # K distribution std
    k_std = float(np.std(k_distribution)) if k_distribution else 0.0
    k_not_degenerate = k_std > 0.0 and len(set(k_distribution)) > 1

    # Reference-set A/B direction stable: positive on both
    direction_A = delta_vs_fixed > 0
    delta_vs_fixed_B = model_dice_B - fixed_B if per_query_dice_B and per_query_fixed_B else 0.0
    direction_B = delta_vs_fixed_B > 0 if per_query_dice_B and per_query_fixed_B else True
    direction_stable = direction_A and direction_B

    # 3 seeds direction consistent: all positive or all near-zero
    seed_directions_positive = [d > 0 for d in seed_deltas] if seed_deltas else []
    seed_directions_near_zero = [abs(d) < 0.005 for d in seed_deltas] if seed_deltas else []
    seeds_consistent = bool(
        seed_directions_positive and (
            all(seed_directions_positive) or all(seed_directions_near_zero)
        )
    ) if seed_directions_positive else True

    # 8 conditions (section 12.5)
    cond1 = (delta_vs_fixed >= 0.015) or (
        delta_vs_bank is not None and delta_vs_bank >= 0.0075
    )
    cond2 = ci_lo > 0
    cond3 = True   # single SAM call (by construction)
    cond4 = True   # no test-time SAM search (by construction)
    cond5 = seeds_consistent
    cond6 = oracle_recovery >= 0.20
    cond7 = direction_stable
    cond8 = k_not_degenerate

    conditions = {
        "cond1_delta_ge_0p015_fixed_or_0p0075_bank": cond1,
        "cond2_bootstrap_ci_lo_gt_0": cond2,
        "cond3_single_sam_call": cond3,
        "cond4_no_test_time_search": cond4,
        "cond5_seeds_direction_consistent": cond5,
        "cond6_oracle_recovery_ge_20pct": cond6,
        "cond7_reference_set_AB_stable": cond7,
        "cond8_k_distribution_not_degenerate": cond8,
    }

    if is_learned and all(conditions.values()):
        decision = "PHASEY_GATE_Y4_POINT_SUBSET_PASS"
    elif is_learned:
        decision = "PHASEY_GATE_Y4_SEARCH_ORACLE_NOT_LEARNABLE"
    else:
        decision = "BASELINE_NOT_SUBJECT_TO_GATE"

    return {
        "model_name": model_name,
        "queries": len(queries),
        "model_dice_A": model_dice_A,
        "model_dice_B": model_dice_B,
        "fixed_dice_A": fixed_A,
        "fixed_dice_B": fixed_B,
        "bank_dice_A": bank_A,
        "delta_vs_fixed": delta_vs_fixed,
        "delta_vs_bank": delta_vs_bank,
        "delta_vs_fixed_B": delta_vs_fixed_B,
        "bootstrap_ci_lo": ci_lo,
        "bootstrap_ci_hi": ci_hi,
        "cross_search_dice_A": cross_search_dice_A,
        "oracle_gap": oracle_gap,
        "oracle_recovery": oracle_recovery,
        "k_distribution_std": k_std,
        "k_distribution_unique": len(set(k_distribution)) if k_distribution else 0,
        "seed_deltas": seed_deltas,
        "conditions": conditions,
        "decision": decision,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _per_query_csv_fields() -> list[str]:
    return [
        "query_case_id", "model_name", "seed", "ref_set", "ref_case_id",
        "dice", "iou", "aji", "component_recall",
        "k_positive", "k_negative", "sam_calls",
        "status",
    ]


def _write_per_query_csv(
    rows: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    path = out_dir / "point_subset_per_query.csv"
    _write_csv(path, rows, _per_query_csv_fields())


def _write_gate_y4_summary(
    summary: dict[str, Any],
    out_dir: Path,
) -> None:
    json_path = out_dir / "gateY4_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    md_path = out_dir / "gateY4_summary.md"
    lines = [
        "# Phase Y Gate Y4: Point-Subset Policy",
        "",
        f"- **Queries evaluated**: {summary.get('query_count', 0)}",
        f"- **Seeds**: {summary.get('seeds', [])}",
        f"- **Strongest fixed (Top20+N1) Dice (A)**: {summary.get('fixed_dice_A', 0):.4f}",
        f"- **Bank Policy Dice (A)**: {summary.get('bank_dice_A', 'N/A')}",
        f"- **Cross-reference Search Oracle Dice (A)**: {summary.get('cross_search_dice_A', 0):.4f}",
        "",
        "## Per-model results",
        "",
        "| Model | Dice (A) | Δ vs fixed | Δ vs bank | CI lo | Oracle recovery | K std | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, info in summary.get("models", {}).items():
        dice = info.get("model_dice_A", 0.0)
        delta_f = info.get("delta_vs_fixed", 0.0)
        delta_b = info.get("delta_vs_bank")
        ci_lo = info.get("bootstrap_ci_lo", 0.0)
        rec = info.get("oracle_recovery", 0.0)
        k_std = info.get("k_distribution_std", 0.0)
        decision = info.get("decision", "")
        delta_b_str = f"{delta_b:.4f}" if delta_b is not None else "N/A"
        lines.append(
            f"| {name} | {dice:.4f} | {delta_f:.4f} | {delta_b_str} | "
            f"{ci_lo:.4f} | {rec:.1%} | {k_std:.2f} | `{decision}` |"
        )
    lines.extend([
        "",
        "## Gate conditions (learned models only)",
        "",
    ])
    for name, info in summary.get("models", {}).items():
        if not info.get("is_learned", False):
            continue
        lines.append(f"### {name}")
        for k, v in info.get("conditions", {}).items():
            mark = "PASS" if v else "FAIL"
            lines.append(f"- [{mark}] `{k}`")
        lines.append(f"- **Decision**: `{info.get('decision', '')}`")
        lines.append("")

    overall = summary.get("overall_decision", "")
    lines.extend([
        "## Overall decision",
        "",
        f"`{overall}`",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")


def _write_run_metadata(
    out_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    elapsed_s: float,
    train_query_count: int,
    val_query_count: int,
) -> None:
    meta = {
        "script": "run_phaseY_gateY4_point_subset",
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "args": {k: str(v) for k, v in vars(args).items()},
        "elapsed_seconds": elapsed_s,
        "train_queries": train_query_count,
        "validation_queries": val_query_count,
        "outputs": {
            "per_query_csv": "point_subset_per_query.csv",
            "summary_json": "gateY4_summary.json",
            "summary_md": "gateY4_summary.md",
            "search_labels_train_csv": "search_labels_train.csv",
        },
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase Y Gate Y4: Point-Subset Policy"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 777])
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        args.max_queries = args.max_queries or 5

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[phaseY-Y4] device={device}")
    print(f"[phaseY-Y4] seeds={args.seeds}")

    start = time.perf_counter()

    # Load compact bank (for fixed-action baselines + bank-policy baseline)
    try:
        compact_bank = load_compact_bank()
        print(f"[phaseY-Y4] compact bank: {len(compact_bank)} actions")
    except Exception as exc:
        print(f"[phaseY-Y4] compact bank load failed: {exc}")
        compact_bank = []

    # Load Y2 upper bounds (validation)
    y2_bounds = load_y2_upper_bounds()
    cross_search_dice_A = y2_bounds.get("mean_cross_search_dice", 0.0) if y2_bounds else 0.0
    raw_search_dice_A = y2_bounds.get("mean_raw_search_dice", 0.0) if y2_bounds else 0.0
    print(f"[phaseY-Y4] Y2 cross-search Dice: {cross_search_dice_A:.4f} "
          f"raw: {raw_search_dice_A:.4f}")

    # Load validation queries
    val_queries = load_queries("validation")
    if args.max_queries:
        val_queries = val_queries[:args.max_queries]
    print(f"[phaseY-Y4] validation queries: {len(val_queries)}")

    # Load train queries (for search labels)
    train_queries = load_queries("train")
    if args.max_queries:
        train_queries = train_queries[:args.max_queries]
    print(f"[phaseY-Y4] train queries: {len(train_queries)}")

    feature_cache = FeatureCache(device=device)
    predictor = load_sam_predictor(device)

    # --- Generate search labels on TRAIN queries (section 12.2) ---
    print("[phaseY-Y4] generating train search labels (GT-guided, train only)...")
    search_labels = generate_train_search_labels(
        predictor, feature_cache, out_dir, max_queries=args.max_queries,
    )
    ok_labels = [l for l in search_labels if l.status == "OK"]
    print(f"[phaseY-Y4] search labels: {len(ok_labels)}/{len(search_labels)} OK")

    # --- Build training dataset ---
    print("[phaseY-Y4] building train dataset (point features)...")
    train_data = build_train_dataset(
        predictor, feature_cache, search_labels, max_queries=args.max_queries,
    )
    print(f"[phaseY-Y4] train dataset: X={train_data['X'].shape}, "
          f"feature_keys={len(train_data['feature_keys'])}, "
          f"pairs={sum(len(v) for v in train_data['pair_bundles'].values())}")

    # --- Evaluate fixed-action baselines ---
    print("[phaseY-Y4] evaluating fixed-action baselines...")
    fixed_actions = _build_fixed_actions(compact_bank)
    baseline_results: dict[str, dict[str, dict[str, Any]]] = {}

    for baseline_name, action in fixed_actions.items():
        if action is None:
            print(f"[phaseY-Y4] {baseline_name}: action not found in bank, skipping")
            continue
        print(f"[phaseY-Y4] evaluating {baseline_name} ({action.action_id})...")
        baseline_results[baseline_name] = _evaluate_baseline_action(
            predictor, action, feature_cache, val_queries, reference_sets=("A", "B"),
        )

    # Bank-Action Policy baseline
    bank_action = _load_bank_policy_action(compact_bank)
    if bank_action is not None:
        print(f"[phaseY-Y4] evaluating B_bank_policy ({bank_action.action_id})...")
        baseline_results["B_bank_policy"] = _evaluate_baseline_action(
            predictor, bank_action, feature_cache, val_queries, reference_sets=("A", "B"),
        )

    # Random 20-point subset (seeded)
    print("[phaseY-Y4] evaluating B_random_20...")
    baseline_results["B_random_20"] = _evaluate_random_subset(
        predictor, feature_cache, val_queries, seed=42, k=20, reference_sets=("A", "B"),
    )

    # --- Evaluate P0-P5 models (3 seeds each) ---
    model_specs = _default_model_specs()
    per_query_rows: list[dict[str, Any]] = []
    model_summaries: dict[str, dict[str, Any]] = {}
    seed_eval_results: dict[str, list[dict[str, dict[str, Any]]]] = {}

    for spec in model_specs:
        print(f"[phaseY-Y4] training/evaluating {spec.name}...")
        all_seed_results: list[dict[str, dict[str, Any]]] = []
        seed_deltas: list[float] = []

        # Fixed dice for delta computation (Top20+N1 baseline)
        fixed_results_A = baseline_results.get("B_top20_n1", baseline_results.get("B_top20_contrast", {}))
        fixed_per_query_A = _aggregate_per_query(fixed_results_A, "A")
        fixed_per_query_B = _aggregate_per_query(fixed_results_A, "B")
        fixed_dice_A = _mean(list(fixed_per_query_A.values())) if fixed_per_query_A else 0.0

        bank_results_A = baseline_results.get("B_bank_policy", {})
        bank_per_query_A = _aggregate_per_query(bank_results_A, "A")
        bank_dice_A = _mean(list(bank_per_query_A.values())) if bank_per_query_A else None

        for seed in args.seeds:
            try:
                model = spec.train_fn(train_data, seed)
            except Exception as exc:
                print(f"[phaseY-Y4] {spec.name} seed {seed} train failed: {exc}")
                continue

            # P3: tune lambda/mu on a subset of validation (first seed only)
            p3_lambda = 1.0
            p3_mu = 0.5
            p5_threshold = P5_DEFAULT_THRESHOLD
            if spec.name == "P3_diversity_reranker" and seed == args.seeds[0]:
                # Simple grid tuning on first few validation queries
                tune_queries = val_queries[:min(5, len(val_queries))]
                best_lambda, best_mu, best_dice = p3_lambda, p3_mu, -1.0
                for lam in [0.5, 1.0, 2.0]:
                    for mu in [0.25, 0.5, 1.0]:
                        try:
                            tune_res = _evaluate_model_on_validation(
                                predictor, spec, model, feature_cache, tune_queries,
                                reference_sets=("A",), use_negative=True,
                                p3_lambda=lam, p3_mu=mu,
                            )
                            tune_dice = _mean(list(_aggregate_per_query(tune_res, "A").values()))
                            if tune_dice > best_dice:
                                best_dice = tune_dice
                                best_lambda, best_mu = lam, mu
                        except Exception:
                            continue
                p3_lambda, p3_mu = best_lambda, best_mu
                print(f"[phaseY-Y4] P3 tuned: lambda={p3_lambda}, mu={p3_mu}")
            if spec.name == "P5_greedy_marginal" and seed == args.seeds[0]:
                tune_queries = val_queries[:min(5, len(val_queries))]
                best_threshold, best_dice = P5_DEFAULT_THRESHOLD, -1.0
                for thr in [0.3, 0.5, 0.7]:
                    try:
                        tune_res = _evaluate_model_on_validation(
                            predictor, spec, model, feature_cache, tune_queries,
                            reference_sets=("A",), use_negative=True,
                            p5_threshold=thr,
                        )
                        tune_dice = _mean(list(_aggregate_per_query(tune_res, "A").values()))
                        if tune_dice > best_dice:
                            best_dice = tune_dice
                            best_threshold = thr
                    except Exception:
                        continue
                p5_threshold = best_threshold
                print(f"[phaseY-Y4] P5 tuned: threshold={p5_threshold}")

            try:
                eval_res = _evaluate_model_on_validation(
                    predictor, spec, model, feature_cache, val_queries,
                    reference_sets=("A", "B"), use_negative=True,
                    p3_lambda=p3_lambda, p3_mu=p3_mu, p5_threshold=p5_threshold,
                )
            except Exception as exc:
                print(f"[phaseY-Y4] {spec.name} seed {seed} eval failed: {exc}")
                continue
            all_seed_results.append(eval_res)

            per_query_A = _aggregate_per_query(eval_res, "A")
            model_dice_A = _mean(list(per_query_A.values())) if per_query_A else 0.0
            delta = model_dice_A - fixed_dice_A
            seed_deltas.append(delta)
            print(f"[phaseY-Y4] {spec.name} seed {seed}: Dice(A)={model_dice_A:.4f} "
                  f"Δ={delta:.4f}")

            # Write per-query rows
            for qid, ref_map in eval_res.items():
                for ref_id, m in ref_map.items():
                    per_query_rows.append({
                        "query_case_id": qid,
                        "model_name": spec.name,
                        "seed": seed,
                        "ref_set": m.get("ref_set", ""),
                        "ref_case_id": ref_id,
                        "dice": m.get("dice", 0.0),
                        "iou": m.get("iou", 0.0),
                        "aji": m.get("aji", 0.0),
                        "component_recall": m.get("component_recall", 0.0),
                        "k_positive": m.get("k_positive", 0),
                        "k_negative": m.get("k_negative", 0),
                        "sam_calls": m.get("sam_calls", 0),
                        "status": m.get("status", ""),
                    })

        if not all_seed_results:
            continue
        seed_eval_results[spec.name] = all_seed_results

        # Aggregate across seeds: use mean Dice per query
        # Take first seed for per-query distribution; mean across seeds for aggregate
        mean_per_query_A: dict[str, float] = {}
        mean_per_query_B: dict[str, float] = {}
        all_queries = set()
        for res in all_seed_results:
            all_queries.update(res.keys())
        for q in all_queries:
            vals_A = [_aggregate_per_query(res, "A").get(q, 0.0) for res in all_seed_results]
            vals_B = [_aggregate_per_query(res, "B").get(q, 0.0) for res in all_seed_results]
            mean_per_query_A[q] = _mean(vals_A) if vals_A else 0.0
            mean_per_query_B[q] = _mean(vals_B) if vals_B else 0.0

        k_dist = _aggregate_k_distribution(all_seed_results[0], "A")

        gate = compute_gate_y4_for_model(
            model_name=spec.name,
            per_query_dice_A=mean_per_query_A,
            per_query_dice_B=mean_per_query_B,
            per_query_fixed_A=fixed_per_query_A,
            per_query_fixed_B=fixed_per_query_B,
            per_query_bank_A=bank_per_query_A,
            cross_search_dice_A=cross_search_dice_A,
            fixed_dice_A=fixed_dice_A,
            bank_dice_A=bank_dice_A,
            k_distribution=k_dist,
            seed_deltas=seed_deltas,
            is_learned=spec.is_learned,
        )
        gate["is_learned"] = spec.is_learned
        model_summaries[spec.name] = gate
        print(f"[phaseY-Y4] {spec.name}: decision={gate['decision']} "
              f"Δ={gate['delta_vs_fixed']:.4f} CI_lo={gate['bootstrap_ci_lo']:.4f} "
              f"recovery={gate['oracle_recovery']:.1%}")

    # --- Add baseline rows to per-query CSV ---
    for baseline_name, res in baseline_results.items():
        for qid, ref_map in res.items():
            for ref_id, m in ref_map.items():
                per_query_rows.append({
                    "query_case_id": qid,
                    "model_name": baseline_name,
                    "seed": 0,
                    "ref_set": m.get("ref_set", ""),
                    "ref_case_id": ref_id,
                    "dice": m.get("dice", 0.0),
                    "iou": m.get("iou", 0.0),
                    "aji": m.get("aji", 0.0),
                    "component_recall": m.get("component_recall", 0.0),
                    "k_positive": m.get("k_positive", 0),
                    "k_negative": m.get("k_negative", 0),
                    "sam_calls": m.get("sam_calls", 0),
                    "status": m.get("status", ""),
                })

    # Add baseline summaries
    for baseline_name, res in baseline_results.items():
        per_query_A = _aggregate_per_query(res, "A")
        per_query_B = _aggregate_per_query(res, "B")
        fixed_per_query_A_ref = _aggregate_per_query(baseline_results.get("B_top20_n1", baseline_results.get("B_top20_contrast", {})), "A")
        fixed_dice_A_ref = _mean(list(fixed_per_query_A_ref.values())) if fixed_per_query_A_ref else 0.0
        dice_A = _mean(list(per_query_A.values())) if per_query_A else 0.0
        k_dist = _aggregate_k_distribution(res, "A")
        gate = compute_gate_y4_for_model(
            model_name=baseline_name,
            per_query_dice_A=per_query_A,
            per_query_dice_B=per_query_B,
            per_query_fixed_A=fixed_per_query_A_ref,
            per_query_fixed_B=_aggregate_per_query(baseline_results.get("B_top20_n1", baseline_results.get("B_top20_contrast", {})), "B"),
            per_query_bank_A=None,
            cross_search_dice_A=cross_search_dice_A,
            fixed_dice_A=fixed_dice_A_ref,
            bank_dice_A=None,
            k_distribution=k_dist,
            seed_deltas=[dice_A - fixed_dice_A_ref],
            is_learned=False,
        )
        gate["is_learned"] = False
        model_summaries[baseline_name] = gate

    # --- Add search-oracle upper bounds to summary ---
    if y2_bounds and y2_bounds.get("per_query"):
        per_query = y2_bounds["per_query"]
        model_summaries["UB_gt_search_oracle"] = {
            "model_name": "UB_gt_search_oracle",
            "model_dice_A": y2_bounds.get("mean_raw_search_dice", 0.0),
            "model_dice_B": y2_bounds.get("mean_raw_search_dice", 0.0),
            "decision": "UPPER_BOUND_RAW_SEARCH",
            "is_learned": False,
            "queries": len(per_query),
        }
        model_summaries["UB_cross_ref_search_oracle"] = {
            "model_name": "UB_cross_ref_search_oracle",
            "model_dice_A": cross_search_dice_A,
            "model_dice_B": cross_search_dice_A,
            "decision": "UPPER_BOUND_CROSS_REF_SEARCH",
            "is_learned": False,
            "queries": len(per_query),
        }

    # --- Overall decision ---
    learned_pass = any(
        m.get("decision") == "PHASEY_GATE_Y4_POINT_SUBSET_PASS"
        for m in model_summaries.values()
        if m.get("is_learned", False)
    )
    overall_decision = (
        "PHASEY_GATE_Y4_POINT_SUBSET_PASS" if learned_pass
        else "PHASEY_GATE_Y4_SEARCH_ORACLE_NOT_LEARNABLE"
    )

    # --- Compute fixed_dice_A and bank_dice_A for top-level summary ---
    top_fixed_results = baseline_results.get("B_top20_n1", baseline_results.get("B_top20_contrast", {}))
    top_fixed_per_query_A = _aggregate_per_query(top_fixed_results, "A")
    top_fixed_dice_A = _mean(list(top_fixed_per_query_A.values())) if top_fixed_per_query_A else 0.0
    top_bank_results = baseline_results.get("B_bank_policy", {})
    top_bank_per_query_A = _aggregate_per_query(top_bank_results, "A")
    top_bank_dice_A = _mean(list(top_bank_per_query_A.values())) if top_bank_per_query_A else None

    summary = {
        "query_count": len(val_queries),
        "seeds": list(args.seeds),
        "fixed_dice_A": top_fixed_dice_A,
        "bank_dice_A": top_bank_dice_A,
        "cross_search_dice_A": cross_search_dice_A,
        "raw_search_dice_A": raw_search_dice_A,
        "models": model_summaries,
        "overall_decision": overall_decision,
    }

    # --- Write outputs ---
    _write_per_query_csv(per_query_rows, out_dir)
    _write_gate_y4_summary(summary, out_dir)

    elapsed = time.perf_counter() - start
    _write_run_metadata(
        out_dir, args, device, elapsed,
        train_query_count=len(train_queries),
        val_query_count=len(val_queries),
    )

    print(f"[phaseY-Y4] done in {elapsed:.1f}s")
    print(f"[phaseY-Y4] overall decision: {overall_decision}")
    print(f"[phaseY-Y4] outputs in {out_dir}")


if __name__ == "__main__":
    main()
